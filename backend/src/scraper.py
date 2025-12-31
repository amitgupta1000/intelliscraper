import time, asyncio, os, sys, json
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from .logging_setup import logger
import aiohttp
from .utils import html_to_markdown, save_text_to_gcs, clean_extracted_text, safe_format_url

# Prefer the minimal config for scraper-only deployments; fall back to sensible defaults
try:
    from .config_minimal import SCRAPE_RESULT_TTL, AIOHTTP_REQUEST_TIMEOUT, MAX_CONCURRENT_SCRAPES
    CACHE_ENABLED = False
    CACHE_TTL = int(getattr(__import__('backend.src.config_minimal', fromlist=['SCRAPE_RESULT_TTL']), 'SCRAPE_RESULT_TTL', SCRAPE_RESULT_TTL) or 86400)
    MIN_EXTRACT_LENGTH = 400
except Exception:
    CACHE_ENABLED, CACHE_TTL, MIN_EXTRACT_LENGTH = True, 86400, 400

@dataclass
class ScrapedContent:
    url: str
    title: str
    text: str
    html: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    scrape_time: float = 0.0
    strategy: Optional[str] = None
    def is_successful(self) -> bool: return self.success
    def to_dict(self) -> Dict[str, Any]: return self.__dict__

class SimpleCache:
    def __init__(self, ttl: int = 3600):
        self._cache, self._timestamps, self._ttl = {}, {}, ttl
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache and time.time() - self._timestamps.get(key, 0) < self._ttl:
            return self._cache[key]
        return None
    def set(self, key: str, value: Any):
        self._cache[key], self._timestamps[key] = value, time.time()

class Scraper:
    def __init__(self, cache_enabled: bool = CACHE_ENABLED, cache_ttl: int = CACHE_TTL):
        self._cache_enabled = cache_enabled
        self.cache = SimpleCache(ttl=cache_ttl) if cache_enabled else None
        self._playwright = None
        self._browser = None
        self._start_lock = asyncio.Lock()
        # Load runtime knobs from config when available so Cloud Run can tune these
        try:
            from .config_minimal import MAX_CONCURRENT_SCRAPES, AIOHTTP_REQUEST_TIMEOUT
            PLAYWRIGHT_MAX_TABS = int(MAX_CONCURRENT_SCRAPES or 3)
            AIOHTTP_RETRIES = 1
            URL_TIMEOUT = int(AIOHTTP_REQUEST_TIMEOUT or 60)
        except Exception:
            PLAYWRIGHT_MAX_TABS = 3
            AIOHTTP_RETRIES = 1
            URL_TIMEOUT = 60

        # GLOBAL LIMIT: number of concurrent Playwright contexts (tabs)
        self._semaphore = asyncio.Semaphore(PLAYWRIGHT_MAX_TABS or 3)
        # aiohttp retry/timeouts can be tuned via config
        self._aiohttp_retries = AIOHTTP_RETRIES if (AIOHTTP_RETRIES is not None) else 3
        self._aiohttp_timeout = URL_TIMEOUT or 30

    async def start(self):
        """Initializes the browser once."""
        async with self._start_lock:
            if self._browser: return
            from playwright.async_api import async_playwright
            logger.info("Starting Persistent Playwright Browser...")
            self._playwright = await async_playwright().start()

            try:
                launch_args = []
                if sys.platform.startswith("linux"):
                    launch_args = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
                elif sys.platform.startswith("win") or sys.platform.startswith("win32"):
                    # Minimize flags on Windows; sandbox flags are unnecessary and can be harmful.
                    launch_args = ["--disable-gpu"]
                else:
                    launch_args = ["--no-sandbox", "--disable-dev-shm-usage"]

                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=launch_args
                )
            except Exception as e:
                logger.warning(f"Chromium launch with custom args failed: {e}; retrying without custom args")
                # Attempt a conservative launch without custom flags
                try:
                    self._browser = await self._playwright.chromium.launch(headless=True)
                except Exception as e2:
                    logger.error(f"Failed to launch Playwright browser: {e2}")
                    try:
                        await self._playwright.stop()
                    except Exception:
                        pass
                    self._playwright = None
                    raise

    async def stop(self):
        """Closes browser on app shutdown. Synchronized with `start()` to avoid races."""
        async with self._start_lock:
            if self._browser:
                try:
                    await self._browser.close()
                except Exception as e:
                    logger.debug(f"Error closing browser in stop(): {e}")
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception as e:
                    logger.debug(f"Error stopping playwright in stop(): {e}")
            self._browser = self._playwright = None

    async def _scrape_with_aiohttp(self, url: str, session=None, retries: Optional[int] = None) -> ScrapedContent:
        import aiohttp
        from bs4 import BeautifulSoup
        start = time.time()
        try:
            from fake_useragent import UserAgent
            ua_gen = UserAgent()
        except Exception:
            ua_gen = None
            fallback_uas = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36",
            ]

        last_error = None
        # Use configured retries when not provided explicitly
        if retries is None:
            retries = int(getattr(self, '_aiohttp_retries', 1))
        timeout_val = int(getattr(self, '_aiohttp_timeout', 30))
        for attempt in range(retries + 1):
            ua_value = None
            if ua_gen:
                try:
                    ua_value = ua_gen.random
                except Exception:
                    ua_value = None
            if not ua_value:
                # simple rotation based on attempt index
                ua_value = fallback_uas[attempt % len(fallback_uas)]

            headers = {
                "User-Agent": ua_value,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
            }
            logger.debug(f"[aiohttp] Attempt {attempt+1} for {url} with headers: {headers}")

            # Internal session management if none provided
            local_session = False
            if session is None:
                # Use ClientTimeout for clarity
                timeout = aiohttp.ClientTimeout(total=timeout_val)
                session = aiohttp.ClientSession(headers=headers, timeout=timeout, cookie_jar=aiohttp.CookieJar())
                local_session = True
      
            try:
                # Always include headers on the per-request call so external sessions get UA rotation
                async with session.get(url, headers=headers, timeout=timeout_val) as response:
                    if response.status != 200:
                        # Retry on server errors or rate limits, otherwise fail fast
                        if response.status in (429, 500, 502, 503, 504) and attempt < retries:
                            logger.debug(f"[scraper][aiohttp] transient status {response.status} for {url}, retrying (attempt={attempt+1})")
                            last_error = f"Status {response.status}"
                            continue
                        logger.debug(f"[scraper][aiohttp] fail {url}: Status {response.status}")
                        return ScrapedContent(url=url, title=url, text="", success=False, error=f"Status {response.status}", strategy="aiohttp")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    # Remove junk to see if the remaining text is actually useful
                    for tag in soup(['script', 'style', 'nav', 'footer']): tag.decompose()
                    text = soup.get_text(separator="\n", strip=True)
                    logger.debug(f"[scraper][aiohttp] success {url}")
                    return ScrapedContent(
                        url=url, title=soup.title.string if soup.title else url,
                        text=soup.get_text(separator="\n", strip=True),
                        html=html, success=True, scrape_time=time.time() - start, strategy="aiohttp"
                    )
            except Exception as e:
                # On exception, retry if attempts remain; otherwise return failure
                logger.debug(f"[scraper][aiohttp] exception on attempt {attempt+1} for {url}: {e}")
                last_error = str(e)
                if attempt < retries:
                    continue
                return ScrapedContent(url=url, title=url, text="", success=False, error=str(e), strategy="aiohttp")
            finally:
                if local_session: await session.close()
        # If we exit loop without successful return, return last observed error
        return ScrapedContent(url=url, title=url, text="", success=False, error=last_error or "aiohttp failed", strategy="aiohttp")

    async def _scrape_with_playwright(self, url: str) -> ScrapedContent:
        # AUTO-HEALING: If browser died or was closed, restart it
        if not self._browser or not self._browser.is_connected():
            logger.warning("Browser disconnected or not started. Re-initializing...")
            await self.start()

        start = time.time()
        async with self._semaphore:
            # Retry once if browser/context was closed unexpectedly
            attempts = 0
            max_attempts = 2
            context = None
            while attempts < max_attempts:
                try:
                    # If the browser is still closed here, something is fundamentally wrong with the install
                    if not self._browser or not getattr(self._browser, 'is_connected', lambda: False)():
                        raise Exception("Playwright browser failed to initialize.")

                    context = await self._browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                    page = await context.new_page()

                    # Use a slightly more aggressive timeout for Cloud Run
                    await page.goto(url, wait_until="domcontentloaded", timeout=25000)
                    title = await page.title()
                    text = await page.evaluate("document.body.innerText")

                    logger.debug(f"[scraper][playwright] success {url}")
                    return ScrapedContent(
                        url=url, title=title, text=text, success=True, scrape_time=time.time()-start, strategy="playwright"
                    )

                except Exception as e:
                    err_str = str(e).lower()
                    logger.error(f"Playwright error for {url} (attempt={attempts+1}): {e}")

                    # If the browser/context was closed, try to restart browser and retry once
                    if any(k in err_str for k in ("closed", "target", "not opened", "disconnected")) and attempts == 0:
                        logger.warning("Playwright context/page error detected; restarting browser and retrying once...")
                        try:
                            # Force a fresh start
                            self._browser = None
                            await self.start()
                        except Exception as e2:
                            logger.debug(f"Failed to restart Playwright browser during retry: {e2}")
                            # fall through to return error
                        attempts += 1
                        # ensure any partially-open context is closed before retrying
                        if context:
                            try:
                                await context.close()
                            except Exception:
                                pass
                        context = None
                        continue

                    # For other errors or if retry already attempted, record failure
                    if "closed" in err_str or "not opened" in err_str:
                        self._browser = None
                    logger.debug(f"[scraper][playwright] exception {url}: {e}")
                    return ScrapedContent(url=url, title=url, text="", success=False, error=str(e), strategy="playwright")
                finally:
                    if context:
                        try:
                            await context.close()
                        except Exception:
                            pass

    async def _is_valid_url(self, url: str) -> bool:
        """
        Lightweight URL pre-flight checks adapted from newer scraper.
        Falls back to conservative defaults if configuration is unavailable.
        """
        if not url or not url.startswith(("http://", "https://")):
            logger.debug(f"Skipping invalid URL (scheme): {url}")
            return False

        SKIP_EXTENSIONS = [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".css",
            ".js",
            ".ico",
            ".json",
            ".xml",
            ".pdf",
        ]

        parsed = urlparse(url)
        netloc = parsed.netloc.lower() if parsed.netloc else ""

        # Skip by extension
        if any(parsed.path.lower().endswith(ext) for ext in (SKIP_EXTENSIONS or [])):
            logger.debug(f"Skipping URL by extension: {url}")
            return False

        # Skip extremely long URLs
        if len(url) > 2000:
            logger.debug(f"Skipping overly long URL: {url}")
            return False

        return True

    async def _scrape_pdf(self, url: str) -> ScrapedContent:
        """Download and extract text from a PDF URL. Uses PyMuPDFLoader when available.
        Falls back to a basic requests-based failure if loader is not installed.
        """
        import aiohttp
        import aiofiles
        import tempfile
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
        except Exception:
            PyMuPDFLoader = None

        tmp_file_path = None
        session = None
        start = time.time()
        try:
            # Asynchronously download the PDF using aiohttp
            session = aiohttp.ClientSession()
            async with session.get(url, timeout=30) as response:
                response.raise_for_status()
                content = await response.read()

            # Asynchronously write to a temporary file using aiofiles
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_file_path = tmp.name

            async with aiofiles.open(tmp_file_path, 'wb') as f:
                await f.write(content)

            if not PyMuPDFLoader:
                logger.warning("PyMuPDFLoader not installed; cannot extract PDF text.")
                return ScrapedContent(url=url, title=os.path.basename(urlparse(url).path), text="", success=False, error="PyMuPDFLoader not installed", scrape_time=time.time() - start, strategy="pdf")

            # Use the loader in a thread
            loader = PyMuPDFLoader(tmp_file_path)
            documents = await asyncio.to_thread(loader.load)
            pdf_text = "\n".join([getattr(doc, 'page_content', '') for doc in documents])

            if pdf_text:
                logger.info(f"Successfully extracted content from PDF: {url}")
                return ScrapedContent(url=url, title=os.path.basename(urlparse(url).path), text=pdf_text, success=True, scrape_time=time.time() - start, strategy="pdf")
            else:
                logger.warning(f"No text extracted from PDF: {url}")
                return ScrapedContent(url=url, title=os.path.basename(urlparse(url).path), text="", success=False, error="No text extracted from PDF", scrape_time=time.time() - start, strategy="pdf")

        except Exception as e:
            logger.error(f"Failed to process PDF {url}: {e}")
            return ScrapedContent(url=url, title=url, text="", success=False, error=f"PDF processing failed: {e}", scrape_time=time.time() - start, strategy="pdf")
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except Exception:
                    pass
            if session:
                await session.close()

    async def scrape(self, url: str, dynamic: bool = False, session=None) -> ScrapedContent:
        """General entrypoint that respects the shared browser.

        Workflow:
        1. Return cached result if present.
        2. Validate URL and skip if invalid.
        3. Route PDFs to the PDF extractor.
        4. Choose strategy: playwright if `dynamic` else aiohttp (use provided session if given).
        5. Cache successful results and return a single `ScrapedContent` object.
        """
        # 1. Cache Check
        if self._cache_enabled and self.cache:
            cached = self.cache.get(url)
            if cached:
                try:
                    logger.debug(f"[scraper] Returning cached result for: {url}")
                    return ScrapedContent(**cached)
                except Exception as e:
                    logger.warning(f"[scraper] Cache load failed: {e}")

        # 2. URL validation
        if not await self._is_valid_url(url):
            logger.warning(f"[scraper] Skipping invalid URL: {url}")
            return ScrapedContent(url=url, title=url, text="", success=False, error="Invalid or blocked URL.")

        res: Optional[ScrapedContent] = None

        # 3. PDF check (scrape() routes to _scrape_pdf)
        if url.lower().endswith('.pdf'):
            res = await self._scrape_pdf(url)
        else:
            # 4. Choose scraping strategy
            try:
                # Try aiohttp up to 3 separate attempts (reuse session when possible)
                aio_attempts = 3
                min_content_len = 500
                res = None
                local_session = session
                created_local = False
                try:
                    if local_session is None:
                        local_session = aiohttp.ClientSession()
                        created_local = True

                    for attempt in range(aio_attempts):
                        logger.debug(f"[scraper] aiohttp attempt {attempt+1}/{aio_attempts} for {url}")
                        try:
                            candidate = await self._scrape_with_aiohttp(url, session=local_session, retries=0)
                        except Exception as e:
                            logger.debug(f"[scraper] aiohttp attempt exception for {url}: {e}")
                            candidate = ScrapedContent(url=url, title=url, text="", success=False, error=str(e), strategy="aiohttp")

                        if candidate and getattr(candidate, 'success', False):
                            primary_text = candidate.html if getattr(candidate, 'html', None) else candidate.text
                            if primary_text and len(primary_text or "") >= min_content_len:
                                res = candidate
                                logger.debug(f"[scraper] aiohttp produced sufficient content for {url} (len={len(primary_text)})")
                                break
                            else:
                                logger.debug(f"[scraper] aiohttp content too short for {url} (len={len(primary_text or '')}); will retry or fallback")
                                # continue to next attempt
                        else:
                            logger.debug(f"[scraper] aiohttp attempt returned no success for {url}; error={getattr(candidate,'error',None)}")

                    # If aiohttp didn't produce sufficient content, fallback to Playwright
                    if res is None:
                        logger.info(f"[scraper] Falling back to Playwright for {url}")
                        res = await self._scrape_with_playwright(url)

                finally:
                    if created_local and local_session:
                        try:
                            await local_session.close()
                        except Exception:
                            pass
            except Exception as e:
                logger.exception("Unexpected error during scrape for %s: %s", url, e)
                res = ScrapedContent(url=url, title=url, text="", success=False, error=str(e))

        # 5. Cache the result if successful
        try:
            if res and getattr(res, 'success', False) and self._cache_enabled and self.cache:
                try:
                    self.cache.set(url, res.to_dict())
                except Exception as e:
                    logger.debug(f"Failed to write to cache for {url}: {e}")
        except Exception:
            logger.debug("Error while attempting to cache result", exc_info=True)

        # Post-process successful scrape: convert to markdown and save to GCS (if configured)
        try:
            if res and getattr(res, 'success', False):
                content_src = res.html if getattr(res, 'html', None) else res.text
                if content_src:
                    try:
                        if getattr(res, 'html', None) and '<' in content_src:
                            markdown = html_to_markdown(content_src)
                        else:
                            markdown = clean_extracted_text(content_src)
                    except Exception as e:
                        logger.debug(f"Markdown conversion failed: {e}")
                        markdown = clean_extracted_text(content_src)

                    # Attempt upload; save path back on the result for API consumers
                    try:
                        gcs_path = save_text_to_gcs(markdown, filename_prefix='scraped')
                        if gcs_path:
                            setattr(res, 'gcs_path', gcs_path)
                    except Exception as e:
                        logger.debug(f"Failed to upload scraped markdown to GCS: {e}")

                    # Attach markdown content (may be large)
                    try:
                        setattr(res, 'markdown', markdown)
                    except Exception:
                        pass
        except Exception:
            logger.debug('Post-processing failed', exc_info=True)

        return res if res is not None else ScrapedContent(url=url, title=url, text="", success=False, error="No result")