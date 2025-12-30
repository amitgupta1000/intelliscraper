import asyncio
import os
import sys
import time

# --- Setup Project Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Imports from your project ---
try:
    from backend.src.nodes import extract_content, AgentState, SearchResult
    from backend.src.config import SCRAPER_USE_IN_MEMORY_CACHE
except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure you are running this script from the 'tools' directory.")
    print(f"Details: {e}")
    sys.exit(1)

import logging
# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def main():
    """Main function to run the scraper speed test."""
    if not SCRAPER_USE_IN_MEMORY_CACHE:
        logger.warning("SCRAPER_USE_IN_MEMORY_CACHE is False. In-memory caching is disabled, so speed improvements will be less dramatic.")
        logger.warning("Set SCRAPER_USE_IN_MEMORY_CACHE=True in your .env file to test in-memory caching.")

    # A list of URLs to test scraping on.
    # Using simple, static sites that are known to work well with aiohttp.
    urls_to_scrape = [
        "http://info.cern.ch/hypertext/WWW/TheProject.html", # The first website
        "https://example.com/",
        "https://www.w3.org/TR/html5/",
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    ]

    # Prepare the initial state for the node
    initial_data = [SearchResult(url=url, title=f"Title for {url}", snippet=f"Snippet for {url}", source="test") for url in urls_to_scrape]
    initial_state: AgentState = {
        "data": initial_data,
        "search_queries": ["test query"],
        "new_query": "test query",
    }

    # --- COLD RUN (CACHE MISS) ---
    print("\n" + "="*50)
    logger.info("PERFORMING COLD RUN (EXPECT CACHE MISSES)")
    print("="*50)
    start_time_cold = time.time()
    cold_run_state = await extract_content(initial_state.copy())
    end_time_cold = time.time()
    cold_duration = end_time_cold - start_time_cold
    scraped_contexts_cold = cold_run_state.get('relevant_contexts', {})
    logger.info(f"Cold run completed in: {cold_duration:.2f} seconds")
    logger.info(f"Successfully scraped {len(scraped_contexts_cold)} URLs.")

    # --- Display some scraped content from the cold run ---
    print("\n--- Sample Content from Cold Run ---")
    count = 0
    for url, data in scraped_contexts_cold.items():
        if count >= 3: # Show content for the first 2 URLs
            break
        print(f"\nURL: {url}")
        print(f"Title: {data.get('title', 'N/A')}")
        content_snippet = (data.get('content', '') or '').strip()[:2000] # Get first 300 chars
        print(f"Content Snippet: {content_snippet}...")
        count += 1
    print("\n" + "-"*30)

    # --- WARM RUN (CACHE HIT) ---
    print("\n" + "="*50)
    logger.info("PERFORMING WARM RUN (EXPECT CACHE HITS)")
    print("="*50)
    start_time_warm = time.time()
    warm_run_state = await extract_content(initial_state.copy())
    end_time_warm = time.time()
    warm_duration = end_time_warm - start_time_warm
    scraped_contexts_warm = warm_run_state.get('relevant_contexts', {})
    logger.info(f"Warm run completed in: {warm_duration:.2f} seconds")
    logger.info(f"Successfully scraped {len(scraped_contexts_warm)} URLs.")

    # --- Display some scraped content from the warm run ---
    print("\n--- Sample Content from Warm Run ---")
    count = 0
    for url, data in scraped_contexts_warm.items():
        if count >= 2: # Show content for the first 2 URLs
            break
        print(f"\nURL: {url}")
        print(f"Title: {data.get('title', 'N/A')}")
        content_snippet = (data.get('content', '') or '').strip()[:300]
        print(f"Content Snippet: {content_snippet}...")
        count += 1
    print("\n" + "-"*30)

    # --- FINAL VERIFICATION ---
    print("\n--- SPEED COMPARISON ---")
    print(f"Cold Run (No Cache): {cold_duration:.2f}s, {len(scraped_contexts_cold)} URLs scraped")
    print(f"Warm Run (Cache Hit): {warm_duration:.2f}s, {len(scraped_contexts_warm)} URLs scraped")

    if len(scraped_contexts_cold) != len(scraped_contexts_warm):
        print("⚠️ WARNING: The number of scraped URLs differs between runs. Caching might be inconsistent.")

    if warm_duration < cold_duration / 2:
        print("✅ SUCCESS: Caching provides a significant speed-up!")
    else:
        print("⚠️ NOTE: Caching speed-up was not as significant as expected. Check logs for cache hit/miss messages.")
    print("--- END OF TEST ---\n")

    # Ensure any shared scraper/playwright resources are cleaned up before event loop shutdown
    try:
        from backend.src.nodes import SHARED_SCRAPER
        if SHARED_SCRAPER:
            try:
                await SHARED_SCRAPER.stop()
                logger.info("Shared scraper stopped cleanly after tests.")
            except Exception as e:
                logger.debug(f"Error stopping shared scraper in test harness: {e}")
    except Exception:
        # If nodes import or stop fails, continue to exit; we attempted best-effort cleanup
        pass

if __name__ == "__main__":
    asyncio.run(main())