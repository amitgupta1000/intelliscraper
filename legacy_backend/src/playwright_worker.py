#!/usr/bin/env python3
"""Small worker to run a single Playwright scrape in a separate process.

This script is invoked by `scraper.py` as a subprocess. It fetches the
page, extracts text and title, and emits a single JSON object to stdout.
"""
import argparse
import asyncio
import json
import sys
import time


async def run_scrape(url: str, timeout: int = 120):
    start = time.time()
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        return {"url": url, "title": url, "text": "", "html": None, "success": False, "error": f"Playwright import failed: {e}", "scrape_time": time.time() - start}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-setuid-sandbox"]) 
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, timeout=timeout * 1000)
            html = await page.content()
            try:
                text = await page.inner_text("body")
            except Exception:
                # fallback to stripping tags from html if inner_text fails
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
            title = None
            try:
                title = await page.title()
            except Exception:
                title = url
            await browser.close()
            return {"url": url, "title": title or url, "text": text, "html": html, "success": True, "error": None, "scrape_time": time.time() - start}
    except Exception as e:
        return {"url": url, "title": url, "text": "", "html": None, "success": False, "error": str(e), "scrape_time": time.time() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    try:
        result = asyncio.run(run_scrape(args.url, timeout=args.timeout))
        print(json.dumps(result))
        # flush to ensure parent receives output
        sys.stdout.flush()
        # exit 0 even on non-success so parent can parse JSON
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"url": args.url, "title": args.url, "text": "", "html": None, "success": False, "error": str(e), "scrape_time": 0}))
        sys.stdout.flush()
        sys.exit(2)


if __name__ == "__main__":
    main()
