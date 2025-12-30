import sys, asyncio
sys.path.insert(0, 'C:/deepsearch-1')

from main import startup_event
from backend.src import nodes

async def run():
    print('Before startup: SHARED_SCRAPER is', nodes.SHARED_SCRAPER)
    await startup_event()
    print('After startup: SHARED_SCRAPER is', nodes.SHARED_SCRAPER)
    scraper = nodes.SHARED_SCRAPER
    if scraper is None:
        print('Scraper not created')
        return
    browser = getattr(scraper, '_browser', None)
    playwright = getattr(scraper, '_playwright', None)
    print('Playwright _playwright:', bool(playwright))
    print('Playwright _browser:', bool(browser))

if __name__ == '__main__':
    asyncio.run(run())
