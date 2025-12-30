import asyncio
import time
from backend.src.scraper import Scraper

async def run_test():
    s = Scraper()
    urls = [
        "https://example.com",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/status/404",
        "https://httpbin.org/delay/3",
    ]
    start = time.time()
    tasks = [asyncio.create_task(s.scrape_url(u, dynamic=False)) for u in urls]
    results = await asyncio.gather(*tasks)
    duration = time.time() - start
    print(f"Total time: {duration:.2f}s")
    for r in results:
        print(r.url, 'success=' + str(r.success), 'error=' + str(r.error))
    await s.close()

if __name__ == '__main__':
    asyncio.run(run_test())
