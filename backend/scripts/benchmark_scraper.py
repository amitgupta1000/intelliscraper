import asyncio
import time
from backend.src.scraper import Scraper

# naive scraper that creates a fresh aiohttp session per request and no Playwright reuse
import aiohttp
from bs4 import BeautifulSoup

async def naive_fetch(url):
    start = time.time()
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                html = await resp.text()
                soup = BeautifulSoup(html, 'html.parser')
                return {'url': url, 'success': True, 'time': time.time()-start}
    except Exception as e:
        return {'url': url, 'success': False, 'error': str(e), 'time': time.time()-start}

async def bench_naive(urls):
    start = time.time()
    results = await asyncio.gather(*[naive_fetch(u) for u in urls])
    return time.time()-start, results

async def bench_shared(urls):
    s = Scraper()
    start = time.time()
    tasks = [asyncio.create_task(s.scrape_url(u, dynamic=False)) for u in urls]
    results = await asyncio.gather(*tasks)
    dur = time.time()-start
    await s.close()
    return dur, [{'url': r.url, 'success': r.success, 'error': r.error if not r.success else None} for r in results]

async def main():
    urls = [
        'https://example.com',
        'https://httpbin.org/delay/2',
        'https://httpbin.org/delay/3',
        'https://httpbin.org/status/404',
    ]
    print('Running naive benchmark...')
    t_naive, res_naive = await bench_naive(urls)
    print(f'Naive total: {t_naive:.2f}s')
    print(res_naive)

    print('\nRunning shared-resources benchmark...')
    t_shared, res_shared = await bench_shared(urls)
    print(f'Shared total: {t_shared:.2f}s')
    print(res_shared)

if __name__ == '__main__':
    asyncio.run(main())
