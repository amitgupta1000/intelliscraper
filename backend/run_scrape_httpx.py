from main import app
import asyncio
from httpx import AsyncClient, ASGITransport

async def run_test_async():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://testserver') as client:
        print('POST /scrape -> https://example.com')
        resp = await client.post('/scrape', json={'url': 'https://example.com'})
        print('Status:', resp.status_code)
        try:
            print('JSON:', resp.json())
        except Exception:
            print('Text:', resp.text[:1000])

def run_test():
    asyncio.run(run_test_async())

if __name__ == '__main__':
    run_test()
