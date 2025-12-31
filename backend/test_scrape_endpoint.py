from backend.src.api import app
from fastapi.testclient import TestClient

def run_test():
    client = TestClient(app)
    print('POST /scrape -> https://example.com')
    resp = client.post('/scrape', json={'url': 'https://example.com'})
    print('Status:', resp.status_code)
    try:
        print('JSON:', resp.json())
    except Exception:
        print('Text:', resp.text[:1000])

if __name__ == '__main__':
    run_test()
