import sys
sys.path.insert(0, r"C:\intelliscraper")
from backend.src.api import scrape, ScrapeRequest
import asyncio

def main():
    print("Calling scrape directly...")
    try:
        result = asyncio.run(scrape(ScrapeRequest(url="https://example.com")))
        print("Result:", result)
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    main()
