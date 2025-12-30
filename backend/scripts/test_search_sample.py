import os, sys, asyncio

# Ensure workspace root is on sys.path when run standalone
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from backend.src.search import UnifiedSearcher

async def main():
    s = UnifiedSearcher()
    try:
        results, cached = await s.search('openai gpt-4 news', None, False)
    except Exception as e:
        print('Search failed:', e)
        return
    print('CACHED:', cached)
    for r in results:
        print('---')
        print(r)

if __name__ == '__main__':
    asyncio.run(main())
