import asyncio

def _check_playwright():
    try:
        import playwright.async_api
        print("Playwright is installed and importable!")
        return True
    except Exception as e:
        print(f"Playwright import failed: {e}")
        return False

async def run_check():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto('https://example.com')
        print(await page.title())
        await browser.close()

if __name__ == '__main__':
    ok = _check_playwright()
    if ok:
        try:
            asyncio.run(run_check())
        except Exception as e:
            print(f"Playwright runtime check failed: {e}")