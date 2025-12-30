import sys, asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from src.scraper import Scraper
async def t():
    s = Scraper()
    try:
        await s.start()
        print("start ok, browser:", bool(getattr(s, "_browser", None)))
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        try: await s.stop()
        except: pass
asyncio.run(t())
