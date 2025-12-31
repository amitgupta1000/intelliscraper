from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

from .scraper import Scraper

app = FastAPI()

class ScrapeRequest(BaseModel):
    url: str


@app.post('/scrape')
async def scrape(req: ScrapeRequest):
    try:
        scraper = Scraper()
        # prefer aiohttp path; Scraper.scrape may be async
        if asyncio.iscoroutinefunction(scraper.scrape):
            scraped = await scraper.scrape(req.url)
        else:
            loop = asyncio.get_event_loop()
            scraped = await loop.run_in_executor(None, scraper.scrape, req.url)

        # Prefer `markdown` when available, fall back to `text`.
        markdown = None
        text = None
        gcs_path = None

        if hasattr(scraped, 'markdown'):
            markdown = getattr(scraped, 'markdown')
        elif isinstance(scraped, dict) and 'markdown' in scraped:
            markdown = scraped.get('markdown')

        if hasattr(scraped, 'text'):
            text = getattr(scraped, 'text')
        elif isinstance(scraped, dict) and 'text' in scraped:
            text = scraped.get('text')

        if hasattr(scraped, 'gcs_path'):
            gcs_path = getattr(scraped, 'gcs_path')
        elif isinstance(scraped, dict) and 'gcs_path' in scraped:
            gcs_path = scraped.get('gcs_path')

        # Choose primary content to return
        primary = markdown if markdown else text
        if not primary:
            raise HTTPException(status_code=500, detail='Scraper returned no content')

        # If we have a GCS path, try to produce a usable download URL:
        download_url = None
        if gcs_path:
            try:
                # parse gs://bucket/name
                bucket = None
                blob_name = None
                if isinstance(gcs_path, str) and gcs_path.startswith('gs://'):
                    _, rest = gcs_path.split('://', 1)
                    parts = rest.split('/', 1)
                    bucket = parts[0]
                    blob_name = parts[1] if len(parts) > 1 else ''
                elif isinstance(gcs_path, str) and gcs_path.startswith('https://storage.googleapis.com/'):
                    # https://storage.googleapis.com/bucket/path
                    rest = gcs_path[len('https://storage.googleapis.com/'):]
                    parts = rest.split('/', 1)
                    bucket = parts[0]
                    blob_name = parts[1] if len(parts) > 1 else ''

                if bucket and blob_name:
                    try:
                        # Try to create a signed URL if google-cloud-storage is available
                        from google.cloud import storage as gcs_lib
                        client = gcs_lib.Client()
                        bucket_obj = client.bucket(bucket)
                        blob = bucket_obj.blob(blob_name)
                        try:
                            # v4 signed URL if supported
                            download_url = blob.generate_signed_url(version='v4', expiration=3600)
                        except Exception:
                            # Fallback to a public URL format
                            from urllib.parse import quote
                            download_url = f"https://storage.googleapis.com/{bucket}/{quote(blob_name)}"
                    except Exception:
                        # google lib not available or signing failed; give public URL
                        from urllib.parse import quote
                        download_url = f"https://storage.googleapis.com/{bucket}/{quote(blob_name)}" if bucket and blob_name else None
            except Exception:
                download_url = None

        return {
            'url': getattr(scraped, 'url', req.url) if not isinstance(scraped, dict) else scraped.get('url', req.url),
            'title': getattr(scraped, 'title', None) if not isinstance(scraped, dict) else scraped.get('title'),
            'markdown': primary,
            'gcs_path': gcs_path,
            'download_url': download_url,
            'text': text if text and text != primary else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
