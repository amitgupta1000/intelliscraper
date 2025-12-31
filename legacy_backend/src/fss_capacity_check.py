import os
from google import genai
from google.genai import types
from .logging_setup import logger
from .api_keys import GOOGLE_API_KEY
FILE_SEARCH_STORE_NAME = os.environ.get("FSS_STORE_NAME", "<YOUR_FILE_SEARCH_STORE_NAME>")
api_key = GOOGLE_API_KEY
client = genai.Client()

def get_fss_storage_usage(store_name):
    print(f"Querying FileSearchStore: {store_name}")
    files = client.file_search_stores.list_files(file_search_store_name=store_name)
    total_bytes = 0
    file_count = 0
    for file in files:
        size = getattr(file, "size_bytes", None)
        if size is None:
            # Try alternative property name if needed
            size = getattr(file, "size", 0)
        total_bytes += size
        file_count += 1
        print(f"File: {getattr(file, 'display_name', file.name)} | Size: {size} bytes")
    print(f"Total files: {file_count}")
    print(f"Total storage used: {total_bytes} bytes ({total_bytes/1024/1024:.2f} MB)")
    print(f"Capacity remaining: {(1073741824-total_bytes)/1024/1024:.2f} MB of 1GB")
    return total_bytes, file_count

if __name__ == "__main__":
    if FILE_SEARCH_STORE_NAME == "<YOUR_FILE_SEARCH_STORE_NAME>":
        print("Please set the FSS_STORE_NAME environment variable or edit the script with your store name.")
    else:
        get_fss_storage_usage(FILE_SEARCH_STORE_NAME)
