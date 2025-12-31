import os
import asyncio
import logging
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API Key from environment
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

async def list_all_stores():
    """Lists all available File Search Stores."""
    print("--- Listing all File Search Stores ---")
    client = genai.Client(api_key=API_KEY)
    try:
        stores = list(client.file_search_stores.list())
        if not stores:
            print("No File Search Stores found.")
            return []
        
        for i, store in enumerate(stores):
            print(f"[{i+1}] Display Name: {store.display_name}")
            print(f"    Store Name:   {store.name}")
        print("-" * 38)
        return stores
    except Exception as e:
        logging.error(f"Failed to list stores: {e}", exc_info=True)
        return []

async def list_files_in_store(store_name: str):
    """Lists all files within a specific File Search Store."""
    print(f"\n--- Listing files in store: {store_name} ---")
    client = genai.Client()
    
    try:
        # The API requires listing all files and then filtering by store name client-side.
        all_files_pager = client.files.list(page_size=1000)
        
        files_found = []
        for file in all_files_pager:
            if hasattr(file, 'file_search_store_name') and file.file_search_store_name == store_name:
                files_found.append(file)

        if not files_found:
            print("No files found in this store.")
            return

        print(f"Found {len(files_found)} file(s):")
        total_size = 0
        for file in files_found:
            size_kb = file.size_bytes / 1024
            total_size += file.size_bytes
            print(f"  - Display Name: {file.display_name}")
            print(f"    File Name:    {file.name}")
            print(f"    Size:         {size_kb:.2f} KB")
            print(f"    State:        {file.state.name}")

        total_size_mb = total_size / (1024 * 1024)
        print(f"\nTotal size of files in store: {total_size_mb:.4f} MB")

    except Exception as e:
        logging.error(f"Failed to list files for store '{store_name}': {e}", exc_info=True)

async def main():
    """Main function to run the inspection."""
    stores = await list_all_stores()
    if not stores:
        return

    try:
        choice = int(input("Enter the number of the store to inspect (or 0 to exit): "))
        if 0 < choice <= len(stores):
            selected_store = stores[choice - 1]
            await list_files_in_store(selected_store.name)
        elif choice != 0:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    asyncio.run(main())