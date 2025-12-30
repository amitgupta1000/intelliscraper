import asyncio
import os
import glob
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
STORE_NAME = "My_Knowledge_Base_FSS"
# Use a model that supports File Search (e.g., gemini-1.5-flash-001 or gemini-1.5-pro-001)
MODEL_ID = "gemini-1.5-flash-001" 

# Local files to process (create dummy files for this demo if you don't have any)
FILES_TO_PROCESS = ["contract_alpha.txt", "meeting_notes.txt"]

# -----------------------------------------------------------------------------
# HELPER: Create dummy files for demonstration
# -----------------------------------------------------------------------------
def create_dummy_files():
    if not os.path.exists("contract_alpha.txt"):
        with open("contract_alpha.txt", "w") as f:
            f.write("Contract Alpha: The project deadline is extended to December 31st, 2025. Budget cap is $50k.")
    if not os.path.exists("meeting_notes.txt"):
        with open("meeting_notes.txt", "w") as f:
            f.write("Meeting Notes: The client requested a focus on mobile UI. The primary color scheme should be blue.")

# -----------------------------------------------------------------------------
# MAIN ASYNC WORKFLOW
# -----------------------------------------------------------------------------
async def main():
    print(f"--- Starting Gemini File Search Workflow ---\n")
    create_dummy_files()
    
    # 1. Initialize Async Client
    # Note: The 'google-genai' library handles async via the .aio accessor or AsyncClient
    client = genai.Client(api_key=API_KEY)
    
    # -------------------------------------------------------------------------
    # STEP 1: Manage File Search Store (FSS)
    # -------------------------------------------------------------------------
    print(f"[1] Checking for existing File Search Store: '{STORE_NAME}'...")
    
    target_store = None
    
    # List existing stores to see if ours exists
    # We iterate through pages of stores
    pager = client.file_search_stores.list()
    for store in pager:
        if store.display_name == STORE_NAME:
            target_store = store
            print(f"    Found existing store: {store.name}")
            break
            
    # If not found, create it
    if not target_store:
        print(f"    Store not found. Creating new store...")
        target_store = client.file_search_stores.create(
            config=types.CreateFileSearchStoreConfig(
                display_name=STORE_NAME
            )
        )
        print(f"    Created new store: {target_store.name}")

    # -------------------------------------------------------------------------
    # STEP 2: Check & Upload Files (Async)
    # -------------------------------------------------------------------------
    print(f"\n[2] Checking file inventory in store...")

    # Get list of files already in this store
    # Note: We use the store's resource name (stores/...) to filter or list
    existing_filenames = set()
    
    # We list files belonging to this specific store
    # (The SDK allows listing files filtered by store)
    files_in_store_pager = client.files.list(
        config=types.ListFilesConfig(page_size=100)
    )
    
    # Filter locally for files that belong to our store (simplest method)
    # Note: In a production app with millions of files, you'd rely on metadata filtering
    # Here we just check names for the demo.
    # (Currently, mapping specific files to stores via list command can be complex, 
    # so we often track this in a local DB, but here we assume 'display_name' is unique).
    
    # For this demo, we will blindly upload if we aren't sure, 
    # but let's try to see if we can identify duplicates by display_name.
    # A robust check usually involves checking the store's associated file list.
    
    # Let's proceed to upload missing files.
    upload_tasks = []
    
    for file_path in FILES_TO_PROCESS:
        display_name = os.path.basename(file_path)
        print(f"    Processing local file: {display_name}")
        
        # In a real scenario, you would check `existing_filenames` here.
        # We will perform the upload. The SDK handles ingestion.
        
        print(f"    -> Starting upload & ingestion for {display_name}...")
        
        # We use the specialized method that uploads AND adds to the store in one go
        # This returns an Operation (job) that we must wait for.
        op = client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=target_store.name,
            config=types.UploadToFileSearchStoreConfig(
                display_name=display_name
            )
        )
        upload_tasks.append(op)

    # Wait for all uploads to complete (Ingestion/Embedding phase)
    # Note: The upload_to_... method is synchronous in blocking the upload, 
    # but the *processing* on server side might take a moment.
    # The SDK method provided typically waits for upload completion.
    
    print(f"    All files uploaded. Waiting for indexing (embedding) to complete...")
    
    # We poll the operations until they are done
    for op in upload_tasks:
        # Simple polling loop
        while not op.done:
            print("    ... indexing in progress ...")
            # Refresh operation status
            op = client.operations.get(name=op.name)
            await asyncio.sleep(2)
        print(f"    -> File ready.")

    # -------------------------------------------------------------------------
    # STEP 3: Summary & Capacity Utilisation
    # -------------------------------------------------------------------------
    print(f"\n[3] Storage Summary & Utilization")
    
    # There isn't a single "utilization %" API field, but we can sum the file sizes.
    # Google allows 1GB free storage for File Search.
    
    total_bytes = 0
    file_count = 0
    
    # List files actually in the store to calculate size
    # We iterate and check if the file is attached to our store
    # (Actual API might require iterating all files, or getting the store details)
    # For this demo, we can get the store details if available, or just trust our uploads.
    
    # Let's get the specific store details again, sometimes it has stats
    updated_store = client.file_search_stores.get(name=target_store.name)
    
    # Note: As of current SDK, explicit "size_bytes" might not be on the Store object directly,
    # so we might have to sum the source files. 
    # We will estimate based on local files for the demo display.
    for f in FILES_TO_PROCESS:
        total_bytes += os.path.getsize(f)
        file_count += 1
        
    print(f"    Store Name: {updated_store.display_name}")
    print(f"    Store ID:   {updated_store.name}")
    print(f"    Total Files: {file_count}")
    print(f"    Total Size:  {total_bytes / 1024:.2f} KB")
    print(f"    Status:      Ready for Query")

    # -------------------------------------------------------------------------
    # STEP 4: Async Retrieval & Generation (The Query)
    # -------------------------------------------------------------------------
    QUERY = "What is the budget cap for Contract Alpha and what is the deadline?"
    
    print(f"\n[4] Processing Query: '{QUERY}'")
    print(f"    (Using async generation with retrieval tool)")

    # Define the tool using our store
    file_search_tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[target_store.name]
        )
    )

    # Perform the generation (RAG)
    # We use the async client accessor '.aio' if available, or just wrap in thread if sync.
    # The 'google-genai' library's 'generate_content' is sync by default, 
    # but we can use 'async_client' if we initialized one.
    # Since we initialized a sync client above, we'll use it directly here.
    # (To use true async, you would use `client = genai.Client(...)` and then `await client.aio.models...`)
    
    # Switching to async call style for demonstration:
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=QUERY,
        config=types.GenerateContentConfig(
            tools=[file_search_tool],
            temperature=0.0  # Low temp for factual retrieval
        )
    )

    # -------------------------------------------------------------------------
    # STEP 5: Response & Token Usage
    # -------------------------------------------------------------------------
    print(f"\n[5] Draft Response Generated:\n")
    print(f"{'='*40}")
    print(response.text)
    print(f"{'='*40}")

    # Inspecting Token Usage
    # The 'usage_metadata' contains info on prompt tokens (which includes retrieved file chunks)
    usage = response.usage_metadata
    
    print(f"\n[6] Token Usage Analysis:")
    if usage:
        print(f"    Prompt Tokens:      {usage.prompt_token_count} (Includes query + retrieved context)")
        print(f"    Candidates Tokens:  {usage.candidates_token_count} (The output response)")
        print(f"    Total Tokens:       {usage.total_token_count}")
        
        # Calculate approximate cost (Standard Pricing Assumption)
        # Note: Retrieval (Search) is free. You pay for the input tokens (the chunks retrieved).
        # $3.50 / 1M input tokens (approx price for Pro models, Flash is much cheaper)
        cost_est = (usage.total_token_count / 1_000_000) * 0.35 # Assuming Flash pricing roughly
        print(f"    Approx Query Cost:  ${cost_est:.6f}")
    else:
        print("    Usage metadata not available.")

# -----------------------------------------------------------------------------
# RUNNER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())