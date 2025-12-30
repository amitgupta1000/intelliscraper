"""
A simple async script to test the UnifiedSearcher's caching behavior.

This script will:
1. Configure logging to see cache hit/miss messages.
2. Initialize the UnifiedSearcher.
3. Run a search for a query, which should result in a CACHE MISS.
4. Run the same search again, which should result in a CACHE HIT.
"""
import asyncio
import logging
import os
import sys

# --- Setup Project Path ---
# Add the project's root directory to the Python path to ensure correct module imports.
# This allows us to run the script from the 'tools' directory and still import from 'backend.src'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Imports from your project ---
try:
    from backend.src.search import UnifiedSearcher
    from backend.src.config import SERPER_API_KEY
except ImportError as e:
    print(f"Error: Could not import necessary modules. Make sure you are running this script from the 'tools' directory.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configure Logging ---
# Set up a basic logger to see the output from the search module.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def main():
    """Main function to run the search test."""
    if not SERPER_API_KEY:
        logger.error("SERPER_API_KEY is not set in your environment. The test cannot run.")
        return

    # 1. Initialize the UnifiedSearcher
    searcher = UnifiedSearcher()
    query = "What are the latest developments in large language models?"

    print("\n" + "="*50)
    logger.info(f"PERFORMING FIRST SEARCH for query: '{query}'")
    print("This should result in a CACHE MISS and perform a live search.")
    print("="*50)

    # 2. Run the search for the first time
    results1, was_cached1 = await searcher.search(query)

    if results1:
        logger.info(f"First search returned {len(results1)} results. Was it from cache? -> {was_cached1}")
        print("--- Top 3 Results ---")
        for i, res in enumerate(results1[:3]):
            print(f"{i+1}. {res.title}")
            print(f"   {res.url}\n")
    else:
        logger.warning("First search returned no results.")

    # Add a small delay to ensure any async cache writes can complete
    await asyncio.sleep(2)

    print("\n" + "="*50)
    logger.info(f"PERFORMING SECOND SEARCH for the same query.")
    print("This should result in a CACHE HIT from Firestore or in-memory.")
    print("="*50)

    # 3. Run the same search again
    results2, was_cached2 = await searcher.search(query)

    if results2:
        logger.info(f"Second search returned {len(results2)} results. Was it from cache? -> {was_cached2}")
        print("--- Top 3 Results ---")
        for i, res in enumerate(results2[:3]):
            print(f"{i+1}. {res.title}")
            print(f"   {res.url}\n")
    else:
        logger.warning("Second search returned no results.")

    # 4. Final verification
    print("\n--- VERIFICATION ---")
    if not was_cached1 and was_cached2:
        print("✅ SUCCESS: Caching test passed! The first search was live and the second was from the cache.")
    elif was_cached1 and was_cached2:
        print("✅ SUCCESS: Caching test passed! Both searches were served from a pre-existing cache.")
    else:
        print("❌ FAILURE: Caching test failed. Check the logs for 'CACHE HIT' or 'CACHE MISS' messages.")
    print("--- END OF TEST ---\n")


if __name__ == "__main__":
    # Ensure you have your .env file with SERPER_API_KEY in the project root
    # or have the environment variable set.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")