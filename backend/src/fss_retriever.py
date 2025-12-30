from google import genai
from google.genai import types
import asyncio
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime
import io, os, tempfile
from .logging_setup import logger
# Import API keys from api_keys.py
try:
    from .api_keys import GOOGLE_API_KEY
except ImportError:
    logger.error("Could not import API keys from api_keys.py. LLMs and embeddings may not initialize.")
    GOOGLE_API_KEY = None

try:
    from .config import GOOGLE_MODEL
except ImportError:
    logger.error("Could not import GOOGLE_MODEL from config.py. Using default model.")
    GOOGLE_MODEL = "gemini-2.0-flash"

# Configure the generative AI library with the API key

gemini_api_key = GOOGLE_API_KEY
gemini_model = GOOGLE_MODEL
FILE_INDEXING_TIMEOUT = 300  # 5 minutes


class GeminiFileSearchRetriever:
    """
    Manages a Gemini File Search Store with parallel uploads, batch querying, 
    and optimized O(1) cleanup.
    """
    def __init__(self, display_name_prefix: str = "crystal-fss", max_concurrent_uploads: int = 10, max_concurrent_generation: int = 10):
        # Generate a unique name for the store for this session
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.display_name = f"{display_name_prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
        
        # Initialize Client
        self.client = genai.Client(api_key=gemini_api_key)
        self.async_client = self.client.aio
        
        self.file_store_name: Optional[str] = None
        
        # Concurrency Controls
        self.upload_semaphore = asyncio.Semaphore(max_concurrent_uploads)
        self.generation_semaphore = asyncio.Semaphore(max_concurrent_generation)
        
        # Track created file names for fast O(1) cleanup
        self.created_file_names: List[str] = []
        
        logger.info(f"[{self.display_name}] Initialized. Upload Limit: {max_concurrent_uploads}, Gen Limit: {max_concurrent_generation}")

    async def _upload_single_file(self, url: str, content: str, store_name: str):
        """Helper: Handles temp file creation and upload with semaphore."""
        if not content or not content.strip():
            logger.warning(f"[{self.display_name}] SKIPPING {url}: Content is empty.")
            return None

        clean_name = url[-128:] if len(url) > 128 else url
        
        # 1. Create a temp file to handle encoding/buffering correctly
        # delete=False is required for cross-platform compatibility (Windows locks open files)
        tf = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8')
        try:
            tf.write(content)
            tf.flush()
            tf.close() # Close write handle so we can read it safely

            async with self.upload_semaphore:
                try:
                    # 2. OPEN the file in Read-Binary mode
                    with open(tf.name, "rb") as f:
                        logger.debug(f"[{self.display_name}] Uploading: {clean_name}")
                        
                        result = await self.async_client.file_search_stores.upload_to_file_search_store(
                            file=f,
                            file_search_store_name=store_name,
                            config=types.UploadToFileSearchStoreConfig(
                                display_name=clean_name,
                                mime_type="text/plain"
                            )
                        )
                    return result
                except Exception as e:
                    logger.error(f"[{self.display_name}] Failed to upload {url}: {e}")
                    return None
        finally:
            # 3. Clean up the local temp file immediately
            if os.path.exists(tf.name):
                os.unlink(tf.name)
 
    async def create_and_upload_contexts(self, relevant_contexts: Dict[str, Dict[str, Any]]) -> Optional[str]:
        if not relevant_contexts:
            return None

        try:
            # 1. Create the Store
            logger.info(f"[{self.display_name}] Creating File Search Store...")
            store_obj = await self.async_client.file_search_stores.create(
                config=types.CreateFileSearchStoreConfig(display_name=self.display_name)
            )
            self.file_store_name = store_obj.name
            logger.info(f"[{self.display_name}] Store created: {self.file_store_name}")

            # 2. Prepare Parallel Upload Tasks
            tasks = []
            for url, data in relevant_contexts.items():
                tasks.append(
                    self._upload_single_file(url, data.get("content", ""), self.file_store_name)
                )

            # 3. Execute Uploads in Parallel
            logger.info(f"[{self.display_name}] Starting parallel upload of {len(tasks)} files...")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 4. Collect created file IDs for cleanup later
            count_success = 0
            for res in results:
                if isinstance(res, Exception) or not res:
                    continue
                
                # Check different SDK return shapes just in case
                if hasattr(res, 'name'): 
                    self.created_file_names.append(res.name)
                    count_success += 1
                elif hasattr(res, 'file') and hasattr(res.file, 'name'):
                     self.created_file_names.append(res.file.name)
                     count_success += 1

            logger.info(f"[{self.display_name}] Uploads complete. Files tracked: {count_success}/{len(tasks)}")
            
            if count_success == 0 and len(tasks) > 0:
                logger.warning(f"[{self.display_name}] All uploads failed. Deleting store.")
                await self.delete_store()
                return None

            return self.file_store_name

        except Exception as e:
            logger.error(f"[{self.display_name}] Critical error in create_and_upload: {e}", exc_info=True)
            await self.delete_store()
            return None

    def _get_default_prompt(self, query: str) -> str:
        return f"""
        You are an expert research analyst. Provide a comprehensive, analytical response to the user's query by synthesizing information from the collected research data.

        USER QUERY: {query}

        INSTRUCTIONS:
        - You have access to a set of files containing research data. Use the file_search tool to find relevant information.
        - Your task is to analyze this data and produce a well-structured analytical response.
        - You must provide clear quantitative data if that is required and available.
        - You must accord the highest priority to recent information from the provided files.
        - The report should be well-structured, using markdown for headings, subheadings, and bullet points where appropriate.
        - Target 500-2000 words.
        - Provide a list of references at the end citing the data sources used.
        """

    async def answer_question(self, query: str, relevant_contexts: Dict[str, Dict[str, Any]], system_instruction: Optional[str] = None) -> str:
        """Single question entry point."""
        try:
            file_store_name = await self.create_and_upload_contexts(relevant_contexts)
            if not file_store_name:
                raise Exception("Failed to populate store.")

            
            fs_tool = types.FileSearch(
                file_search_store_names=[file_store_name]
            )
            
            # 2. Wrap it in a Tool object explicitly
            tool = types.Tool(
                file_search=fs_tool
            )
            
            # 3. Create the AFC Disable config explicitly
            # This ensures the SDK strictly respects the disable flag
            afc_config = types.AutomaticFunctionCallingConfig(
                disable=True
            )
            
            final_instruction = system_instruction if system_instruction else self._get_default_prompt(query)

            logger.info(f"[{self.display_name}] Generating content for single query...")
            
            response = await self.async_client.models.generate_content(
                model=GOOGLE_MODEL, 
                contents=[query],
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.1,
                    system_instruction=final_instruction,
                    automatic_function_calling={"disable": True},
                )
            )

            if not response.text:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                raise ValueError(f"Empty response. Reason: {finish_reason}")
            
            return response.text

        except Exception as e:
            logger.error(f"[{self.display_name}] Error: {e}", exc_info=True)
            raise
        finally:
            await self.delete_store()

    async def answer_batch_questions(self, queries: List[str], context_data: Dict, system_instruction_template: Optional[str] = None) -> Dict[str, str]:
        # 1. Create the store ONCE
        store_name = await self.create_and_upload_contexts(context_data)
        if not store_name:
            logger.error(f"[{self.display_name}] Store creation failed or empty.")
            return {}
        
        # 2. Define helper for single generation
        async def ask_single(q):
            final_instruction = system_instruction_template if system_instruction_template else self._get_default_prompt(q)
            
            # --- FIX: STRICT TYPE CONSTRUCTION ---
            # 1. Create the FileSearch object explicitly
            fs_tool = types.FileSearch(
                file_search_store_names=[store_name]
            )
            
            # 2. Wrap it in a Tool object explicitly
            tool = types.Tool(
                file_search=fs_tool
            )
            
            # 3. Create the AFC Disable config explicitly
            # This ensures the SDK strictly respects the disable flag
            afc_config = types.AutomaticFunctionCallingConfig(
                disable=True
            )

            async with self.generation_semaphore:
                try:
                    response = await self.async_client.models.generate_content(
                        model=GOOGLE_MODEL,
                        contents=[q],
                        config=types.GenerateContentConfig(
                            tools=[tool], # Pass as a list of types.Tool
                            temperature=0.1,
                            system_instruction=final_instruction,
                            automatic_function_calling=afc_config # Pass the Typed Object
                        )
                    )
                    
                    # Handle cases where response is filtered/blocked
                    if not response.text:
                         return q, f"Error: Empty response (Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'})"
                         
                    return q, response.text
                    
                except Exception as e:
                    logger.error(f"Error answering '{q}': {e}")
                    return q, f"Error generating response: {str(e)}"

        try:
            logger.info(f"[{self.display_name}] Processing {len(queries)} queries in parallel...")
            tasks = [ask_single(q) for q in queries]
            results = await asyncio.gather(*tasks)
            return dict(results)
        
        except Exception as e:
            logger.error(f"[{self.display_name}] Batch Error: {e}", exc_info=True)
            raise
        finally:
            await self.delete_store()

    async def delete_store(self):
        """Robust deletion: Ensures all files are deleted before deleting the store."""
        if not self.file_store_name:
            return

        store_to_delete = self.file_store_name
        self.file_store_name = None  # Prevent re-entry

        try:
            # 1. Attempt to delete all tracked files first
            if self.created_file_names:
                #logger.info(f"[{self.display_name}] Deleting {len(self.created_file_names)} tracked files concurrently...")

                async def delete_file_safe(name):
                    try:
                        await self.async_client.files.delete(name=name)
                    except Exception as e:
                        # Suppress 404 Not Found warnings, log others at debug level
                        if hasattr(e, 'status') and getattr(e, 'status', None) == 404:
                            pass
                        elif '404' in str(e) or 'Not Found' in str(e):
                            pass
                        else:
                            logger.debug(f"[{self.display_name}] Could not delete file {name}: {e}")

                tasks = [delete_file_safe(name) for name in self.created_file_names]
                await asyncio.gather(*tasks)
                self.created_file_names = []

            # 2. List any remaining files in the store and attempt to delete them
            try:
                file_list_resp = await self.async_client.file_search_stores.files.list(parent=store_to_delete)
                remaining_files = getattr(file_list_resp, 'files', [])
                if remaining_files:
                    logger.info(f"[{self.display_name}] Found {len(remaining_files)} remaining files in store, attempting to delete...")
                    async def delete_remaining_file(file_obj):
                        try:
                            await self.async_client.files.delete(name=file_obj.name)
                        except Exception as e:
                            # Suppress 404 Not Found warnings, log others at debug level
                            if hasattr(e, 'status') and getattr(e, 'status', None) == 404:
                                pass
                            elif '404' in str(e) or 'Not Found' in str(e):
                                pass
                            else:
                                logger.debug(f"[{self.display_name}] Could not delete remaining file {file_obj.name}: {e}")
                    tasks = [delete_remaining_file(f) for f in remaining_files]
                    await asyncio.gather(*tasks)
            except Exception as e:
                logger.warning(f"[{self.display_name}] Could not list or delete remaining files: {e}")

            # 3. Try deleting the store
            logger.info(f"[{self.display_name}] Deleting store: {store_to_delete}")
            await self.async_client.file_search_stores.delete(name=store_to_delete)

        except Exception as e:
            logger.error(f"[{self.display_name}] Cleanup error: {e}")
