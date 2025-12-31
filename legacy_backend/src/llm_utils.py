# llm_utils.py
#===================

import logging, os, random, asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from .logging_setup import logger

"""Archived stub for `llm_utils.py`.

The heavy LLM initialization code and Gemini/GenAI wrappers were archived.
Restore from Git history if you need to re-enable LLM features.
"""

raise ImportError("backend.src.llm_utils has been archived to backend/archived/. Restore from Git history if needed.")

# nest_asyncio removed - not needed for web deployment with FastAPI/uvicorn
# FastAPI with uvicorn handles async execution properly without nested event loops

try:
    # Try to import from LangChain first for compatibility
    from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain core messages not available: {e}. Using fallback message classes.")
    
    # Create our own message classes that are compatible with the API
    class BaseMessage:
        def __init__(self, content: str = "", **kwargs):
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def __str__(self):
            return f"{self.__class__.__name__}(content='{self.content}')"
    
    class AnyMessage(BaseMessage): pass
    class AIMessage(BaseMessage): pass
    class SystemMessage(BaseMessage): pass
    class HumanMessage(BaseMessage): pass
    
    LANGCHAIN_CORE_AVAILABLE = False

try:
    from google import genai
    from google.genai.types import Content, Part
    GOOGLE_GENAI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import google.genai: {e}")
    GOOGLE_GENAI_AVAILABLE = False


# Import API keys from api_keys.py
try:
    from .api_keys import GOOGLE_API_KEY
except ImportError:
    logger.error("Could not import API keys from api_keys.py. LLMs and embeddings may not initialize.")
    GOOGLE_API_KEY = None

# Import configuration
try: 
    from .config import (
        GOOGLE_MODEL,
        MAX_RETRIES,
        BASE_DELAY,
        MAX_CONCURRENT_CALLS,
        MAX_CALLS_PER_SECOND,
    )
except ImportError:
    logger.error("Could not import config parameters from config.py. LLMs and embeddings may not initialize.")
    GOOGLE_MODEL = "gemini-2.0-flash"
    MAX_RETRIES = 5
    BASE_DELAY = 1
    MAX_CONCURRENT_CALLS = 10
    MAX_CALLS_PER_SECOND = 40

# --- Embedding Model Initialization ---

embeddings = None # The primary embeddings model

# Enhanced embeddings with task type support
try:
    from .enhanced_embeddings import EnhancedGoogleEmbeddings, create_enhanced_embeddings
    USE_ENHANCED_EMBEDDINGS = True
except ImportError:
    logger.warning("Enhanced embeddings not available, falling back to standard implementation")
    USE_ENHANCED_EMBEDDINGS = False

# For embedding/indexing - use enhanced embeddings if available
try:
    if GOOGLE_API_KEY:
        if USE_ENHANCED_EMBEDDINGS:
            # Use enhanced embeddings with task type optimization
            embeddings = create_enhanced_embeddings(
                google_api_key=GOOGLE_API_KEY,
                use_case="retrieval",  # Optimized for document retrieval
                output_dimensionality=768,  # Efficient size
                normalize_embeddings=True,
                batch_size=100  # Reasonable batch size
            )
            logger.info("Initialized Enhanced Google Embeddings with gemini-embedding-001 (task-optimized)")
        
        else:
            embeddings = None
            logger.error("No embedding implementation available")
    else:
         embeddings = None
         logger.error("No Google API key available for initializing embeddings.")

except Exception as e:
    embeddings = None
    logger.error(f"Failed to initialize embeddings model: {e}")
    

# --- LLM Model Initialization --- 
from google import genai
import asyncio
import time
from ratelimit import limits, RateLimitException, sleep_and_retry # Import rate limiting decorators

# Configure the generative AI library with the API key
from google import genai
from google.genai import types

gemini_model =  GOOGLE_MODEL
gemini_api_key = GOOGLE_API_KEY
if not gemini_api_key:
    gemini_api_key = GOOGLE_API_KEY  # Fallback to GEMINI_API_KEY if GOOGLE_API_KEY not set

if not gemini_api_key:
    logger.error("Neither GOOGLE_API_KEY nor GEMINI_API_KEY found in environment.")
    # Handle this case, perhaps skip Gemini initialization or raise an error
else:
    logger.debug("Gemini API configured successfully.")

# Create a global semaphore to limit concurrent calls
semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
async def llm_call_async(messages: List[AnyMessage], max_tokens: int = None): # Change parameter to accept a list of AnyMessage and added max_tokens
    """
    Asynchronously call the Gemini API with the provided message objects.
    Returns the content of the assistant's reply.
    Includes retry logic and rate limiting.
    Allows setting max_output_tokens.
    
    Args:
        messages: List of message objects with .content attribute (SystemMessage, HumanMessage, AIMessage)
        max_tokens: Maximum tokens to generate (optional)
    """

    if not gemini_api_key:
        logger.error("Gemini API key not available. Skipping API call.")
        return None

    client = None # Initialize client to None outside the try block
    try:
        client = genai.Client(api_key=gemini_api_key) # Use genai.Client

        gemini_contents = []
        for message in messages:
            if isinstance(message, SystemMessage):
                 if gemini_contents and gemini_contents[0].role == 'user':
                     gemini_contents[0].parts[0].text = f"System Instruction: {message.content}\n\n" + gemini_contents[0].parts[0].text
                 else:
                     gemini_contents.insert(0, genai.types.Content(role='user', parts=[genai.types.Part(text=f"System Instruction: {message.content}\n\n")]))

            elif isinstance(message, HumanMessage):
                gemini_contents.append(genai.types.Content(role='user', parts=[genai.types.Part(text=message.content)]))
            elif isinstance(message, AIMessage):
                 gemini_contents.append(genai.types.Content(role='model', parts=[genai.types.Part(text=message.content)]))

        if not gemini_contents:
            logger.warning("No valid messages to send to Gemini API.")
            if client: await client.close() # Close client before returning
            return None

        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:
                    response = await client.aio.models.generate_content(
                                      model=gemini_model,
                                      contents=gemini_contents,
                                      config=types.GenerateContentConfig(
                                          temperature=0.1,
                                          max_output_tokens=30000
                                      )
                                  )

                    logger.debug(f"Successfully called Gemini API on attempt {attempt + 1}")
                    return response.text # Return the text of the response

            except RateLimitException as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}. Waiting before retrying...")
                await asyncio.sleep(e.period) # Wait for the duration specified by the rate limit decorator
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed calling GEMINI Inference API: {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1) # Exponential backoff with jitter
                    logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Failed to call GEMINI Inference API.")
                    if client: client.close() # Close client before returning
                    return None # Return None after max retries

    except Exception as e:
        logger.error(f"An error occurred before attempting API calls: {e}")
        #if client: client.close() # Close client before returning
        return None