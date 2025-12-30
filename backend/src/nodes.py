import json, re, asyncio
import importlib
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Union
from .logging_setup import logger
from .api_keys import GOOGLE_API_KEY, SERPER_API_KEY

# Try to import optional dependencies with fallbacks
try:
    from pydantic import BaseModel, Field, ValidationError, conlist
except ImportError:
    logger.warning("pydantic not available. Using basic classes instead.")
    class BaseModel: pass
    def Field(**kwargs): return None
    ValidationError = Exception # Fallback to a generic exception
    def conlist(type_, **kwargs): return List[type_]

try:
    from langchain_core.documents import Document
except ImportError:
    logger.warning("langchain_core.documents not available. Using basic Document class.")
    class Document: # type: ignore
        def __init__(self, page_content: str = "", metadata: Optional[dict] = None):
            self.page_content = page_content; self.metadata = metadata or {}

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    logger.warning("langchain_core.messages not available. Using basic message classes.")
    class BaseMessage:
        def __init__(self, content: str = "", **kwargs): self.content = content
    class SystemMessage(BaseMessage): pass # type: ignore
    class HumanMessage(BaseMessage): pass # type: ignore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
except ImportError:
    logger.error("Could not import necessary LangChain components. Embedding and indexing may fail.")
    class RecursiveCharacterTextSplitter: # type: ignore
        def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
            self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        def split_documents(self, documents):
            # Simplified fallback implementation
            return [doc for doc in documents]
    FAISS = None # type: ignore

# Import necessary classes and functions from other modules
# Initialize names with safe defaults so partial imports don't disable everything
llm_call_async = None
embeddings = None
UnifiedSearcher = None
SearchResult = None
Scraper = None
ScrapedContent = None
HybridRetriever = None
create_hybrid_retriever = None
GeminiFileSearchRetriever = None
safe_format = get_current_date = clean_extracted_text = rank_urls = None
save_report_to_text = format_research_report = enhance_report_readability = None
query_writer_instructions_legal = query_writer_instructions_general = None
query_writer_instructions_macro = query_writer_instructions_deepsearch = None
query_writer_instructions_person_search = query_writer_instructions_investment = None
web_search_validation_instructions = analytical_question_instructions = None

def _try_import(module_path: str, names: list):
    """Helper to attempt importing specific names from a module and return a dict of found objects.

    Supports relative imports by passing the current package to importlib.import_module.
    """
    results = {}
    try:
        if module_path.startswith('.'):
            mod = importlib.import_module(module_path, package=__package__)
        else:
            mod = importlib.import_module(module_path)
        for n in names:
            results[n] = getattr(mod, n)
        return results
    except Exception as e:
        logger.debug(f"Optional import failed for {module_path}: {e}")
        return {}

# Try importing utilities and local components individually so one missing module
# doesn't disable unrelated functionality.
_res = _try_import('.llm_utils', ['llm_call_async', 'embeddings'])
if _res:
    llm_call_async = _res.get('llm_call_async')
    embeddings = _res.get('embeddings')
else:
    logger.critical("Failed to import llm utilities (.llm_utils). LLM features will be disabled.")

_res = _try_import('.search', ['UnifiedSearcher', 'SearchResult'])
if _res:
    UnifiedSearcher = _res.get('UnifiedSearcher')
    SearchResult = _res.get('SearchResult')
else:
    logger.error("Failed to import UnifiedSearcher/SearchResult (.search). Search functionality may be limited.")

_res = _try_import('.scraper', ['Scraper', 'ScrapedContent'])
if _res:
    Scraper = _res.get('Scraper')
    ScrapedContent = _res.get('ScrapedContent')
else:
    logger.error("Failed to import Scraper/ScarpedContent (.scraper). Scraping will be disabled.")

_res = _try_import('.hybrid_retriever', ['HybridRetriever', 'create_hybrid_retriever'])
if _res:
    HybridRetriever = _res.get('HybridRetriever')
    create_hybrid_retriever = _res.get('create_hybrid_retriever')

_res = _try_import('.fss_retriever', ['GeminiFileSearchRetriever'])
if _res:
    GeminiFileSearchRetriever = _res.get('GeminiFileSearchRetriever')

_res = _try_import('.utils', [
    'safe_format', 'get_current_date', 'clean_extracted_text', 'rank_urls',
    'save_report_to_text', 'format_research_report', 'enhance_report_readability'
])
if _res:
    safe_format = _res.get('safe_format')
    get_current_date = _res.get('get_current_date')
    clean_extracted_text = _res.get('clean_extracted_text')
    rank_urls = _res.get('rank_urls')
    save_report_to_text = _res.get('save_report_to_text')
    format_research_report = _res.get('format_research_report')
    enhance_report_readability = _res.get('enhance_report_readability')
else:
    logger.error("Failed to import utilities (.utils). Some helper functions will be unavailable.")

_res = _try_import('.prompt', [
    'query_writer_instructions_legal', 'query_writer_instructions_general',
    'query_writer_instructions_macro', 'query_writer_instructions_deepsearch',
    'query_writer_instructions_person_search', 'query_writer_instructions_investment',
    'web_search_validation_instructions', 'analytical_question_instructions',
])
if _res:
    query_writer_instructions_legal = _res.get('query_writer_instructions_legal')
    query_writer_instructions_general = _res.get('query_writer_instructions_general')
    query_writer_instructions_macro = _res.get('query_writer_instructions_macro')
    query_writer_instructions_deepsearch = _res.get('query_writer_instructions_deepsearch')
    query_writer_instructions_person_search = _res.get('query_writer_instructions_person_search')
    query_writer_instructions_investment = _res.get('query_writer_instructions_investment')
    web_search_validation_instructions = _res.get('web_search_validation_instructions')
    analytical_question_instructions = _res.get('analytical_question_instructions')
else:
    logger.error("Failed to import prompt templates (.prompt). Query generation prompts will be unavailable.")

try:
    from .config import(
            USE_PERSISTENCE,
            MAX_RESULTS,
            CACHE_TTL,
            CACHE_ENABLED,
            EMBEDDING_MODEL,
            REPORT_FORMAT,
            REPORT_FILENAME_TEXT,
            MAX_SEARCH_QUERIES,
            MAX_SEARCH_RESULTS,
            MAX_CONCURRENT_SCRAPES,
            MAX_SEARCH_RETRIES,
            MAX_AI_ITERATIONS,
            DEFAULT_USER_AGENT,
            DEFAULT_REFERER,
            URL_TIMEOUT,
            SKIP_EXTENSIONS,
            BLOCKED_DOMAINS,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            # Hybrid retrieval configuration
            USE_HYBRID_RETRIEVAL,
            RETRIEVAL_TOP_K,
            HYBRID_VECTOR_WEIGHT,
            HYBRID_BM25_WEIGHT,
            HYBRID_FUSION_METHOD,
            HYBRID_RRF_K,
            VECTOR_SCORE_THRESHOLD,
            MIN_CHUNK_LENGTH,
            MIN_WORD_COUNT,
            USE_RERANKING,
            RERANKER_CANDIDATES_MULTIPLIER,
            SCRAPER_ENABLE_PLAYWRIGHT,
            # Cross-encoder reranking configuration
            USE_CROSS_ENCODER_RERANKING,
            CROSS_ENCODER_MODEL,
            CROSS_ENCODER_TOP_K,
            RERANK_TOP_K,
            CROSS_ENCODER_BATCH_SIZE,
            # Enhanced embedding configuration
            USE_ENHANCED_EMBEDDINGS,
            EMBEDDING_TASK_TYPE,
            EMBEDDING_DIMENSIONALITY,
            EMBEDDING_NORMALIZE,
            EMBEDDING_BATCH_SIZE,
            # Multi-query retrieval settings
            USE_MULTI_QUERY_RETRIEVAL,
            MAX_RETRIEVAL_QUERIES,
            QUERY_CHUNK_DISTRIBUTION,
    )
except ImportError:
    logger.warning("Could not import config settings. Using defaults.")
    
    logger.warning("Imports from config failed. Using defaults.")
    USE_PERSISTENCE = False
    MAX_RESULTS = 5
    CACHE_TTL = 3600
    CACHE_ENABLED = False
    EMBEDDING_MODEL = "gemini-embedding-001"
    REPORT_FORMAT = "md"
    REPORT_FILENAME_TEXT = "CrystalSearchReport.txt"
    MAX_SEARCH_QUERIES = 5
    MAX_SEARCH_RESULTS = 10
    MAX_CONCURRENT_SCRAPES = 4
    MAX_SEARCH_RETRIES = 2
    MAX_AI_ITERATIONS = 1
    MAX_USER_QUERY_LOOPS = 1
    DEFAULT_USER_AGENT = "intellISearch-bot/1.0"
    DEFAULT_REFERER = "https://www.google.com"
    URL_TIMEOUT = 45    
    SKIP_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3', '.zip', '.exe']
    BLOCKED_DOMAINS = []
    # Hybrid retrieval fallback defaults
    USE_HYBRID_RETRIEVAL = True
    RETRIEVAL_TOP_K = 20
    HYBRID_VECTOR_WEIGHT = 0.6
    HYBRID_BM25_WEIGHT = 0.4
    HYBRID_FUSION_METHOD = "rrf"
    HYBRID_RRF_K = 60
    VECTOR_SCORE_THRESHOLD = 0.1
    MIN_CHUNK_LENGTH = 50
    MIN_WORD_COUNT = 10
    USE_RERANKING = False
    RERANKER_CANDIDATES_MULTIPLIER = 3
    # Enhanced embedding fallback defaults
    USE_ENHANCED_EMBEDDINGS = True
    EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
    EMBEDDING_DIMENSIONALITY = 768
    EMBEDDING_NORMALIZE = True
    EMBEDDING_BATCH_SIZE = 50
    # Multi-query retrieval fallback defaults
    USE_MULTI_QUERY_RETRIEVAL = True
    MAX_RETRIEVAL_QUERIES = 5
    QUERY_CHUNK_DISTRIBUTION = True
    # Color constants and chunking fallbacks
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

class AgentState(TypedDict):
    session_id: Optional[str]
    new_query: Optional[str]
    analytical_questions: Optional[List[str]]
    search_queries: Optional[List[str]]
    rationale: Optional[str]
    data: Optional[List[Any]]
    relevant_contexts: Optional[Dict[str, Dict[str, str]]]
    relevant_chunks: Optional[List[Document]]
    retriever_responses: Optional[Dict[str, str]]
    qa_pairs: Optional[List[Dict]]
    all_citations: Optional[List[Dict]]
    proceed: Optional[bool]
    visited_urls: Optional[List[str]]
    failed_urls: Optional[List[str]]
    iteration_count: Optional[int]
    error: Optional[str]
    evaluation_response: Optional[str]
    suggested_follow_up_queries: Optional[List[str]]
    prompt_type: Optional[str]
    search_iteration_count: Optional[int]
    snippet_state: Optional[Dict[str, str]]
    analysis_content: Optional[str]
    appendix_content: Optional[str]
    analysis_filename: Optional[str]
    appendix_filename: Optional[str]
    requires_new_search: Optional[bool]
    max_search_queries: Optional[int]
    max_search_results: Optional[int]
    max_ai_iterations: Optional[int]


# Pydantic models for LLM output validation (moved from initial cells)
class SearchQueryResponse(BaseModel):
    """Represents the expected JSON structure from the create_queries LLM call."""
    rationale: Optional[str] = Field(default=None, description="The rationale for the generated search queries.")
    query: List[str] = Field(default_factory=list, min_length=1, description="A list of search queries for web search.")
    analytical_questions: List[str] = Field(default_factory=list, description="A list of deep, analytical questions to be answered from the scraped content.")

class EvaluationResponse(BaseModel):
    """Represents the expected JSON structure from the AI_evaluate LLM call."""
    is_sufficient: bool = Field(description="Whether the extracted information is sufficient to answer the query.")
    knowledge_gap: str = Field(description="Description of the knowledge gap if the information is not sufficient.")
    follow_up_queries: List[str] = Field(description="A list of follow-up queries if the information is not sufficient.")
    coverage_assessment: Optional[str] = Field(default=None, description="String explaining how well the Q&A pairs address the original query.")

class FollowUpRoutingResponse(BaseModel):
    """Pydantic model for the conversational routing decision."""
    requires_new_search: bool = Field(description="True if the follow-up question requires a new web search, False if it can be answered from the provided context.")
    reasoning: str = Field(description="A brief explanation for the routing decision.")

# Consolidated helper and evaluation implementation
import hashlib
from typing import List

def hash_snippet(url: str, snippet: str) -> str:
    return hashlib.sha256(f"{url}|{snippet}".encode()).hexdigest()

# --- Node Functions ---
async def check_infrastructure(state: AgentState) -> AgentState:
    """
    Checks if the required infrastructure (API keys, LLM connectivity) is available.
    This acts as a 'pre-flight' check at the start of a session.
    """
    logger.info("--- Checking Infrastructure Health ---")
    missing_keys = []
    if not GOOGLE_API_KEY: missing_keys.append("GOOGLE_API_KEY")
    if not SERPER_API_KEY: missing_keys.append("SERPER_API_KEY")

    if missing_keys:
        state["error"] = f"Infrastructure check failed: Missing {', '.join(missing_keys)}"
        logger.error(state["error"])
    return state

#=============================================================================================
async def start_research(state: AgentState) -> AgentState:
    """
    Initialize a new research session. This creates a `session_id`, sets
    initial state fields, and records a minimal session document in Firestore
    when available. This node is intended to run at the start of a new
    research workflow when there is no prior conversation context.
    """
    # Respect incoming session_id if provided; otherwise create a new one
    state = state or {}
    if state.get("session_id"):
        sid = state.get("session_id")
    else:
        import uuid
        sid = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        state["session_id"] = sid
    state.setdefault("visited_urls", [])
    state.setdefault("failed_urls", [])
    state.setdefault("data", [])
    state.setdefault("relevant_contexts", {})
    state.setdefault("qa_pairs", [])
    state.setdefault("suggested_follow_up_queries", [])
    state.setdefault("proceed", True)

    # Try to create a minimal Firestore document so frontends can observe session metadata
    try:
        from google.cloud import firestore
        fs = firestore.Client()
        doc = {
            "created_at": datetime.now(),
            "query": state.get("new_query", ""),
            "current_step": "started",
            "progress": 0,
        }
        try:
            fs.collection("research_sessions").document(sid).set(doc)
            logger.info("Created Firestore session doc for %s", sid)
        except Exception as e:
            logger.debug("Could not write session doc to Firestore: %s", e)
    except Exception:
        logger.debug("Firestore client not available; skipping session persistence.")

    logger.info("start_research: initialized session %s", sid)
    return state
#=============================================================================================
async def create_queries(state: AgentState) -> AgentState:
    """
    Uses the user input from the initial state to generate rationale and a list of queries using LLM.
    Uses Pydantic for robust parsing of LLM output and includes error handling.
    Also checks for and uses suggested_follow_up_queries if available.
    """
    # Get prompt type from state
    # Prioritize suggested_follow_up_queries if they exist and we are in a refinement loop
    suggested_queries = state.get("suggested_follow_up_queries")
    new_query = state.get("new_query")
    current_iteration = state.get("iteration_count", 0)

    generated_search_queries = set()
    rationale = ""
    analytical_questions = []
    error = None

    # If there are suggested follow-up queries from a previous AI evaluation and we haven't exceeded max iterations,
    # use them directly instead of asking the LLM to generate new ones based on the original query.
    if suggested_queries and current_iteration < MAX_AI_ITERATIONS:
         logger.debug("Using %d suggested follow-up queries from previous iteration.", len(suggested_queries))
         generated_search_queries.update(suggested_queries)
         rationale = f"Refining search based on the previous evaluation's suggested queries ({len(suggested_queries)} queries)."
         # Clear suggested_follow_up_queries after using them
         state["suggested_follow_up_queries"] = []
         state["search_queries"] = list(generated_search_queries)
         # In a refinement loop, analytical questions should be the same as search queries
         state["analytical_questions"] = list(generated_search_queries)
         state["rationale"] = rationale
         state["error"] = None # Clear previous error if using suggested queries
         return state


    # Proceed with initial query generation if no suggested queries or max iterations reached
    logger.info("Generating initial search queries based on user query: %s", new_query)

    # Use dynamic config from state, fallback to global config
    number_queries = state.get("max_search_queries", MAX_SEARCH_QUERIES)

    if new_query and llm_call_async: # Ensure llm_call_async is available
        # LLM-based research type routing: infer type from query
        lowered = new_query.lower()
        query_writer_instructions = query_writer_instructions_general # Default
        if "legal" in lowered:
            query_writer_instructions = query_writer_instructions_legal
            state["prompt_type"] = "legal"
            logger.debug("Routing to legal research based on query content.")
        elif any(word in lowered for word in ["share", "equity", "stock", "investment", "fund", "portfolio"]):
            query_writer_instructions = query_writer_instructions_investment
            state["prompt_type"] = "investment"
            logger.debug("Routing to investment research based on query content.")
        elif "deepsearch" in lowered or "deep search" in lowered:
            query_writer_instructions = query_writer_instructions_deepsearch
            state["prompt_type"] = "deepsearch"
            logger.debug("Routing to deepsearch research based on query content.")
        elif any(word in lowered for word in ["gold", "equities", "bonds", "crude oil", "commodities", "copper", "us dollar", "indian rupee", "japanese yen", "euro"]):
            query_writer_instructions = query_writer_instructions_macro
            state["prompt_type"] = "macro"
            logger.debug("Routing to macro research based on query content.")
        elif re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", new_query):
            query_writer_instructions = query_writer_instructions_person_search
            state["prompt_type"] = "person_search"
            logger.debug("Routing to person research based on query content.")

        messages = [SystemMessage(content="""
        You are an expert research analyst. Your goal is to generate two sets of questions based on a user's query.

        1.  **Search Queries**: A list of {number_queries} concise queries for a web search engine. These should be keyword-focused.
        2.  **Analytical Questions**: A list of 10 deeper, more comprehensive questions. These will be used later to interrogate the scraped web content and form the basis of the final report. They should be more detailed and analytical than the search queries.

        **Topic:** {topic}
        **Current Date:** {current_date}

        Respond with a single JSON object containing 'rationale', 'query' (for search), and 'analytical_questions'.
        """.format(
            number_queries=number_queries,
            current_date=get_current_date(),
            topic=new_query
        )),
        HumanMessage(content=f"User Query: {new_query}\n\nPlease provide the JSON object as instructed.")
        ]

        try:
            # Use the general llm_call_async utility
            response = await llm_call_async(messages) 

            if response and isinstance(response, str):
                # Use a more robust regex to find the JSON block
                json_match = re.search(r'```json\s*(\{.*\})\s*```|(\{.*\})', response, re.DOTALL)

                if json_match:
                    json_string = json_match.group(1) if json_match.group(1) else json_match.group(2)
                    # Clean the JSON string: remove trailing commas before brackets/braces and control characters
                    cleaned_json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
                    cleaned_json_string = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_json_string)


                    # Using model_validate_json for Pydantic V2 compatibility
                    try:
                        parsed_response = SearchQueryResponse.model_validate_json(cleaned_json_string)
                        rationale = parsed_response.rationale or "No rationale provided."
                        search_queries = parsed_response.query
                        analytical_questions = parsed_response.analytical_questions


                        if isinstance(search_queries, list) and all(isinstance(q, str) for q in search_queries):
                            generated_search_queries.update(search_queries)
                            logger.info("Generated %d search queries.", len(generated_search_queries))
                        else:
                            # This case should ideally be caught by Pydantic validation, but keeping as a safeguard
                            error = "LLM response 'query' key is not a valid list of strings after Pydantic parsing."
                            logger.error(f"{error} Response: {response.content}")

                        if isinstance(analytical_questions, list) and all(isinstance(q, str) for q in analytical_questions):
                            logger.info("Generated %d analytical questions.", len(analytical_questions))
                            print (analytical_questions)
                            # analytical_questions is a list; no set-like update is needed here.
                            # Keep the parsed list as-is for downstream processing.
                        else:
                            error = "LLM response 'analytical_questions' key is not a valid list of strings after Pydantic parsing."
                            logger.error(f"{error} Response: {response.content}")

                    except ValidationError as e:
                        error = f"Pydantic validation error parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logger.error(error)
                    except json.JSONDecodeError as e:
                        error = f"JSON decoding error parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logger.error(error)
                    except Exception as e:
                        error = f"An unexpected error occurred parsing search queries: {e}. JSON String: {cleaned_json_string}"
                        logger.error(error)

                else:
                    error = "Could not find JSON block in LLM response for query generation."
                    logger.error(f"{error} Response: {response}")


            else:
                error = "No or invalid response received from LLM for query generation."
                logger.error(error)

        except Exception as e:
            error = f"An unexpected error occurred during LLM call for query generation: {e}"
            logger.error(error)


    else:
        if not new_query:
            error = "No initial query provided in state."
            logger.warning(error)
        elif not llm_call_async:
            error = "Primary LLM is not initialized. Cannot generate queries."
            logger.error(error)
        else:
            error = "Prompt instructions not loaded. Cannot generate queries."
            logger.error(error)


    state['rationale'] = rationale if rationale else "No rationale generated."
    state['search_queries'] = list(generated_search_queries) if generated_search_queries else []
    state['analytical_questions'] = analytical_questions if analytical_questions else state['search_queries']

    # Append new error to existing error state
    current_error = state.get('error', '') or ''
    state['error'] = (current_error + "\n" + error).strip() if error else current_error.strip()
    state['error'] = None if state['error'] == "" else state['error']

    # Ensure suggested_follow_up_queries is cleared if we generated new queries
    state["suggested_follow_up_queries"] = []

    return state
#=============================================================================================
async def route_follow_up_question(state: AgentState) -> AgentState:
    """
    Determines if a follow-up question can be answered from existing context (qa_pairs)
    or if it requires a new web search. This node acts as the entry-point router.
    """
    logger.debug("--- Entering route_follow_up_question node ---")
    follow_up_query = state.get("new_query")
    qa_pairs = state.get("qa_pairs", [])

    # If there's no previous context (qa_pairs), it's a new conversation. A search is required.
    if not qa_pairs:
        logger.debug("No existing context (qa_pairs) found. Routing to new search.")
        state["requires_new_search"] = True
        return state

    # If there is context, use an LLM to decide.
    logger.debug("Existing context found. Evaluating if new search is needed for follow-up.")

    qa_summary = "\n\n".join([
        f"**Q: {pair.get('question', 'N/A')}**\n**A:** {pair.get('answer', 'N/A')[:300]}..."
        for pair in qa_pairs
    ])

    prompt = f"""
    You are a routing agent. Your task is to determine if a user's follow-up question can be answered using only the context from the previous conversation turn.

    **Previous Conversation Context (Summarized Q&A):**
    ---
    {qa_summary}
    ---

    **User's Follow-up Question:**
    "{follow_up_query}"

    **Analysis Task:**
    1.  Carefully read the user's follow-up question.
    2.  Review the summarized Q&A context.
    3.  Decide if the context contains enough information to form a comprehensive answer to the follow-up question. The answer doesn't have to be explicitly stated, but the necessary information must be present.

    **Output your decision as a JSON object with the following structure:**
    {{
      "requires_new_search": boolean,
      "reasoning": "A brief explanation of why you made this decision."
    }}

    - Set "requires_new_search" to `false` if the question can be answered from the context.
    - Set "requires_new_search" to `true` if the question introduces new topics, asks for more recent information, or requires details not present in the context.
    """

    messages = [SystemMessage(content="You are an intelligent routing agent."), HumanMessage(content=prompt)]

    try:
        response_text = await llm_call_async(messages)
        if not response_text:
            raise ValueError("LLM returned an empty response.")

        json_match = re.search(r'```json\s*(\{.*\})\s*```|(\{.*\})', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("Could not find a valid JSON block in the LLM response.")

        json_string = json_match.group(1) if json_match.group(1) else json_match.group(2)
        routing_decision = FollowUpRoutingResponse.model_validate_json(json_string)

        state["requires_new_search"] = routing_decision.requires_new_search
        logger.debug(f"Routing decision: requires_new_search = {routing_decision.requires_new_search}. Reasoning: {routing_decision.reasoning}")

    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse routing decision from LLM, defaulting to new search. Error: {e}")
        state["requires_new_search"] = True # Default to performing a new search on error
        state["error"] = f"Routing Error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred in routing, defaulting to new search. Error: {e}")
        state["requires_new_search"] = True
        state["error"] = f"Unexpected Routing Error: {e}"

    return state

    # user_approval_for_queries and choose_report_type functions removed as requested
#=============================================================================================
async def fast_search_results_to_final_urls(state: AgentState) -> AgentState:
    """
    Go straight from search results to deduplication and save results in final_urls (no LLM evaluation).
    """
    search_queries = state.get("search_queries", []) or []
    existing_data = state.get("data", []) or []
    visited_urls = set(state.get("visited_urls", []) or [])
    failed_urls = set(state.get("failed_urls", []) or [])
    errors = []
    any_cache_hit = False

    if not search_queries:
        logger.warning("No search queries found to evaluate.")
        state.update({
            "data": existing_data,
            "visited_urls": list(visited_urls),
            "error": state.get('error'),
        })
        return state

    if not UnifiedSearcher:
        error_msg = "UnifiedSearcher class not available. Cannot perform search."
        logger.error(error_msg)
        state.update({
            "data": [],
            "visited_urls": list(visited_urls),
            "error": error_msg,
        })
        return state

    # Use dynamic config from state for max_results, fallback to global config
    max_results_per_query = state.get("max_search_results", MAX_SEARCH_RESULTS)
    search_engine = UnifiedSearcher(max_results=max_results_per_query)

    # Assuming search_engine.search now returns a tuple: (results, was_cached)
    async def search_with_cache_tracking(query):
        return await search_engine.search(query)

    search_tasks = [search_with_cache_tracking(q) for q in search_queries]
    search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

    all_results = []
    for i, result_set in enumerate(search_results_list):
        query = search_queries[i] if i < len(search_queries) else f"<unknown_{i}>"
        if isinstance(result_set, Exception):
            errors.append(f"Search failed for query '{query}': {result_set}")
            continue

        # Unpack the tuple: (results, was_cached)
        results, was_cached = result_set if isinstance(result_set, tuple) else (result_set, False)
        if was_cached:
            any_cache_hit = True

        if not results:
            # No search results for this specific query — log as warning but
            # do not treat this as a workflow-level error. Empty results are
            # common for narrow queries and should not cause the overall
            # research session to be marked as failed when other queries may
            # have returned usable data.
            logger.warning(f"No results for query: {query}")
            continue
        for r in results:
            url = getattr(r, 'url', None)
            # Filter out blocked domains
            if url and url not in visited_urls and url not in failed_urls:
                if any(domain in url.lower() for domain in BLOCKED_DOMAINS):
                    logger.debug(f"Skipping blocked domain URL: {url}")
                    continue
                all_results.append(r)
                visited_urls.add(url)

    # Deduplicate by URL
    deduplicated = {item.url: item for item in existing_data + all_results if hasattr(item, 'url') and not any(domain in item.url.lower() for domain in BLOCKED_DOMAINS)}
    final_data = list(deduplicated.values())
    final_urls = [item.url for item in final_data if hasattr(item, 'url')]
    state.update({
        "data": final_data,
        "visited_urls": list(visited_urls),
        "used_cache": any_cache_hit, # Set the cache status explicitly in the state
        "final_urls": final_urls,
        "error": "\n".join(errors) if errors else None
    })
    logger.debug(f"fast_search_results_to_final_urls: Final URLs count: {len(final_urls)}")
    
    
    return state
#============#==============#=============
from .scraper import Scraper
from .logging_setup import logger
import asyncio, aiohttp
SHARED_SCRAPER = Scraper()
def create_shared_scraper():
    return SHARED_SCRAPER

#=============================================================================================
async def extract_content(state: AgentState) -> AgentState:
    from .config import URL_TIMEOUT
    data = state.get('data', []) # Original search results including snippets
    relevant_contexts = {}
    errors = []
    url_timeout = URL_TIMEOUT # Use timeout from config
    
    # Check if data is empty
    if not data:
        logger.debug("No data found to extract content from.")
        state["relevant_contexts"] = {} # Ensure relevant_contexts is initialized
        # Append new errors to existing ones in state
        current_error = state.get('error', '') or ''
        state['error'] = (current_error + "\nNo data found to extract content from.").strip()
        state['error'] = None if state['error'] == "" else state['error']
        return state

    # Rank URLs based on collected data (Assuming data contains SearchResult objects with url and snippet)
    valid_data = [item for item in data if isinstance(item, SearchResult) and item.url and item.snippet]

    if not valid_data:
         logger.debug("No valid data found for ranking.")
         state["relevant_contexts"] = {}
         # Append new errors to existing ones in state
         current_error = state.get('error', '') or ''
         state['error'] = (current_error + "\nNo valid data found for ranking.").strip()
         state['error'] = None if state['error'] == "" else state['error']
         return state

    context_for_ranking = {item.url: item.snippet for item in valid_data}

    # Use the original new_query for ranking relevance if available, otherwise use the first search query
    ranking_query = state.get("new_query", state.get("search_queries", [None])[0])
    if ranking_query:
        # Use the imported rank_urls
        ranked_urls = rank_urls(ranking_query, [item.url for item in valid_data], context_for_ranking)
    else:
        logger.debug("No query available for ranking URLs. Proceeding without ranking.")
        ranked_urls = [item.url for item in valid_data] # Use original order if no query

    urls_to_process = ranked_urls[:30] # limit to top 30 urls
    logger.debug("Relevant and ranked URLs for extraction: %s", urls_to_process)

    # Get the list of previously failed URLs
    failed_urls = state.get('failed_urls', []) or []
    
    # Filter out failed URLs from processing
    urls_to_process = [url for url in urls_to_process if url not in failed_urls]
    if len(ranked_urls[:30]) > len(urls_to_process):
        skipped_count = len(ranked_urls[:30]) - len(urls_to_process)
        logger.debug("Skipped %d previously failed URLs", skipped_count)

    # 2. Batch Processing with Shared Session
    relevant_contexts = {}
    failed_for_static = []

    # We'll collect a unified `results` list of tuples: (url, content_dict_or_None, success_bool, strategy)
    results: List[tuple] = []

    # Pre-flight: validate URLs with Scraper._is_valid_url to skip blocked/unsupported URLs
    try:
        validity_checks = await asyncio.gather(*[SHARED_SCRAPER._is_valid_url(u) for u in urls_to_process])
    except Exception:
        # If validation itself fails, fall back to attempting all URLs
        logger.debug("URL validation failed; proceeding with all candidate URLs", exc_info=True)
        validity_checks = [True] * len(urls_to_process)

    validated_urls = [u for u, ok in zip(urls_to_process, validity_checks) if ok]
    invalid_urls = [u for u, ok in zip(urls_to_process, validity_checks) if not ok]
    if invalid_urls:
        logger.debug("Skipping %d invalid/blocked URLs before scraping: %s", len(invalid_urls), invalid_urls)
        # Persist skipped URLs into failed_urls so they aren't retried in this session
        current_failed = set(state.get('failed_urls', []) or [])
        state['failed_urls'] = list(current_failed.union(set(invalid_urls)))

    urls_to_process = validated_urls
    # Separate PDFs from HTML so we can handle PDFs with the scraper's PDF path
    pdf_urls = [u for u in urls_to_process if isinstance(u, str) and u.lower().endswith('.pdf')]
    html_urls = [u for u in urls_to_process if u not in pdf_urls]

    # Process PDFs first with bounded concurrency using SHARED_SCRAPER.scrape (routes to _scrape_pdf)
    if pdf_urls:
        logger.info(f"Processing {len(pdf_urls)} PDF URLs via scraper PDF path")
        pdf_sem = asyncio.Semaphore(3)
        async def proc_pdf(u: str):
            async with pdf_sem:
                try:
                    res = await SHARED_SCRAPER.scrape(u)
                except Exception as e:
                    logger.debug(f"PDF scrape error for {u}: {e}")
                    res = None
                if not res:
                    results.append((u, None, False, 'pdf'))
                    return
                if getattr(res, 'success', False) and getattr(res, 'text', None):
                    results.append((res.url, {"content": res.text[:15000], "title": getattr(res, 'title', 'Untitled')}, True, getattr(res, 'strategy', 'pdf')))
                    logger.info("[EXTRACT SUCCESS][pdf] Extracted %d chars from %s", len(getattr(res,'text','')), res.url)
                else:
                    # fallback to snippet if available
                    original = next((item for item in valid_data if item.url == getattr(res,'url', u)), None)
                    if original and getattr(original, 'snippet', None):
                        results.append((getattr(res,'url', u), {"content": original.snippet, "title": getattr(original, 'title', 'Untitled')}, False, getattr(res, 'strategy', 'pdf')))
                        logger.info("[EXTRACT FALLBACK][pdf] Using snippet for %s", getattr(res,'url', u))
                    else:
                        results.append((getattr(res,'url', u), None, False, getattr(res, 'strategy', 'pdf')))

        await asyncio.gather(*[proc_pdf(u) for u in pdf_urls])

    # Now proceed with HTML/static URLs
    urls_to_process = html_urls

    # --- PASS 1: LIGHTWEIGHT (aiohttp) ---
    logger.info(f"Pass 1: Attempting static scrape for {len(urls_to_process)} URLs")
    async with aiohttp.ClientSession() as session:
        static_tasks = [SHARED_SCRAPER._scrape_with_aiohttp(url, session=session) for url in urls_to_process]
        static_results = await asyncio.gather(*static_tasks)

    for res in static_results:
        if not res:
            continue
        url = getattr(res, 'url', None)
        text = getattr(res, 'text', None) or ''
        title = getattr(res, 'title', None) or 'Untitled'
        success = bool(getattr(res, 'success', False))
        # If successful AND content is substantial, keep it
        if success and len(text) > 600:
            content = {"content": text[:15000], "title": title}
            results.append((url, content, True, getattr(res, 'strategy', 'aiohttp')))
            logger.info("[EXTRACT SUCCESS] Extracted %d chars from %s", len(text), url)
        else:
            # Mark as failed for fallback processing
            results.append((url, None, False, getattr(res, 'strategy', 'aiohttp')))
            failed_for_static.append(url)
            logger.info("[EXTRACT FAIL] Static scrape failed or insufficient for %s", url)

    # --- PASS 2: HEAVYWEIGHT FALLBACK (Playwright) ---
    if failed_for_static:
        logger.info(f"Pass 2: Attempting Playwright fallback for {len(failed_for_static)} URLs")

        # The SHARED_SCRAPER.scrape(..., dynamic=True) uses the internal Semaphore(3)
        pw_tasks = [SHARED_SCRAPER.scrape(url, dynamic=True) for url in failed_for_static]
        pw_results = await asyncio.gather(*pw_tasks)

        # Update existing results entries with Playwright outcomes
        for res in pw_results:
            if not res:
                continue
            url = getattr(res, 'url', None)
            text = getattr(res, 'text', None) or ''
            title = getattr(res, 'title', None) or 'Untitled'
            success = bool(getattr(res, 'success', False))

            # Find index in results and replace tuple if present
            for idx, (r_url, _, _, _) in enumerate(results):
                if r_url == url:
                    if success and len(text) > 0:
                        results[idx] = (url, {"content": text[:15000], "title": title}, True, getattr(res, 'strategy', 'playwright'))
                        logger.info("[EXTRACT SUCCESS][playwright] Extracted %d chars from %s", len(text), url)
                    else:
                        # Fallback to search snippet if both methods failed
                        original = next((item for item in valid_data if item.url == url), None)
                        if original and getattr(original, 'snippet', None):
                            results[idx] = (url, {"content": original.snippet, "title": getattr(original, 'title', 'Untitled')}, False, getattr(res, 'strategy', 'playwright'))
                            logger.info("[EXTRACT FALLBACK] Using snippet for %s", url)
                        else:
                            # leave as failed (None content)
                            results[idx] = (url, None, False, getattr(res, 'strategy', 'playwright'))
                    break
            else:
                # If not found in results for some reason, append
                if success and len(text) > 0:
                    results.append((url, {"content": text[:15000], "title": title}, True, getattr(res, 'strategy', 'playwright')))
                else:
                    original = next((item for item in valid_data if item.url == url), None)
                    if original and getattr(original, 'snippet', None):
                        results.append((url, {"content": original.snippet, "title": getattr(original, 'title', 'Untitled')}, False, getattr(res, 'strategy', 'playwright')))
                    else:
                        results.append((url, None, False, getattr(res, 'strategy', 'playwright')))

    # After both passes, assemble relevant_contexts and compute stats from unified `results`

    # 3. Context assembly and summary
    # total should reflect all validated URLs (including PDFs)
    total = len(validated_urls)
    success_count = 0
    aiohttp_count = 0
    playwright_count = 0
    pdf_count = 0
    aiohttp_success = 0
    playwright_success = 0
    pdf_success = 0

    # Aggregate results into contexts and counters
    for url, content, success, strategy in results:
        if strategy == 'aiohttp':
            aiohttp_count += 1
            if success:
                aiohttp_success += 1
        elif strategy == 'playwright':
            playwright_count += 1
            if success:
                playwright_success += 1
        elif strategy == 'pdf':
            pdf_count += 1
            if success:
                pdf_success += 1

        if content:
            relevant_contexts[url] = content
            if success:
                success_count += 1

    failed_count = total - success_count

    # Update state's failed_urls with any URLs that still have no content
    newly_failed = [url for url, content, success, _ in results if not content]
    if newly_failed:
        current_failed = set(state.get('failed_urls', []) or [])
        updated_failed = list(current_failed.union(set(newly_failed)))
        state['failed_urls'] = updated_failed
        logger.debug("Added %d newly failed URLs to state.failed_urls", len(newly_failed))

    # Attach a concise scrape summary to the state for downstream reporting
    state['scrape_summary'] = {
        'total_urls': total,
        'successful_urls': success_count,
        'failed_urls': failed_count,
        'aiohttp_total': aiohttp_count,
        'aiohttp_success': aiohttp_success,
        'playwright_total': playwright_count,
        'playwright_success': playwright_success,
        'pdf_total': pdf_count,
        'pdf_success': pdf_success,
    }
    # Log a concise, human-readable scrape summary so it appears in app.log
    try:
        ss = state['scrape_summary']
        logger.info(
            "Scrape summary for session %s: total=%d successful=%d failed=%d | aiohttp=%d/%d playwright=%d/%d pdf=%d/%d",
            state.get('session_id', 'unknown'),
            ss.get('total_urls', 0), ss.get('successful_urls', 0), ss.get('failed_urls', 0),
            ss.get('aiohttp_success', 0), ss.get('aiohttp_total', 0),
            ss.get('playwright_success', 0), ss.get('playwright_total', 0),
            ss.get('pdf_success', 0), ss.get('pdf_total', 0),
        )
    except Exception:
        # Avoid disrupting workflow if logging fails for any reason
        logger.debug("Failed to write scrape summary log entry", exc_info=True)
    
    state["relevant_contexts"] = relevant_contexts
        # Try to update Firestore for the active session to reflect scraping completion
    try:
        from google.cloud import firestore
        try:
            fs = firestore.Client()
            if state.get('session_id'):
                try:
                    fs.collection("research_sessions").document(state.get('session_id')).update({
                        "processing_urls": [],
                        "current_step": "gathering_sources",
                        "updated_at": datetime.now(),
                        "progress": 60,
                    })
                except Exception as _e:
                    logger.debug(f"Could not update Firestore after scraping for session {state.get('session_id')}: {_e}")
        except Exception:
            logger.debug("Firestore client initialization failed when updating scraping completion.")
    except Exception:
        pass
    return state

#=============================================================================================
async def fss_retrieve(state: dict) -> dict:
    """
    Creates a Gemini File Search Store, uploads the relevant contexts,
    generates an answer using the store, and then cleans up the store.
    The generated answer is stored in 'analysis_content'.
    """
    contexts_to_use = state.get("relevant_contexts", {})
    questions_to_answer = state.get("analytical_questions") or state.get("search_queries", [])
    session_id = state.get("session_id", "default-session")

    logger.debug(f"[FSS Node] fss_retrieve received retrieval_method: '{state.get('retrieval_method')}'")
    logger.debug(f"[FSS Node] Entering fss_retrieve for {len(questions_to_answer)} questions with {len(contexts_to_use)} contexts.")
    logger.debug(f"[DEBUG][fss_retrieve] questions: {questions_to_answer[:3]}... (total {len(questions_to_answer)}) | contexts: {list(contexts_to_use.keys())[:3]}... (total {len(contexts_to_use)})")

    try:
        if not contexts_to_use or not questions_to_answer:
            logger.warning("[FSS Node] No relevant contexts or queries available. Skipping FSS batch QA.")
            state["qa_pairs"] = []
            return state

        if not GeminiFileSearchRetriever:
            logger.error("[FSS Node] GeminiFileSearchRetriever class not available.")
            raise RuntimeError("GeminiFileSearchRetriever not available.")

        retriever = GeminiFileSearchRetriever(display_name_prefix=f"crystal-{session_id}")
        qa_dict = await retriever.answer_batch_questions(questions_to_answer, contexts_to_use)
        logger.debug(f"[FSS Node] Batch QA complete. Got {len(qa_dict)} QA pairs.")

        # Convert to list of dicts for downstream compatibility
        qa_pairs = []
        for q in questions_to_answer:
            answer = qa_dict.get(q, "No answer generated.")
            qa_pairs.append({"question": q, "answer": answer, "citations": []})


        state["qa_pairs"] = qa_pairs
        state["file_store_name"] = None
        logger.debug(f"[FSS Node] Stored {len(qa_pairs)} QA pairs in state.")

    except Exception as e:
        error_msg = f"[FSS Node] failed: {e}"
        logger.error(error_msg, exc_info=True)
        state["error"] = error_msg
        state["file_store_name"] = None
    return state

#=============================================================================================
async def classic_retrieve(state: dict) -> dict:
    """
    Uses the hybrid retriever to answer each query in search_queries using relevant_contexts.
    Outputs qa_pairs for write_report, mirroring the fss_retrieve batch QA pattern.
    """
    questions_to_answer = state.get("analytical_questions") or state.get("search_queries", [])
    contexts_to_use = state.get("relevant_contexts", {})
    session_id = state.get("session_id", "default-session")


    # Use EnhancedGoogleEmbeddings from enhanced_embeddings.py
    try:
        from backend.src.enhanced_embeddings import EnhancedGoogleEmbeddings
        from .config import GOOGLE_API_KEY, EMBEDDING_MODEL
        embeddings = EnhancedGoogleEmbeddings(google_api_key=GOOGLE_API_KEY, model=EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"[Classic Node] Could not initialize EnhancedGoogleEmbeddings: {e}")
        embeddings = None

    logger.info(f"[Classic Node] classic_retrieve received {len(questions_to_answer)} questions and {len(contexts_to_use)} contexts.")
    
    try:
        if not contexts_to_use or not questions_to_answer:
            logger.warning("[Classic Node] No relevant contexts or queries available. Skipping hybrid batch QA.")
            state["qa_pairs"] = []
            return state
        if not create_hybrid_retriever or embeddings is None:
            logger.error("[Classic Node] Hybrid retriever or embeddings not available.")
            state["qa_pairs"] = []
            return state

        retriever = create_hybrid_retriever(embeddings=embeddings)
        if not retriever.build_index(contexts_to_use):
            logger.error("[Classic Node] Failed to build hybrid retriever index.")
            state["qa_pairs"] = []
            return state

        # Use the hybrid retriever to answer each query
        _, query_responses = retriever.retrieve_with_query_responses(questions_to_answer)
        logger.info(f"[Classic Node] Batch QA complete. Got {len(query_responses)} QA pairs.")

        # Convert to list of dicts for downstream compatibility
        qa_pairs = []
        for q in questions_to_answer:
            answer = query_responses.get(q, "No answer generated.")
            qa_pairs.append({"question": q, "answer": answer, "citations": []})

        state["qa_pairs"] = qa_pairs
        logger.info(f"[Classic Node] Stored {len(qa_pairs)} QA pairs in state.")

    except Exception as e:
        error_msg = f"[Classic Node] failed: {e}"
        logger.error(error_msg, exc_info=True)
        state["error"] = error_msg
    return state

#=============================================================================================
async def AI_evaluate(state: AgentState) -> AgentState:
    """
    Evaluates the Q&A pairs to determine if they provide sufficient depth to answer the original user query.
    Updates 'proceed' based on AI assessment of Q&A pair coverage.
    Suggests follow-up queries if the Q&A pairs don't adequately address the original question.
    Tracks search_iteration_count to prevent infinite loops.
    """

    logger.debug(f"[DEBUG][AI_evaluate] state['qa_pairs'] length: {len(state.get('qa_pairs', []))}")
    qa_pairs = state.get("qa_pairs", [])
    original_query = state.get("new_query", "")

    # This ensures the evaluation is based on the most refined set of questions.
    questions_being_answered = state.get("analytical_questions") or state.get("search_queries", [])
    questions_context = "\n".join([f"- {q}" for q in questions_being_answered])
    
    logger.info("AI Evaluation started: evaluating %d Q&A pairs against %d questions.", len(qa_pairs), len(questions_being_answered))

    state["search_iteration_count"] = state.get("search_iteration_count", 0) + 1
    # Use dynamic config from state, fallback to global config
    max_iterations = state.get("max_ai_iterations", MAX_AI_ITERATIONS)
    errors = []
    state["proceed"] = True

    if not llm_call_async:
        msg = "LLM not initialized. Skipping AI evaluation."
        errors.append(msg)
        logger.error(msg)
        state['error'] = msg
        return state

    if not qa_pairs:
        msg = f"No Q&A pairs available for evaluation ({state['search_iteration_count']}/{max_iterations})."
        logger.warning(msg)
        if state["search_iteration_count"] < max_iterations:
            state["proceed"] = False
            state['suggested_follow_up_queries'] = []
            state['knowledge_gap'] = "No Q&A pairs generated this round."
        else:
            state["proceed"] = True
            state['knowledge_gap'] = "Max iterations reached. Proceeding with current Q&A pairs."
        state['error'] = msg
        return state

    # Format Q&A pairs for evaluation
    qa_text = "\n\n".join([
        f"**Q{i+1}: {pair['question']}**\n**A{i+1}:** {pair['answer']}"
        for i, pair in enumerate(qa_pairs)
    ])

    # Create evaluation prompt focused on Q&A coverage
    evaluation_prompt = f"""
    You are an adversarial evaluator. Your goal is to determine if the provided "Evidence" (Q&A pairs) is sufficient to answer the "Questions to be Answered".

    **Original User Query:** {original_query}

    **Questions to be Answered:**
    {questions_context}

    **Available Q&A Pairs:**
    {qa_text}

    **Adversarial Evaluation Task:**
    1. **Play Devil's Advocate**: Act as a skeptical user. Do the answers truly and comprehensively address the original query, or are they superficial?
    2. **Identify Weaknesses**: Pinpoint the weakest part of the collected information. What crucial aspect is missing or poorly explained?
    3. **Assess Sufficiency**: Based on this adversarial review, is the information sufficient to create a high-quality, trustworthy report that answers the "Questions to be Answered"?

    
    **Instructions**:
    Respond with a JSON object containing:
    {{
        "is_sufficient": boolean,
        "knowledge_gap": "string describing any gaps if not sufficient",
        "follow_up_queries": ["list of specific queries needed to fill gaps"],
        "coverage_assessment": "string explaining how well the Q&A pairs address the original query"
    }}
    """

    messages = [
        SystemMessage(content="You are an expert research analyst evaluating whether Q&A pairs provide sufficient coverage for answering a user's original query."),
        HumanMessage(content=evaluation_prompt)
    ]

    try:
        response = await llm_call_async(messages)
        response_text = response

        if not response_text:
            raise ValueError("No response received from LLM.")

        # Extract and clean JSON
        match = re.search(r'```json\s*(\{.*\})\s*```|(\{.*\})', response_text, re.DOTALL)
        if match:
            json_block = match.group(1) if match.group(1) else match.group(2)
            json_block = re.sub(r',\s*([\]}])', r'\1', json_block)
            json_block = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_block)

            eval_result = EvaluationResponse.model_validate_json(json_block)

            if eval_result.is_sufficient:
                state["proceed"] = True
                state["suggested_follow_up_queries"] = []
                state["knowledge_gap"] = ""
                logger.info("AI_evaluate: Q&A pairs provide sufficient coverage, proceeding to report.")
            else:
                state["proceed"] = False
                state["suggested_follow_up_queries"] = eval_result.follow_up_queries
                state["knowledge_gap"] = eval_result.knowledge_gap
                logger.info("AI_evaluate: Q&A pairs insufficient, generating follow-up queries.")

        else:
            raise ValueError("JSON block not found in response.")

    except Exception as e:
        logger.exception("AI_evaluate error: %s", e)
        state["proceed"] = True
        state["suggested_follow_up_queries"] = []
        state["knowledge_gap"] = f"Fallback triggered due to error: {e}"
        errors.append(str(e))

    # Final check for iteration cap
    if not state["proceed"] and state["search_iteration_count"] >= max_iterations:
        logger.warning("Max iterations reached. Forcing report generation.")
        state["proceed"] = True
        state["suggested_follow_up_queries"] = []
        state["knowledge_gap"] = "Max iterations hit. Report generated with available Q&A pairs."

    # Error handling
    if errors:
        prev_error = state.get("error", "") or ""
        state["error"] = (prev_error + "\n" + "\n".join(errors)).strip()

    return state
#=============================================================================================
def deduplicate_content(text: str) -> str:
    """
    Remove duplicate sentences and similar content from the report to reduce repetition.
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique_sentences = []
    seen_content = set()
    
    for sentence in sentences:
        # Clean and normalize the sentence for comparison
        cleaned = re.sub(r'\s+', ' ', sentence.strip().lower())
        cleaned = re.sub(r'[^\w\s]', '', cleaned)  # Remove punctuation for comparison
        
        # Skip very short sentences or headers
        if len(cleaned.split()) < 3:
            unique_sentences.append(sentence)
            continue
            
        # Check for similarity with existing content
        is_duplicate = False
        for seen in seen_content:
            # Calculate simple similarity (common words)
            words1 = set(cleaned.split())
            words2 = set(seen.split())
            if len(words1 & words2) / max(len(words1), len(words2), 1) > 0.7:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_sentences.append(sentence)
            seen_content.add(cleaned)
    
    return ' '.join(unique_sentences)
#=============================================================================================
async def write_report(state: AgentState) -> AgentState:
    """
    Generates a response based on the conversational context.
    - If it's a new search, it creates a full research report.
    - If it's a follow-up, it generates a direct answer from existing context.
    """
    errors: List[str] = []
    query = state.get('new_query', 'the topic')
    full_session_id = state.get('session_id', 'unknown_session')
    short_session_id = full_session_id.split('-')[0]
    qa_pairs = state.get("qa_pairs", [])

    analysis_content = ""
    analysis_filename = None
    appendix_content = None
    appendix_filename = None

    logger.info(f"--- Entering write_report node for session '{short_session_id}' ---")

    if not llm_call_async:
        errors.append("LLM not available to generate response.")
        analysis_content = "Error: LLM is not configured."
    elif not qa_pairs:
        analysis_content = "I was unable to find any information to answer your question."
        errors.append("No Q&A pairs were available to generate a report.")
    else:
        # Always generate a full research report from the available Q&A pairs
        qa_md = "\n\n".join([f"**Q: {qa.get('question', 'N/A')}**\nA: {qa.get('answer', 'N/A')}" for qa in qa_pairs])
        logger.info("Generating a full research report.")
        synthesis_prompt = f"""
        You are an expert research analyst. Your task is to synthesize a comprehensive, analytical report based on the provided Q&A pairs to answer the user's original query.

        **Original User Query:** {query}

        **Evidence Base (Q&A Pairs):**
        ---
        {qa_md}
        ---
        **Report Instructions:**
        1.  Write a unified, well-structured report in markdown format.
        2.  Use the Q&A pairs as your primary evidence. Do not introduce outside information.
        3.  Structure the report logically with a clear introduction, body, and a concluding summary.
        4.  Ensure the response is cohesive and directly addresses all aspects of the original query.
        5.  You must provide precise answers to queries - avoid superficial or vague statements about the topic or the research process.
        5.  The target length for this analytical section is 400-2000 words.
        6.  You must present relevant information as the user wants to understand the topic deeply.
        7.  Avoid repetition and ensure clarity throughout the report.

        GENERATE THE REPORT NOW BY FOLLOWING THE INSTRUCTIONS ABOVE.
        """
        messages = [
            SystemMessage(content="You are an expert research analyst writing a detailed report."),
            HumanMessage(content=synthesis_prompt)
        ]

        try:
            # Signal to the live session that we're entering the report-writing step
            try:
                from google.cloud import firestore
                fs_client = firestore.Client()
                try:
                    fs_client.collection("research_sessions").document(full_session_id).update({
                        "current_step": "writing_report",
                        "updated_at": datetime.now(),
                        "progress": 80,
                    })
                    logger.debug(f"Marked session {short_session_id} as 'writing_report' in Firestore.")
                except Exception as _e:
                    logger.debug(f"Could not update Firestore current_step for session {short_session_id}: {_e}")
            except Exception:
                logger.debug("Firestore client not available; skipping current_step update for writing_report.")

            intellisearch_response = await llm_call_async(messages)
            # Deduplicate the synthesized content before finalizing
            intellisearch_response = deduplicate_content(intellisearch_response)

            # Structure the full report
            part1_query = f"# Research Report\n\n## 1. Original User Query\n\n**{query}**\n\n---\n"
            part2_response = f"## 2. IntelliSearch Response\n\n{intellisearch_response}\n\n---\n"
            analysis_content = part1_query + part2_response

            # Create appendix for the full report
            appendix_md = "\n".join([f"### Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs])
            appendix_content = f"## Appendix: Q&A Pairs\n\n{appendix_md if appendix_md else 'No Q&A pairs available.'}"

        except Exception as e:
            error_msg = f"Error during LLM synthesis for full report: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            analysis_content = "Error: Failed to synthesize the final report."

    # Save analysis to Firestore
    db_client = None
    try:
        from google.cloud import firestore
        db_client = firestore.Client()
    except (ImportError, Exception):
        logger.warning("Firestore client not available. Cannot save report files.")

    if db_client:
        # include a datetime timestamp in filenames to make them unique and traceable
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if analysis_content:
            analysis_filename = f"{short_session_id}_analysis_{ts}.txt"
            try:
                db_client.collection("report_files").document(analysis_filename).set({"content": analysis_content})
                logger.info(f"Successfully saved analysis report to Firestore: {analysis_filename}")
            except Exception as e_fs:
                errors.append(f"Failed to save analysis to Firestore: {e_fs}")

        if appendix_content:
            appendix_filename = f"{short_session_id}_appendix_{ts}.txt"
            try:
                db_client.collection("report_files").document(appendix_filename).set({"content": appendix_content})
                logger.info(f"Successfully saved appendix to Firestore: {appendix_filename}")
            except Exception as e_fs:
                errors.append(f"Failed to save appendix to Firestore: {e_fs}")

    # Final state update
    current_error = state.get('error', '') or ''
    state['error'] = (current_error + "\n" + "\n".join(errors)).strip() if errors else (current_error.strip() if current_error else None)
    if state['error'] == '': state['error'] = None

    # Preserve qa_pairs for conversation, but clear other intermediate data
    state.update({
        "analysis_content": analysis_content,
        "appendix_content": appendix_content,
        "analysis_filename": analysis_filename,
        "appendix_filename": appendix_filename,
        # --- Data to clean up ---
        'data': [],
        'relevant_contexts': {},
        'analytical_questions': [],
        'suggested_follow_up_queries': [],
        'knowledge_gap': "",
        'rationale': "",
        'iteration_count': 0,
        'search_queries': [], # Clear search queries as they've been processed
    })
    logger.info("Report generation completed. State updated.")
    return state

logger.info("nodes.py loaded with LangGraph node functions.")
#=============================================================================================