"""Archived stub for `enhanced_embeddings.py`.

Enhanced embeddings implementation was archived to simplify the scraper-only
deployment. Restore from Git history to re-enable.
"""

raise ImportError("backend.src.enhanced_embeddings has been archived to backend/archived/. Restore from Git history if needed.")

# LangChain base class import
try:
    from langchain_core.embeddings import Embeddings
    LANGCHAIN_EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain Embeddings base class not available: {e}")
    LANGCHAIN_EMBEDDINGS_AVAILABLE = False
    # Fallback base class
    class Embeddings:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            raise NotImplementedError
        def embed_query(self, text: str) -> List[float]:
            raise NotImplementedError

# Google AI imports
try:
    from google import genai
    from google.genai import types
    from google.genai.types import EmbedContentConfig
    GOOGLE_GENAI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Google GenAI not available: {e}")
    GOOGLE_GENAI_AVAILABLE = False

# Task type configurations for different use cases
@dataclass
class EmbeddingTask:
    """Embedding task configuration with optimized settings."""
    task_type: str
    description: str
    use_cases: List[str]
    
    # Common task types from Gemini documentation
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"  # For documents to be retrieved
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"        # For search queries
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY" # For similarity comparison
    CLASSIFICATION = "CLASSIFICATION"           # For text classification
    CLUSTERING = "CLUSTERING"                   # For clustering tasks
    QUESTION_ANSWERING = "QUESTION_ANSWERING"   # For questions in QA systems
    FACT_VERIFICATION = "FACT_VERIFICATION"     # For fact-checking statements
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY" # For code search queries


from .config import GOOGLE_API_KEY, EMBEDDING_MODEL

google_api_key = GOOGLE_API_KEY
embedding_model = EMBEDDING_MODEL

class EnhancedGoogleEmbeddings(Embeddings):
    """
    Enhanced Google Embeddings using direct genai.Client with task type specification.
    Supports the latest gemini-embedding-001 model with optimized configurations.
    Inherits from LangChain Embeddings base class for compatibility.
    """
    
    def __init__(
        self,
        google_api_key: str = google_api_key,
        model: str = embedding_model,
        default_task_type: str = "RETRIEVAL_DOCUMENT",
        output_dimensionality: int = 768,  # Default to 768 for efficiency
        normalize_embeddings: bool = True,
        batch_size: int = 100,
        max_retries: int = 3,
        request_timeout: int = 30
    ):
        """
        Initialize enhanced Google embeddings.
        
        Args:
            google_api_key: Google AI API key
            model: Embedding model name (default: gemini-embedding-001)
            default_task_type: Default task type for embeddings
            output_dimensionality: Embedding vector size (128-3072, recommended: 768, 1536, 3072)
            normalize_embeddings: Whether to normalize embeddings for non-3072 dimensions
            batch_size: Batch size for processing multiple texts
            max_retries: Maximum number of retries for API calls
            request_timeout: Timeout for API requests
        """
        if not google_api_key:
            raise ValueError("Google API key is required")
        
        if not GOOGLE_GENAI_AVAILABLE:
            raise ImportError("google.genai package is required but not available")
        
        self.api_key = google_api_key
        self.model = model
        self.default_task_type = default_task_type
        self.output_dimensionality = output_dimensionality
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        # Validate output dimensionality
        if not (128 <= output_dimensionality <= 3072):
            logger.warning(f"Output dimensionality {output_dimensionality} may not be optimal. "
                          "Recommended: 768, 1536, or 3072")
        
        # Auto-enable normalization for non-3072 dimensions
        if output_dimensionality != 3072 and not normalize_embeddings:
            logger.debug(f"Auto-enabling normalization for {output_dimensionality} dimensions")
            self.normalize_embeddings = True
        
        logger.debug(f"Initialized Enhanced Google Embeddings: {model} "
                    f"(task: {default_task_type}, dim: {output_dimensionality})")
    
    def _create_embed_config(self, task_type: Optional[str] = None) -> EmbedContentConfig:
        """Create embedding configuration with task type."""
        effective_task_type = task_type or self.default_task_type
        
        return EmbedContentConfig(
            task_type=effective_task_type,
            output_dimensionality=self.output_dimensionality
        )
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector."""
        if not self.normalize_embeddings:
            return embedding
        
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        
        if norm == 0:
            logger.warning("Zero norm encountered in embedding normalization")
            return embedding
        
        return (embedding_array / norm).tolist()
    
    def _embed_with_retry(
        self,
        contents: Union[str, List[str]],
        task_type: Optional[str] = None
    ) -> List[List[float]]:
        """Embed content with retry logic."""
        config = self._create_embed_config(task_type)
        
        for attempt in range(self.max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
                
                # Extract embeddings and normalize if needed
                embeddings = []
                for embedding_obj in result.embeddings:
                    embedding = embedding_obj.values
                    if self.normalize_embeddings:
                        embedding = self._normalize_embedding(embedding)
                    embeddings.append(embedding)
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"All embedding attempts failed for task_type={task_type}")
                    raise
                
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.debug(f"Retrying in {wait_time} seconds...")
                import time
                time.sleep(wait_time)
        
        return []
    
    def embed_documents(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """
        Embed documents for retrieval/indexing.
        
        Args:
            texts: List of texts to embed
            task_type: Task type for optimization (default: RETRIEVAL_DOCUMENT)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = self._embed_with_retry(batch_texts, task_type)
                all_embeddings.extend(batch_embeddings)
                
                if len(batch_texts) > 1:
                    logger.debug(f"Embedded batch {i//self.batch_size + 1} "
                               f"({len(batch_texts)} documents) with task_type={task_type}")
                
            except Exception as e:
                logger.error(f"Failed to embed batch {i//self.batch_size + 1}: {e}")
                # Add empty embeddings for failed batch
                empty_embedding = [0.0] * self.output_dimensionality
                all_embeddings.extend([empty_embedding] * len(batch_texts))
        
        return all_embeddings
    
    def embed_query(
        self,
        text: str,
        task_type: str = "RETRIEVAL_QUERY"
    ) -> List[float]:
        """
        Embed a query for search/retrieval.
        
        Args:
            text: Query text to embed
            task_type: Task type for optimization (default: RETRIEVAL_QUERY)
            
        Returns:
            Embedding vector
        """
        if not text:
            return [0.0] * self.output_dimensionality
        
        try:
            embeddings = self._embed_with_retry([text], task_type)
            return embeddings[0] if embeddings else [0.0] * self.output_dimensionality
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return [0.0] * self.output_dimensionality
    
    # LangChain-compatible interface methods (required for FAISS compatibility)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain-compatible method for embedding documents.
        Uses default RETRIEVAL_DOCUMENT task type.
        """
        return self.embed_documents_with_task(texts, self.default_task_type)
    
    def embed_query(self, text: str) -> List[float]:
        """
        LangChain-compatible method for embedding queries.
        Uses RETRIEVAL_QUERY task type for optimal query embedding.
        """
        return self.embed_query_with_task(text, "RETRIEVAL_QUERY")
    
    def embed_documents_with_task(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """
        Embed documents for retrieval/indexing with specified task type.
        
        Args:
            texts: List of texts to embed
            task_type: Task type for optimization (default: RETRIEVAL_DOCUMENT)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = self._embed_with_retry(batch_texts, task_type)
                all_embeddings.extend(batch_embeddings)
                
                if len(batch_texts) > 1:
                    logger.debug(f"Embedded batch {i//self.batch_size + 1} "
                               f"({len(batch_texts)} documents) with task_type={task_type}")
                
            except Exception as e:
                logger.error(f"Failed to embed batch {i//self.batch_size + 1}: {e}")
                # Add empty embeddings for failed batch
                empty_embedding = [0.0] * self.output_dimensionality
                all_embeddings.extend([empty_embedding] * len(batch_texts))
        
        return all_embeddings
    
    def embed_query_with_task(
        self,
        text: str,
        task_type: str = "RETRIEVAL_QUERY"
    ) -> List[float]:
        """
        Embed a query for search/retrieval with specified task type.
        
        Args:
            text: Query text to embed
            task_type: Task type for optimization (default: RETRIEVAL_QUERY)
            
        Returns:
            Embedding vector
        """
        if not text:
            return [0.0] * self.output_dimensionality
        
        try:
            embeddings = self._embed_with_retry([text], task_type)
            return embeddings[0] if embeddings else [0.0] * self.output_dimensionality
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return [0.0] * self.output_dimensionality
    
    # Specialized methods for different use cases
    def embed_for_semantic_similarity(self, texts: List[str]) -> List[List[float]]:
        """Embed texts for semantic similarity comparison."""
        return self.embed_documents_with_task(texts, task_type="SEMANTIC_SIMILARITY")
    
    def embed_for_classification(self, texts: List[str]) -> List[List[float]]:
        """Embed texts for classification tasks."""
        return self.embed_documents_with_task(texts, task_type="CLASSIFICATION")
    
    def embed_for_clustering(self, texts: List[str]) -> List[List[float]]:
        """Embed texts for clustering analysis."""
        return self.embed_documents_with_task(texts, task_type="CLUSTERING")
    
    def embed_question(self, question: str) -> List[float]:
        """Embed a question for QA systems."""
        return self.embed_query_with_task(question, task_type="QUESTION_ANSWERING")
    
    def embed_code_query(self, query: str) -> List[float]:
        """Embed a natural language query for code retrieval."""
        return self.embed_query_with_task(query, task_type="CODE_RETRIEVAL_QUERY")
    
    def embed_fact_verification(self, statement: str) -> List[float]:
        """Embed a statement for fact verification."""
        return self.embed_query_with_task(statement, task_type="FACT_VERIFICATION")
    
    async def aembed_documents(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """Async version of embed_documents."""
        # For now, run sync version in executor
        # TODO: Implement true async when Google AI supports it
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents_with_task, texts, task_type)
    
    async def aembed_query(
        self,
        text: str,
        task_type: str = "RETRIEVAL_QUERY"
    ) -> List[float]:
        """Async version of embed_query."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query_with_task, text, task_type)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding configuration."""
        return {
            "model": self.model,
            "default_task_type": self.default_task_type,
            "output_dimensionality": self.output_dimensionality,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "supported_task_types": [
                "RETRIEVAL_DOCUMENT",
                "RETRIEVAL_QUERY", 
                "SEMANTIC_SIMILARITY",
                "CLASSIFICATION",
                "CLUSTERING",
                "QUESTION_ANSWERING",
                "FACT_VERIFICATION",
                "CODE_RETRIEVAL_QUERY"
            ]
        }


# Factory function for easy creation
def create_enhanced_embeddings(
    google_api_key: str,
    use_case: str = "retrieval",
    **kwargs
) -> EnhancedGoogleEmbeddings:
    """
    Factory function to create embeddings optimized for specific use cases.
    
    Args:
        google_api_key: Google AI API key
        use_case: Use case optimization ('retrieval', 'similarity', 'classification', 'qa')
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnhancedGoogleEmbeddings instance
    """
    # Use case specific defaults
    use_case_configs = {
        "retrieval": {
            "default_task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 768,
            "normalize_embeddings": True
        },
        "similarity": {
            "default_task_type": "SEMANTIC_SIMILARITY", 
            "output_dimensionality": 768,
            "normalize_embeddings": True
        },
        "classification": {
            "default_task_type": "CLASSIFICATION",
            "output_dimensionality": 768,
            "normalize_embeddings": True
        },
        "qa": {
            "default_task_type": "QUESTION_ANSWERING",
            "output_dimensionality": 1536,
            "normalize_embeddings": True
        },
        "code": {
            "default_task_type": "CODE_RETRIEVAL_QUERY",
            "output_dimensionality": 768,
            "normalize_embeddings": True
        }
    }
    
    # Apply use case specific configuration
    config = use_case_configs.get(use_case, use_case_configs["retrieval"])
    config.update(kwargs)  # Override with user-provided kwargs
    
    return EnhancedGoogleEmbeddings(
        google_api_key=google_api_key,
        **config
    )


logger.debug("enhanced_embeddings.py loaded successfully")