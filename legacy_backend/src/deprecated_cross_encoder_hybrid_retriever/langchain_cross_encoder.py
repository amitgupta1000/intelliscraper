#!/usr/bin/env python3

"""Archived stub for deprecated cross-encoder langchain integration.

Restore from Git history to re-enable.
"""

raise ImportError("backend.src.deprecated_cross_encoder_hybrid_retriever.langchain_cross_encoder archived. Restore from Git history if needed.")

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
    
    # ContextualCompressionRetriever and CrossEncoderReranker are not available 
    # in current LangChain versions - we'll implement our own pattern
    ContextualCompressionRetriever = None
    CrossEncoderReranker = None
    
except ImportError:
    logging.warning("LangChain not fully available. Using fallback implementations.")
    LANGCHAIN_AVAILABLE = False
    
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class BaseRetriever:
        pass
    
    class ContextualCompressionRetriever:
        pass
    
    class CrossEncoderReranker:
        pass

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    CrossEncoder = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ServerFriendlyConfig:
    """Configuration for server-friendly cross-encoder behavior"""
    
    # Model selection
    model_tier: str = "balanced"  # fast, balanced, quality
    
    # Performance limits
    max_candidates: int = 40      # Conservative limit for server performance
    min_candidates: int = 8       # Skip reranking below this threshold
    max_content_length: int = 400 # Truncate content for faster processing
    
    # Safety features
    timeout_seconds: float = 3.0  # Conservative timeout
    fallback_on_error: bool = True # Never break retrieval pipeline
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 500
    cache_ttl: int = 1800  # 30 minutes


# Model configurations with server-friendly options
CROSS_ENCODER_MODELS = {
    "fast": {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "size_mb": 90,
        "avg_latency_ms": 50,
        "description": "Fastest option, good for high-volume production"
    },
    "balanced": {
        "name": "cross-encoder/ms-marco-MiniLM-L-12-v2", 
        "size_mb": 110,
        "avg_latency_ms": 100,
        "description": "Best balance of speed and quality"
    },
    "quality": {
        "name": "cross-encoder/ms-marco-electra-base",
        "size_mb": 440,
        "avg_latency_ms": 200,
        "description": "Highest quality, use for low-volume critical queries"
    }
}


class ServerFriendlyCrossEncoderReranker:
    """
    Server-friendly implementation of CrossEncoderReranker with performance safeguards
    """
    
    def __init__(self, config: ServerFriendlyConfig = None):
        self.config = config or ServerFriendlyConfig()
        self._model = None
        self.query_cache = {}
        self.performance_metrics = {
            "total_queries": 0,
            "avg_latency": 0.0,
            "cache_hits": 0,
            "timeout_count": 0,
            "error_count": 0
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate and adjust configuration for server safety"""
        if self.config.model_tier not in CROSS_ENCODER_MODELS:
            self.logger.warning(f"Unknown model tier {self.config.model_tier}, using 'balanced'")
            self.config.model_tier = "balanced"
        
        # Ensure conservative limits
        self.config.max_candidates = min(self.config.max_candidates, 100)
        self.config.timeout_seconds = min(self.config.timeout_seconds, 10.0)
        
        model_info = CROSS_ENCODER_MODELS[self.config.model_tier]
        self.logger.info(f"Cross-encoder config: {model_info['description']} "
                        f"(~{model_info['size_mb']}MB, ~{model_info['avg_latency_ms']}ms)")
    
    @property
    def model(self):
        """Lazy load model with error handling"""
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_name = CROSS_ENCODER_MODELS[self.config.model_tier]["name"]
                start_time = time.time()
                self._model = CrossEncoder(model_name)
                load_time = time.time() - start_time
                self.logger.info(f"Cross-encoder model loaded in {load_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Failed to load cross-encoder model: {e}")
                self._model = None
        return self._model
    
    def is_available(self) -> bool:
        """Check if cross-encoder is available and healthy"""
        return (SENTENCE_TRANSFORMERS_AVAILABLE and 
                self.model is not None and
                self.performance_metrics["error_count"] < 10)
    
    def should_rerank(self, query: str, documents: List[Document]) -> bool:
        """Intelligent decision on whether to use cross-encoder"""
        
        if not self.is_available():
            return False
        
        # Check candidate count thresholds
        if len(documents) < self.config.min_candidates:
            self.logger.debug("Too few candidates for reranking")
            return False
        
        # Skip very simple queries
        if len(query.split()) < 3:
            self.logger.debug("Query too simple for reranking")
            return False
        
        return True
    
    def compress_documents(self, 
                          documents: List[Document], 
                          query: str) -> List[Document]:
        """
        Main compression method compatible with LangChain ContextualCompressionRetriever
        """
        start_time = time.time()
        
        try:
            # Pre-checks
            if not self.should_rerank(query, documents):
                return documents
            
            # Limit candidates for performance
            candidates = documents[:self.config.max_candidates]
            
            # Check cache
            cache_key = self._get_cache_key(query, candidates)
            if self.config.enable_caching and cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.config.cache_ttl:
                    self.performance_metrics["cache_hits"] += 1
                    self.logger.debug("Cache hit for cross-encoder reranking")
                    return cached_result['documents']
            
            # Perform reranking
            reranked_docs = self._perform_reranking(query, candidates)
            
            # Cache result
            if self.config.enable_caching:
                self.query_cache[cache_key] = {
                    'documents': reranked_docs,
                    'timestamp': time.time()
                }
                
                # Limit cache size
                if len(self.query_cache) > self.config.cache_size:
                    oldest_keys = sorted(self.query_cache.keys(), 
                                       key=lambda k: self.query_cache[k]['timestamp'])[:10]
                    for key in oldest_keys:
                        del self.query_cache[key]
            
            # Update metrics
            duration = time.time() - start_time
            self.performance_metrics["total_queries"] += 1
            self.performance_metrics["avg_latency"] = (
                (self.performance_metrics["avg_latency"] * (self.performance_metrics["total_queries"] - 1) + duration) 
                / self.performance_metrics["total_queries"]
            )
            
            self.logger.debug(f"Cross-encoder reranking completed in {duration:.3f}s")
            return reranked_docs
            
        except Exception as e:
            self.performance_metrics["error_count"] += 1
            self.logger.warning(f"Cross-encoder reranking failed: {e}")
            
            if self.config.fallback_on_error:
                return documents  # Return original ranking
            else:
                raise
    
    def _perform_reranking(self, query: str, documents: List[Document]) -> List[Document]:
        """Perform the actual reranking with timeout protection"""
        if not documents:
            return documents
        
        # Prepare content with truncation
        prepared_docs = []
        for doc in documents:
            content = doc.page_content
            if len(content) > self.config.max_content_length:
                # Truncate at sentence boundary if possible
                sentences = content.split('. ')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence + ". ") > self.config.max_content_length:
                        break
                    truncated += sentence + ". "
                content = truncated if truncated else content[:self.config.max_content_length]
            prepared_docs.append((doc, content))
        
        # Create query-document pairs
        pairs = [(query, content) for doc, content in prepared_docs]
        
        # Get scores with timeout protection
        try:
            scores = self._predict_with_timeout(pairs)
        except TimeoutError:
            self.performance_metrics["timeout_count"] += 1
            self.logger.warning(f"Cross-encoder timeout after {self.config.timeout_seconds}s")
            return documents  # Return original order
        
        # Combine documents with scores and sort
        scored_docs = [(doc, float(score)) for (doc, _), score in zip(prepared_docs, scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Add scores to metadata and return
        reranked_docs = []
        for doc, score in scored_docs:
            doc.metadata['cross_encoder_score'] = score
            doc.metadata['reranking_model'] = CROSS_ENCODER_MODELS[self.config.model_tier]["name"]
            reranked_docs.append(doc)
        
        return reranked_docs
    
    def _predict_with_timeout(self, pairs: List[tuple]) -> List[float]:
        """Predict scores with timeout protection (Windows compatible)"""
        import threading
        import time
        
        result = [None]
        error = [None]
        
        def predict_worker():
            try:
                result[0] = self.model.predict(pairs)
            except Exception as e:
                error[0] = e
        
        # Start prediction in a separate thread
        thread = threading.Thread(target=predict_worker)
        thread.daemon = True
        thread.start()
        
        # Wait for completion or timeout
        thread.join(timeout=self.config.timeout_seconds)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError("Cross-encoder prediction timeout")
        
        if error[0]:
            raise error[0]
        
        if result[0] is None:
            raise RuntimeError("Prediction failed without error")
        
        return result[0]
    
    def _get_cache_key(self, query: str, documents: List[Document]) -> str:
        """Generate cache key for query-document combination"""
        import hashlib
        doc_hashes = [hash(doc.page_content[:100]) for doc in documents]
        combined = f"{query}|{len(documents)}|{hash(tuple(doc_hashes))}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return {
            **self.performance_metrics,
            "model_tier": self.config.model_tier,
            "avg_latency_ms": self.performance_metrics["avg_latency"] * 1000,
            "cache_hit_rate": (self.performance_metrics["cache_hits"] / 
                              max(1, self.performance_metrics["total_queries"]))
        }


class LangChainContextualCompressionRetriever(BaseRetriever):
    """
    Custom implementation of contextual compression retriever pattern
    Compatible with current LangChain architecture
    """
    
    def __init__(self, 
                 base_retriever: BaseRetriever,
                 compressor: ServerFriendlyCrossEncoderReranker):
        """
        Initialize contextual compression retriever
        
        Args:
            base_retriever: The base retriever (e.g., EnsembleRetriever, BM25, etc.)
            compressor: The document compressor/reranker
        """
        super().__init__()
        self.base_retriever = base_retriever
        self.compressor = compressor
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_relevant_documents(self, 
                               query: str,
                               run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        """
        Get relevant documents using base retriever and then compress/rerank them
        """
        try:
            # Step 1: Get documents from base retriever
            if hasattr(self.base_retriever, 'invoke'):
                documents = self.base_retriever.invoke(query)
            elif hasattr(self.base_retriever, '_get_relevant_documents'):
                documents = self.base_retriever._get_relevant_documents(query, run_manager)
            else:
                # Fallback for different retriever interfaces
                documents = self.base_retriever.get_relevant_documents(query)
            
            self.logger.debug(f"Base retriever returned {len(documents)} documents")
            
            # Step 2: Apply compression/reranking
            if documents and self.compressor:
                compressed_docs = self.compressor.compress_documents(documents, query)
                self.logger.debug(f"Compressor returned {len(compressed_docs)} documents")
                return compressed_docs
            else:
                return documents
                
        except Exception as e:
            self.logger.error(f"Error in contextual compression retrieval: {e}")
            # Fallback to base retriever only
            try:
                if hasattr(self.base_retriever, 'invoke'):
                    return self.base_retriever.invoke(query)
                else:
                    return self.base_retriever.get_relevant_documents(query)
            except Exception as fallback_e:
                self.logger.error(f"Fallback retrieval also failed: {fallback_e}")
                return []
    
    def invoke(self, query: str, config=None, **kwargs) -> List[Document]:
        """
        Invoke method for compatibility with newer LangChain interfaces
        """
        return self._get_relevant_documents(query)
    
    @property
    def retriever_type(self) -> str:
        """Return the retriever type"""
        return "contextual_compression_with_cross_encoder"


def create_langchain_cross_encoder_retriever(base_retriever: BaseRetriever,
                                           model_tier: str = "balanced") -> Optional[LangChainContextualCompressionRetriever]:
    """
    Create a LangChain-compatible ContextualCompressionRetriever with server-friendly cross-encoder
    
    Args:
        base_retriever: The base retriever (e.g., EnsembleRetriever)
        model_tier: "fast", "balanced", or "quality"
    
    Returns:
        LangChainContextualCompressionRetriever with cross-encoder or None if unavailable
    """
    
    if not LANGCHAIN_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("LangChain or sentence-transformers not available for cross-encoder")
        return None
    
    try:
        # Create server-friendly configuration
        config = ServerFriendlyConfig(model_tier=model_tier)
        
        # Create our custom compressor
        compressor = ServerFriendlyCrossEncoderReranker(config)
        
        if not compressor.is_available():
            logger.warning("Cross-encoder compressor not available")
            return None
        
        # Create our custom ContextualCompressionRetriever
        compression_retriever = LangChainContextualCompressionRetriever(
            base_retriever=base_retriever,
            compressor=compressor
        )
        
        logger.info(f"LangChainContextualCompressionRetriever created with {model_tier} cross-encoder")
        return compression_retriever
        
    except Exception as e:
        logger.error(f"Failed to create cross-encoder retriever: {e}")
        return None


def create_standard_langchain_cross_encoder(model_tier: str = "balanced") -> Optional[ServerFriendlyCrossEncoderReranker]:
    """
    Create a cross-encoder reranker compatible with LangChain patterns
    """
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("sentence-transformers not available")
        return None
    
    try:
        config = ServerFriendlyConfig(model_tier=model_tier)
        reranker = ServerFriendlyCrossEncoderReranker(config)
        
        if not reranker.is_available():
            logger.warning("Cross-encoder reranker not available")
            return None
            
        return reranker
        
    except Exception as e:
        logger.error(f"Failed to create CrossEncoderReranker: {e}")
        return None


# Test function
def test_langchain_cross_encoder():
    """Test the LangChain-compatible implementation"""
    
    # Mock retriever for testing
    class MockRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, run_manager=None) -> List[Document]:
            return [
                Document("Python machine learning with scikit-learn", {"source": "doc1"}),
                Document("JavaScript web development frameworks", {"source": "doc2"}),
                Document("Deep learning neural networks TensorFlow", {"source": "doc3"}),
                Document("Machine learning algorithms data science", {"source": "doc4"})
            ]
        
        def invoke(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query)
    
    base_retriever = MockRetriever()
    query = "machine learning algorithms"
    
    # Test our implementation
    compression_retriever = create_langchain_cross_encoder_retriever(
        base_retriever, model_tier="fast"
    )
    
    if compression_retriever:
        try:
            results = compression_retriever.invoke(query)
            print(f"LangChain cross-encoder test successful: {len(results)} documents")
            
            # Show scores
            for i, doc in enumerate(results, 1):
                score = doc.metadata.get('cross_encoder_score', 'N/A')
                print(f"  {i}. [Score: {score}] {doc.page_content[:50]}...")
            
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            return False
    else:
        print("Cross-encoder not available")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_langchain_cross_encoder()
    print(f"Test {'PASSED' if success else 'FAILED'}")