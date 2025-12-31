#!/usr/bin/env python3

"""Archived stub for deprecated cross-encoder implementation.

Restore the original module from Git history if required.
"""

raise ImportError("backend.src.deprecated_cross_encoder_hybrid_retriever.cross_encoder archived. Restore from Git history if needed.")

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logging.warning("langchain_core not available. Using fallback Document class.")
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    LANGCHAIN_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("sentence-transformers not available. Cross-encoder functionality disabled.")
    CrossEncoder = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Model configuration with performance characteristics
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
        "avg_latency_ms": 300,
        "description": "Highest quality, use only if performance allows"
    }
}


@dataclass
class CrossEncoderConfig:
    """Configuration for cross-encoder reranking"""
    model_tier: str = "fast"  # fast, balanced, quality
    top_k: int = 50           # Documents to retrieve before reranking
    final_k: int = 20         # Final documents after reranking
    batch_size: int = 32      # Batch size for processing
    enable_caching: bool = True    # Enable model instance caching
    max_text_length: int = 512     # Maximum text length for processing
    score_threshold: float = 0.0   # Minimum score threshold
    timeout_seconds: float = 30.0  # Timeout for reranking operation


class ModelCache:
    """Thread-safe singleton cache for cross-encoder models"""
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get cached model or load if not cached"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
            
        if model_name not in self._models:
            with self._lock:
                if model_name not in self._models:
                    try:
                        logger.info(f"Loading cross-encoder model: {model_name}")
                        start_time = time.time()
                        model = CrossEncoder(model_name)
                        load_time = time.time() - start_time
                        logger.info(f"Model loaded in {load_time:.2f}s: {model_name}")
                        self._models[model_name] = model
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {e}")
                        return None
        
        return self._models.get(model_name)
    
    def clear_cache(self):
        """Clear all cached models"""
        with self._lock:
            self._models.clear()
            logger.info("Model cache cleared")


class OptimizedCrossEncoder:
    """
    Optimized cross-encoder with caching, batching, and performance monitoring
    """
    
    def __init__(self, config: CrossEncoderConfig = None):
        self.config = config or CrossEncoderConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_cache = ModelCache()
        self.model = None
        self.model_name = None
        self._performance_metrics = {
            "total_requests": 0,
            "total_documents": 0,
            "total_time": 0.0,
            "cache_hits": 0
        }
        
        # Initialize model based on config
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("sentence-transformers not available")
            return
        
        model_tier = self.config.model_tier
        if model_tier not in CROSS_ENCODER_MODELS:
            self.logger.warning(f"Unknown model tier '{model_tier}', using 'fast'")
            model_tier = "fast"
        
        self.model_name = CROSS_ENCODER_MODELS[model_tier]["name"]
        
        if self.config.enable_caching:
            self.model = self.model_cache.get_model(self.model_name)
        else:
            try:
                self.model = CrossEncoder(self.model_name)
                self.logger.info(f"Direct model load: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model directly: {e}")
                self.model = None
    
    def is_available(self) -> bool:
        """Check if cross-encoder is available and functional"""
        return self.model is not None and SENTENCE_TRANSFORMERS_AVAILABLE
    
    def _prepare_text_pairs(self, query: str, documents: List[Document]) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for scoring"""
        pairs = []
        max_length = self.config.max_text_length
        
        for doc in documents:
            # Truncate text if too long
            text = doc.page_content
            if len(text) > max_length:
                text = text[:max_length]
            
            pairs.append((query, text))
        
        return pairs
    
    def _process_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Process a batch of query-document pairs"""
        try:
            scores = self.model.predict(pairs)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Return neutral scores as fallback
            return [0.0] * len(pairs)
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using cross-encoder model
        
        Args:
            query: Search query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents with scores in metadata
        """
        if not self.is_available():
            self.logger.warning("Cross-encoder not available, returning original order")
            return documents[:self.config.final_k]
        
        if not documents:
            return documents
        
        start_time = time.time()
        self._performance_metrics["total_requests"] += 1
        self._performance_metrics["total_documents"] += len(documents)
        
        try:
            # Prepare text pairs
            pairs = self._prepare_text_pairs(query, documents)
            
            # Process in batches for better performance
            all_scores = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self._process_batch(batch_pairs)
                all_scores.extend(batch_scores)
            
            # Create scored documents
            scored_docs = []
            for doc, score in zip(documents, all_scores):
                # Add score to metadata
                doc.metadata = doc.metadata.copy() if doc.metadata else {}
                doc.metadata['cross_encoder_score'] = float(score)
                doc.metadata['reranking_method'] = 'cross_encoder'
                
                # Apply score threshold if configured
                if score >= self.config.score_threshold:
                    scored_docs.append((doc, score))
            
            # Sort by score (descending) and return top-k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            reranked_docs = [doc for doc, _ in scored_docs[:self.config.final_k]]
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._performance_metrics["total_time"] += processing_time
            
            self.logger.debug(
                f"Reranked {len(documents)} -> {len(reranked_docs)} documents "
                f"in {processing_time:.3f}s"
            )
            
            return reranked_docs
            
        except Exception as e:
            self.logger.error(f"Cross-encoder reranking failed: {e}")
            # Graceful fallback to original order
            return documents[:self.config.final_k]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self._performance_metrics.copy()
        if metrics["total_requests"] > 0:
            metrics["avg_time_per_request"] = metrics["total_time"] / metrics["total_requests"]
            metrics["avg_docs_per_request"] = metrics["total_documents"] / metrics["total_requests"]
        else:
            metrics["avg_time_per_request"] = 0.0
            metrics["avg_docs_per_request"] = 0.0
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_available():
            return {"available": False, "reason": "Model not loaded"}
        
        model_info = CROSS_ENCODER_MODELS.get(self.config.model_tier, {})
        return {
            "available": True,
            "model_name": self.model_name,
            "model_tier": self.config.model_tier,
            "estimated_size_mb": model_info.get("size_mb", "unknown"),
            "avg_latency_ms": model_info.get("avg_latency_ms", "unknown"),
            "config": {
                "top_k": self.config.top_k,
                "final_k": self.config.final_k,
                "batch_size": self.config.batch_size,
                "caching_enabled": self.config.enable_caching
            }
        }


def create_cross_encoder(
    model_tier: str = "fast",
    top_k: int = 50,
    final_k: int = 20,
    batch_size: int = 32,
    enable_caching: bool = True
) -> OptimizedCrossEncoder:
    """
    Factory function to create an optimized cross-encoder
    
    Args:
        model_tier: "fast", "balanced", or "quality"
        top_k: Documents to retrieve before reranking
        final_k: Final documents after reranking
        batch_size: Batch size for processing
        enable_caching: Enable model instance caching
        
    Returns:
        OptimizedCrossEncoder instance
    """
    config = CrossEncoderConfig(
        model_tier=model_tier,
        top_k=top_k,
        final_k=final_k,
        batch_size=batch_size,
        enable_caching=enable_caching
    )
    
    return OptimizedCrossEncoder(config)


def test_cross_encoder():
    """Test function for the optimized cross-encoder"""
    logger.info("Testing OptimizedCrossEncoder...")
    
    # Create test documents
    test_docs = [
        Document(page_content="Machine learning algorithms for data analysis", metadata={"source": "test1"}),
        Document(page_content="Deep learning neural networks and AI", metadata={"source": "test2"}),
        Document(page_content="Python programming tutorials", metadata={"source": "test3"}),
    ]
    
    # Test cross-encoder
    cross_encoder = create_cross_encoder(model_tier="fast", final_k=3)
    
    print(f"Cross-encoder available: {cross_encoder.is_available()}")
    print(f"Model info: {cross_encoder.get_model_info()}")
    
    if cross_encoder.is_available():
        query = "machine learning algorithms"
        reranked = cross_encoder.rerank_documents(query, test_docs)
        
        print(f"\nReranked {len(test_docs)} -> {len(reranked)} documents:")
        for i, doc in enumerate(reranked, 1):
            score = doc.metadata.get('cross_encoder_score', 'N/A')
            print(f"  {i}. [Score: {score:.3f}] {doc.page_content[:50]}...")
        
        print(f"\nPerformance metrics: {cross_encoder.get_performance_metrics()}")
    
    logger.info("Test completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_cross_encoder()