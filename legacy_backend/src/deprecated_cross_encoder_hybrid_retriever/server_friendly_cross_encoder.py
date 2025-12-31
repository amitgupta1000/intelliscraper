#!/usr/bin/env python3

"""Archived stub for deprecated cross-encoder helpers.

Restore the full implementation from Git history if required.
"""

raise ImportError("backend.src.deprecated_cross_encoder_hybrid_retriever.server_friendly_cross_encoder archived. Restore from Git history if needed.")
#!/usr/bin/env python3
"""
Server-Friendly Cross-Encoder Implementation for INTELLISEARCH
Optimized for performance, memory usage, and graceful degradation
"""

import logging
import time
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache
import threading

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    CrossEncoder = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Model configuration with server-friendly options
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


@dataclass
class CrossEncoderConfig:
    """Configuration for server-friendly cross-encoder reranking"""
    
    # Model selection
    model_tier: str = "balanced"  # fast, balanced, quality
    
    # Performance limits
    max_candidates: int = 40      # Conservative limit for server performance
    min_candidates: int = 8       # Skip reranking below this threshold
    max_content_length: int = 400 # Truncate content for faster processing
    batch_size: int = 16          # Smaller batches for memory efficiency
    
    # Adaptive behavior
    use_adaptive_candidates: bool = True  # Adjust based on query complexity
    use_content_truncation: bool = True   # Truncate long documents
    use_query_filtering: bool = True      # Skip simple/specific queries
    
    # Performance safeguards
    timeout_seconds: float = 3.0          # Conservative timeout
    max_memory_mb: int = 500              # Memory limit before fallback
    
    # Caching for performance
    enable_caching: bool = True
    cache_size: int = 500                 # Conservative cache size
    cache_ttl: int = 1800                 # 30 minutes
    
    # Fallback behavior
    fallback_on_timeout: bool = True      # Always fallback gracefully
    fallback_on_error: bool = True        # Never break the retrieval pipeline
    fallback_on_memory: bool = True       # Fallback if memory usage too high


class PerformanceMetrics:
    """Track cross-encoder performance for optimization"""
    
    def __init__(self):
        self.rerank_times = []
        self.candidate_counts = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.timeout_count = 0
        self.error_count = 0
        self.total_queries = 0
        self.lock = threading.Lock()
    
    def log_reranking(self, duration: float, candidates: int, cache_hit: bool = False):
        """Log performance metrics"""
        with self.lock:
            self.rerank_times.append(duration)
            self.candidate_counts.append(candidates)
            self.total_queries += 1
            
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            # Keep only recent metrics
            if len(self.rerank_times) > 100:
                self.rerank_times = self.rerank_times[-50:]
                self.candidate_counts = self.candidate_counts[-50:]
    
    def get_avg_latency(self) -> float:
        """Get average reranking latency"""
        with self.lock:
            if not self.rerank_times:
                return 0.0
            return sum(self.rerank_times) / len(self.rerank_times)
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        with self.lock:
            total = self.cache_hits + self.cache_misses
            return self.cache_hits / total if total > 0 else 0.0
    
    def should_adjust_config(self) -> Dict[str, Any]:
        """Suggest configuration adjustments"""
        suggestions = {}
        avg_latency = self.get_avg_latency()
        
        if avg_latency > 2.0:
            suggestions["model_tier"] = "fast"
            suggestions["reason"] = "High latency detected"
        elif avg_latency > 1.0:
            suggestions["max_candidates"] = max(20, self.candidate_counts[-10:] if self.candidate_counts else 30)
            suggestions["reason"] = "Moderate latency, reduce candidates"
        
        if self.timeout_count > 5:
            suggestions["timeout_seconds"] = min(5.0, self.timeout_seconds * 1.5)
            suggestions["reason"] = "Frequent timeouts"
        
        return suggestions


class ServerFriendlyCrossEncoder:
    """
    Production-ready cross-encoder that prioritizes server stability
    """
    
    def __init__(self, config: CrossEncoderConfig = None):
        self.config = config or CrossEncoderConfig()
        self._model = None
        self.metrics = PerformanceMetrics()
        self.query_cache = {}
        self.last_memory_check = time.time()
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
        self.config.batch_size = min(self.config.batch_size, 64)
        
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
                self.metrics.error_count < 10)
    
    def should_use_reranking(self, query: str, candidates: List[Document]) -> bool:
        """Intelligent decision on whether to use cross-encoder"""
        
        if not self.is_available():
            return False
        
        # Check candidate count thresholds
        if len(candidates) < self.config.min_candidates:
            self.logger.debug("Too few candidates for reranking")
            return False
        
        if len(candidates) > self.config.max_candidates:
            self.logger.debug("Too many candidates, will truncate")
        
        # Skip very simple queries if enabled
        if self.config.use_query_filtering:
            if len(query.split()) < 3:
                self.logger.debug("Query too simple for reranking")
                return False
            
            # Skip very specific queries (likely handled well by BM25)
            specific_indicators = ['exact', 'specific', 'precise', 'definition of']
            if any(indicator in query.lower() for indicator in specific_indicators):
                self.logger.debug("Query appears specific, skipping reranking")
                return False
        
        return True
    
    def get_optimal_candidate_count(self, query: str, total_candidates: int) -> int:
        """Dynamically adjust candidate count based on query and performance"""
        if not self.config.use_adaptive_candidates:
            return min(self.config.max_candidates, total_candidates)
        
        base_count = self.config.max_candidates
        
        # Adjust based on query complexity
        query_words = len(query.split())
        if query_words <= 4:
            # Simple queries: fewer candidates
            base_count = int(base_count * 0.7)
        elif query_words > 8:
            # Complex queries: more candidates (up to limit)
            base_count = int(base_count * 1.2)
        
        # Adjust based on recent performance
        avg_latency = self.metrics.get_avg_latency()
        if avg_latency > 1.5:
            # If slow, reduce candidates
            base_count = int(base_count * 0.8)
        
        return min(base_count, total_candidates, self.config.max_candidates)
    
    def prepare_content(self, doc: Document) -> str:
        """Prepare document content for efficient processing"""
        content = doc.page_content
        
        if not self.config.use_content_truncation:
            return content
        
        max_length = self.config.max_content_length
        if len(content) <= max_length:
            return content
        
        # Try to truncate at sentence boundary
        sentences = content.split('. ')
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence + ". ") > max_length:
                break
            truncated += sentence + ". "
        
        return truncated if truncated else content[:max_length]
    
    def get_cache_key(self, query: str, doc_hashes: List[str]) -> str:
        """Generate cache key for query-document combination"""
        combined = f"{query}|{len(doc_hashes)}|{hash(tuple(sorted(doc_hashes)))}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_doc_hash(self, doc: Document) -> str:
        """Quick hash for document content"""
        content = self.prepare_content(doc)[:100]  # Use first 100 chars
        return hashlib.md5(content.encode()).hexdigest()
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents with comprehensive error handling and optimization
        """
        start_time = time.time()
        
        # Pre-checks
        if not self.should_use_reranking(query, documents):
            return documents
        
        # Optimize candidate count
        candidate_count = self.get_optimal_candidate_count(query, len(documents))
        candidates = documents[:candidate_count]
        
        try:
            # Check cache first
            doc_hashes = [self.get_doc_hash(doc) for doc in candidates]
            cache_key = self.get_cache_key(query, doc_hashes)
            
            if self.config.enable_caching and cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                # Check if cache is still valid
                if time.time() - cached_result['timestamp'] < self.config.cache_ttl:
                    self.metrics.log_reranking(time.time() - start_time, len(candidates), cache_hit=True)
                    self.logger.debug("Cache hit for cross-encoder reranking")
                    return cached_result['documents']
            
            # Perform reranking with timeout
            reranked_docs = self._do_reranking_with_timeout(query, candidates)
            
            # Cache result
            if self.config.enable_caching:
                self.query_cache[cache_key] = {
                    'documents': reranked_docs,
                    'timestamp': time.time()
                }
                
                # Limit cache size
                if len(self.query_cache) > self.config.cache_size:
                    # Remove oldest entries
                    oldest_keys = sorted(self.query_cache.keys(), 
                                       key=lambda k: self.query_cache[k]['timestamp'])[:10]
                    for key in oldest_keys:
                        del self.query_cache[key]
            
            duration = time.time() - start_time
            self.metrics.log_reranking(duration, len(candidates))
            self.logger.debug(f"Cross-encoder reranking completed in {duration:.3f}s")
            
            return reranked_docs
            
        except Exception as e:
            self.metrics.error_count += 1
            self.logger.warning(f"Cross-encoder reranking failed: {e}")
            
            if self.config.fallback_on_error:
                return documents  # Return original ranking
            else:
                raise
    
    def _do_reranking_with_timeout(self, query: str, documents: List[Document]) -> List[Document]:
        """Perform reranking with timeout protection"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Cross-encoder reranking timeout")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.config.timeout_seconds))
        
        try:
            result = self._do_reranking(query, documents)
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            self.metrics.timeout_count += 1
            self.logger.warning(f"Cross-encoder timeout after {self.config.timeout_seconds}s")
            raise
    
    def _do_reranking(self, query: str, documents: List[Document]) -> List[Document]:
        """Actual reranking implementation"""
        if not documents:
            return documents
        
        # Prepare content
        prepared_docs = [(doc, self.prepare_content(doc)) for doc in documents]
        
        # Create query-document pairs
        pairs = [(query, content) for doc, content in prepared_docs]
        
        # Get scores in batches
        all_scores = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = self.model.predict(batch_pairs)
            all_scores.extend(batch_scores)
        
        # Combine documents with scores and sort
        scored_docs = [(doc, float(score)) for (doc, _), score in zip(prepared_docs, all_scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Add scores to metadata and return
        reranked_docs = []
        for doc, score in scored_docs:
            doc.metadata['cross_encoder_score'] = score
            doc.metadata['reranking_model'] = CROSS_ENCODER_MODELS[self.config.model_tier]["name"]
            reranked_docs.append(doc)
        
        return reranked_docs
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return {
            "total_queries": self.metrics.total_queries,
            "avg_latency_ms": self.metrics.get_avg_latency() * 1000,
            "cache_hit_rate": self.metrics.get_cache_hit_rate(),
            "timeout_count": self.metrics.timeout_count,
            "error_count": self.metrics.error_count,
            "model_tier": self.config.model_tier,
            "suggestions": self.metrics.should_adjust_config()
        }


def create_server_friendly_cross_encoder(model_tier: str = "balanced") -> Optional[ServerFriendlyCrossEncoder]:
    """
    Factory function to create a server-friendly cross-encoder
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("sentence-transformers not available for cross-encoder")
        return None
    
    try:
        config = CrossEncoderConfig(model_tier=model_tier)
        reranker = ServerFriendlyCrossEncoder(config)
        
        if not reranker.is_available():
            logger.warning("Cross-encoder not available")
            return None
        
        logger.info(f"Server-friendly cross-encoder created: {model_tier}")
        return reranker
        
    except Exception as e:
        logger.error(f"Failed to create cross-encoder: {e}")
        return None


# Test function
def test_server_friendly_cross_encoder():
    """Test the server-friendly implementation"""
    # Create test documents
    docs = [
        Document("Python machine learning libraries like scikit-learn", {"source": "doc1"}),
        Document("JavaScript frameworks for web development", {"source": "doc2"}), 
        Document("Deep learning with neural networks and TensorFlow", {"source": "doc3"}),
        Document("Machine learning algorithms and data science", {"source": "doc4"})
    ]
    
    query = "machine learning and AI"
    
    reranker = create_server_friendly_cross_encoder("fast")
    if reranker:
        result = reranker.rerank(query, docs)
        print(f"Reranked {len(docs)} -> {len(result)} documents")
        print("Performance:", reranker.get_performance_summary())
        return True
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_server_friendly_cross_encoder()
    print(f"Test {'PASSED' if success else 'FAILED'}")