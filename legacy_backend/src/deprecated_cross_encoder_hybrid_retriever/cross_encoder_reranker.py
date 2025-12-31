#!/usr/bin/env python3
"""
Cross-Encoder Reranker Implementation for INTELLISEARCH
Using LangChain's native CrossEncoderReranker for seamless integration
"""
"""Archived stub for deprecated cross-encoder reranker.

Restore full implementation from Git history if needed.
"""

raise ImportError("backend.src.deprecated_cross_encoder_hybrid_retriever.cross_encoder_reranker archived. Restore from Git history if needed.")
    logging.warning("langchain_core not available. Using fallback Document class.")
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    LANGCHAIN_CORE_AVAILABLE = True

# LangChain imports with fallbacks for different versions
LANGCHAIN_COMPRESSORS_AVAILABLE = False
CrossEncoderReranker = None
BaseCrossEncoder = None

# Try multiple import paths for LangChain cross-encoder
import_paths = [
    ("langchain.retrievers.document_compressors", "langchain.retrievers.document_compressors.cross_encoder"),
    ("langchain_community.document_compressors", "langchain_community.document_compressors.cross_encoder"),
    ("langchain.document_compressors", "langchain.document_compressors.cross_encoder"),
]

for compressor_module, cross_encoder_module in import_paths:
    try:
        exec(f"from {compressor_module} import CrossEncoderReranker")
        exec(f"from {cross_encoder_module} import BaseCrossEncoder")
        LANGCHAIN_COMPRESSORS_AVAILABLE = True
        logging.info(f"LangChain cross-encoder imports successful from {compressor_module}")
        break
    except ImportError:
        continue

if not LANGCHAIN_COMPRESSORS_AVAILABLE:
    logging.warning("LangChain document compressors not available. Using custom implementation.")

# Sentence transformers for the actual cross-encoder model
try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    STCrossEncoder = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SentenceTransformersCrossEncoder(BaseCrossEncoder):
    """
    LangChain-compatible wrapper for sentence-transformers cross-encoder models.
    Implements the BaseCrossEncoder interface for seamless integration.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the cross-encoder with a sentence-transformers model.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
                       Popular choices:
                       - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast)
                       - "cross-encoder/ms-marco-MiniLM-L-12-v2" (balanced) 
                       - "cross-encoder/ms-marco-electra-base" (best quality)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self._model = None
        logger.info(f"Initializing cross-encoder: {model_name}")
    
    @property 
    def model(self) -> Any:
        """Lazy load the model to avoid loading during import"""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = STCrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        return self._model
    
    def score(self, query: str, documents: List[Document]) -> List[float]:
        """
        Score documents against query using cross-encoder.
        
        Args:
            query: The search query
            documents: List of documents to score
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not documents:
            return []
        
        # Create query-document pairs for cross-encoder input
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Convert to Python floats and return
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)


def create_cross_encoder_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_n: int = 20
) -> Optional[Any]:
    """
    Create a LangChain-compatible cross-encoder reranker.
    
    Args:
        model_name: HuggingFace model name for cross-encoder
        top_n: Number of top documents to return after reranking
        
    Returns:
        CrossEncoderReranker instance or None if dependencies unavailable
    """
    if not LANGCHAIN_COMPRESSORS_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("Cross-encoder dependencies not available. Skipping reranker creation.")
        return None
    
    try:
        # Create the sentence-transformers wrapper
        cross_encoder = SentenceTransformersCrossEncoder(model_name)
        
        # Create LangChain's CrossEncoderReranker
        reranker = CrossEncoderReranker(
            model=cross_encoder,
            top_n=top_n
        )
        
        logger.info(f"Cross-encoder reranker created: {model_name} (top_n={top_n})")
        return reranker
        
    except Exception as e:
        logger.error(f"Failed to create cross-encoder reranker: {e}")
        return None


def test_cross_encoder_reranker():
    """Test the cross-encoder reranker implementation"""
    logger.info("Testing cross-encoder reranker...")
    
    # Create test documents
    test_docs = [
        Document(
            page_content="Python is a programming language used for web development and data science.",
            metadata={"source": "doc1"}
        ),
        Document(
            page_content="Machine learning algorithms can analyze large datasets to find patterns.",
            metadata={"source": "doc2"}
        ),
        Document(
            page_content="JavaScript is primarily used for frontend web development and browser scripting.",
            metadata={"source": "doc3"}
        ),
        Document(
            page_content="Deep learning neural networks are a subset of machine learning techniques.",
            metadata={"source": "doc4"}
        )
    ]
    
    # Test query
    query = "machine learning and data science"
    
    # Create reranker
    reranker = create_cross_encoder_reranker(top_n=3)
    
    if reranker is None:
        logger.error("Failed to create reranker for testing")
        return False
    
    try:
        # Test reranking
        reranked_docs = reranker.compress_documents(test_docs, query)
        
        logger.info(f"Original documents: {len(test_docs)}")
        logger.info(f"Reranked documents: {len(reranked_docs)}")
        
        for i, doc in enumerate(reranked_docs):
            logger.info(f"Rank {i+1}: {doc.metadata.get('source')} - {doc.page_content[:50]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run test
    success = test_cross_encoder_reranker()
    print(f"Test {'PASSED' if success else 'FAILED'}")