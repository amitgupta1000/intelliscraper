"""Archived stub for `hybrid_retriever.py`.

Hybrid retrieval implementation was archived to reduce dependencies. Restore
from Git history if hybrid retrieval is required again.
"""

raise ImportError("backend.src.hybrid_retriever has been archived to backend/archived/. Restore from Git history if needed.")

# Type definitions and fallbacks
try:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except ImportError as e:
    logger.warning(f"LangChain imports failed: {e}. Using fallback types.")
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    class RecursiveCharacterTextSplitter:
        def __init__(self, **kwargs): pass
        def split_text(self, text): return [text]
    FAISS, BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever, CrossEncoderReranker = None, None, None, None, None

# Cross-encoder imports for semantic reranking
try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    logger.warning("HuggingFaceCrossEncoder not available. Reranking will be disabled.")
    HuggingFaceCrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False

# Use config.py for all settings
try:
    from backend.src.config import (
        RETRIEVAL_TOP_K,
        HYBRID_VECTOR_WEIGHT,
        HYBRID_BM25_WEIGHT,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        MIN_CHUNK_LENGTH,
        MIN_WORD_COUNT,
        USE_CROSS_ENCODER_RERANKING,
        CROSS_ENCODER_MODEL,
        RERANK_TOP_K
    )
except ImportError:
    logger.warning("Could not import from config. Using default values for retriever.")
    RETRIEVAL_TOP_K = 20
    HYBRID_VECTOR_WEIGHT = 0.6
    HYBRID_BM25_WEIGHT = 0.4
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MIN_CHUNK_LENGTH = 50
    MIN_WORD_COUNT = 10
    USE_CROSS_ENCODER_RERANKING = False
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    RERANK_TOP_K = 20

class HybridRetriever:
    """
    A simplified hybrid retriever using standard LangChain components.
    """
    def __init__(self, embeddings=None):
        self.embeddings = embeddings
        # Use all config settings from config.py for full alignment
        self.top_k = RETRIEVAL_TOP_K
        self.vector_weight = HYBRID_VECTOR_WEIGHT
        self.bm25_weight = HYBRID_BM25_WEIGHT
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.min_chunk_length = MIN_CHUNK_LENGTH
        self.min_word_count = MIN_WORD_COUNT
        self.use_cross_encoder = USE_CROSS_ENCODER_RERANKING
        self.cross_encoder_model = CROSS_ENCODER_MODEL
        self.rerank_top_k = RERANK_TOP_K
        self.vector_store = None
        self.bm25_retriever = None
        self.final_retriever = None
        self.documents = []
        from .logging_setup import logger
        self.logger = logger

    def build_index(self, relevant_contexts: Dict[str, Dict[str, str]]) -> bool:
        """
        Builds the retrieval pipeline from documents.
        """
        try:
            documents = self._process_documents(relevant_contexts)
            if not documents:
                self.logger.warning("No documents to index.")
                return False
            self.documents = documents

            # 1. Build base retrievers
            if not self._build_vector_index(documents) or not self._build_bm25_index(documents):
                self.logger.error("Failed to build one or more base retrievers.")
                return False

            # 2. Create Ensemble Retriever
            vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[self.bm25_weight, self.vector_weight],
            )
            self.logger.debug("LangChain EnsembleRetriever created successfully.")

            # 3. Optionally wrap with Reranker
            if self.use_cross_encoder and CROSS_ENCODER_AVAILABLE:
                self.logger.debug(f"Initializing reranker with model: {self.cross_encoder_model}")
                model = HuggingFaceCrossEncoder(model_name=self.cross_encoder_model)
                compressor = CrossEncoderReranker(model=model, top_n=self.rerank_top_k)
                self.final_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=ensemble_retriever,
                )
                self.logger.debug(f"Wrapped ensemble retriever with ContextualCompressionRetriever. Will return top {self.rerank_top_k} docs.")
            else:
                self.final_retriever = ensemble_retriever

            return True

        except Exception as e:
            self.logger.error(f"Error building hybrid index: {e}", exc_info=True)
            return False

    def _process_documents(self, relevant_contexts: Dict[str, Dict[str, str]]) -> List[Document]:
        """Process and chunk documents from relevant contexts."""
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        for url, content_data in relevant_contexts.items():
            content = content_data.get('content', '')
            if len(content.strip()) < 100:
                continue

            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                if (len(chunk) < self.min_chunk_length or len(chunk.split()) < self.min_word_count):
                    continue

                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "title": content_data.get('title', 'Untitled'),
                        "chunk_index": i,
                    }
                ))

        self.logger.debug(f"Processed {len(documents)} document chunks.")
        return documents

    def _build_vector_index(self, documents: List[Document]) -> bool:
        """Build FAISS vector index."""
        try:
            if not self.embeddings or not FAISS:
                self.logger.warning("Embeddings or FAISS not available.")
                return False
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.logger.debug("Vector index built successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error building vector index: {e}", exc_info=True)
            return False

    def _build_bm25_index(self, documents: List[Document]) -> bool:
        """Build BM25 index."""
        try:
            if not BM25Retriever:
                self.logger.warning("BM25Retriever not available.")
                return False
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = self.top_k
            self.logger.debug("BM25 index built successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error building BM25 index: {e}", exc_info=True)
            return False

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents using the configured pipeline.
        """
        if not self.final_retriever:
            self.logger.warning("Retriever not built. Returning empty list.")
            return []
        try:
            return self.final_retriever.invoke(query)
        except Exception as e:
            self.logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []

    def retrieve_with_query_responses(self, queries: List[str]) -> Tuple[List[Document], Dict[str, str]]:
        """
        Retrieve documents using multiple queries and generate simple responses.
        This version is simplified as batching is handled by underlying components.
        """
        if not self.final_retriever:
            return [], {}

        query_responses = {}
        seen_docs = set()

        if not queries:
            return [], {}

        # 1. Retrieve documents for each query to get a comprehensive set
        all_retrieved_docs = []
        for query in queries:
            if query.strip():
                docs = self.retrieve(query)
                for doc in docs:
                    doc_key = self._get_doc_key(doc)
                    if doc_key not in seen_docs:
                        all_retrieved_docs.append(doc)
                        seen_docs.add(doc_key)

        self.logger.info(f"Retrieved {len(all_retrieved_docs)} unique documents from {len(queries)} queries.")

        # Generate simple responses for all queries based on the retrieved docs
        for query in queries:
            if not query.strip():
                continue

            if all_retrieved_docs:
                top_docs_preview = all_retrieved_docs[:3]
                response_parts = []

                for doc in top_docs_preview:
                    title = doc.metadata.get('title', 'Untitled')
                    content_preview = doc.page_content[:150]
                    response_parts.append(f"From {title}: {content_preview}...")

                query_responses[query] = "\n\n".join(response_parts)
            else:
                query_responses[query] = "No relevant information found for this query."
        return all_retrieved_docs, query_responses

    def _get_doc_key(self, doc: Document) -> str:
        """Generate unique key for document deduplication."""
        source = doc.metadata.get('source', '')
        chunk_idx = doc.metadata.get('chunk_index', 0)
        return f"{source}_{chunk_idx}"

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_documents": len(self.documents),
            "retriever_type": self.final_retriever.__class__.__name__ if self.final_retriever else "None",
            "config": {
                "top_k": self.top_k,
                "vector_weight": self.vector_weight,
                "bm25_weight": self.bm25_weight,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "min_chunk_length": self.min_chunk_length,
                "min_word_count": self.min_word_count,
                "use_cross_encoder": self.use_cross_encoder,
                "cross_encoder_model": self.cross_encoder_model,
                "rerank_top_k": self.rerank_top_k,
            }
        }


# Factory function for easy integration
def create_hybrid_retriever(embeddings=None) -> HybridRetriever:
    """
    Factory function to create a new hybrid retriever using config.py settings.
    """
    return HybridRetriever(embeddings=embeddings)


logger.debug("hybrid_retriever.py loaded successfully")