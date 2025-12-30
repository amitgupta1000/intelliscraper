#!/usr/bin/env python3
"""
Hybrid Retriever Integration with Cross-Encoder Reranking
Shows how to integrate the contextual compression pattern into your existing code
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logging.warning("LangChain not available")
    LANGCHAIN_AVAILABLE = False
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    class BaseRetriever:
        pass

# Import our cross-encoder implementation
try:
    from .langchain_cross_encoder import (
        create_langchain_cross_encoder_retriever,
        ServerFriendlyConfig
    )
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    logging.warning("Cross-encoder module not available")
    CROSS_ENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridRetrieverWithCrossEncoder:
    """
    Enhanced hybrid retriever following the LangChain ContextualCompressionRetriever pattern
    
    Pattern: Base Retriever (Ensemble) -> Cross-Encoder Reranking -> Final Results
    
    This follows your existing pattern:
    ```
    if reranker:
        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=ensemble_retriever
        )
    else:
        self.retriever = ensemble_retriever
    ```
    """
    
    def __init__(self, 
                 bm25_retriever: BaseRetriever,
                 semantic_retriever: BaseRetriever,
                 cross_encoder_tier: str = "balanced",
                 enable_cross_encoder: bool = True):
        """
        Initialize hybrid retriever with optional cross-encoder reranking
        
        Args:
            bm25_retriever: BM25/keyword-based retriever
            semantic_retriever: Dense vector retriever
            cross_encoder_tier: "fast", "balanced", or "quality"
            enable_cross_encoder: Whether to enable cross-encoder reranking
        """
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.cross_encoder_tier = cross_encoder_tier
        self.enable_cross_encoder = enable_cross_encoder
        
        # Build retriever hierarchy following your pattern
        self.retriever = self._build_retriever()
        
        logger.info(f"Hybrid retriever initialized with cross-encoder: {self.retriever is not None}")
    
    def _build_retriever(self) -> Optional[BaseRetriever]:
        """
        Build the retriever following the LangChain pattern:
        Base Ensemble -> Cross-Encoder Compression -> Final Retriever
        """
        try:
            # --- Build Ensemble Retriever (Base) ---
            logging.info("Building Ensemble Retriever...")
            ensemble_retriever = self._create_ensemble_retriever()
            
            if not ensemble_retriever:
                logging.warning("Failed to create ensemble retriever")
                return None
            
            # --- Contextual Compression Retriever (using Ensemble and Reranker if available) ---
            logging.info("Building final Retriever...")
            
            if self.enable_cross_encoder and CROSS_ENCODER_AVAILABLE:
                # Try to create reranker
                reranker = self._create_cross_encoder_reranker()
                
                if reranker:
                    # Create ContextualCompressionRetriever pattern
                    compression_retriever = create_langchain_cross_encoder_retriever(
                        base_retriever=ensemble_retriever,
                        model_tier=self.cross_encoder_tier
                    )
                    
                    if compression_retriever:
                        logging.info("Contextual Compression Retriever initialized with Reranker.")
                        return compression_retriever
                    else:
                        logging.warning("Failed to create compression retriever, using ensemble only")
                else:
                    logging.warning("Reranker creation failed")
            
            # Fallback: Use the Ensemble retriever directly if reranker fails
            logging.warning("Final Retriever initialized without Reranker (due to failure).")
            return ensemble_retriever
            
        except Exception as e:
            logging.error(f"Error building retriever: {e}")
            return None
    
    def _create_ensemble_retriever(self) -> Optional[BaseRetriever]:
        """Create ensemble retriever combining BM25 and semantic retrievers"""
        try:
            # Import the custom EnsembleRetriever from your existing code
            from .hybrid_retriever import EnsembleRetriever
            
            ensemble = EnsembleRetriever(
                retrievers=[
                    self.semantic_retriever,
                    self.bm25_retriever
                ],
                weights=[0.6, 0.4]  # Favor semantic slightly
            )
            
            logging.info("Ensemble retriever created successfully")
            return ensemble
            
        except Exception as e:
            logging.error(f"Failed to create ensemble retriever: {e}")
            return None
    
    def _create_cross_encoder_reranker(self):
        """Create cross-encoder reranker with server-friendly settings"""
        try:
            from .langchain_cross_encoder import create_standard_langchain_cross_encoder
            
            reranker = create_standard_langchain_cross_encoder(
                model_tier=self.cross_encoder_tier
            )
            
            if reranker and reranker.is_available():
                logging.info(f"Cross-encoder reranker created: {self.cross_encoder_tier}")
                return reranker
            else:
                logging.warning("Cross-encoder reranker not available")
                return None
                
        except Exception as e:
            logging.error(f"Failed to create cross-encoder reranker: {e}")
            return None
    
    def retrieve(self, query: str, k: int = 20) -> List[Document]:
        """
        Retrieve documents using the configured retriever pipeline
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents, reranked if cross-encoder is enabled
        """
        if not self.retriever:
            logging.error("No retriever available")
            return []
        
        try:
            # Use the configured retriever (with or without cross-encoder)
            if hasattr(self.retriever, 'invoke'):
                results = self.retriever.invoke(query)
            else:
                results = self.retriever.get_relevant_documents(query)
            
            # Limit to requested number
            results = results[:k] if results else []
            
            # Add retrieval metadata
            for doc in results:
                doc.metadata['retrieval_method'] = 'hybrid_with_cross_encoder' if self.enable_cross_encoder else 'hybrid_only'
                doc.metadata['cross_encoder_tier'] = self.cross_encoder_tier if self.enable_cross_encoder else None
            
            logging.debug(f"Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            return []
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about the configured retriever"""
        return {
            "has_retriever": self.retriever is not None,
            "retriever_type": type(self.retriever).__name__ if self.retriever else None,
            "cross_encoder_enabled": self.enable_cross_encoder,
            "cross_encoder_tier": self.cross_encoder_tier,
            "cross_encoder_available": CROSS_ENCODER_AVAILABLE
        }


def create_enhanced_hybrid_retriever(bm25_retriever: BaseRetriever,
                                   semantic_retriever: BaseRetriever,
                                   cross_encoder_tier: str = "balanced") -> HybridRetrieverWithCrossEncoder:
    """
    Factory function to create enhanced hybrid retriever
    
    Args:
        bm25_retriever: BM25 retriever
        semantic_retriever: Semantic/vector retriever
        cross_encoder_tier: Model tier for cross-encoder ("fast", "balanced", "quality")
    
    Returns:
        Enhanced hybrid retriever with cross-encoder reranking
    """
    return HybridRetrieverWithCrossEncoder(
        bm25_retriever=bm25_retriever,
        semantic_retriever=semantic_retriever,
        cross_encoder_tier=cross_encoder_tier,
        enable_cross_encoder=True
    )


# Example integration with your existing code pattern
def demo_integration_pattern():
    """
    Demonstrate how this integrates with your existing code pattern
    """
    
    # Mock retrievers for demonstration
    class MockBM25Retriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, run_manager=None) -> List[Document]:
            return [
                Document("BM25 result: Python programming tutorial", {"source": "bm25_1", "score": 0.8}),
                Document("BM25 result: Machine learning algorithms", {"source": "bm25_2", "score": 0.7}),
            ]
        
        def invoke(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query)
    
    class MockSemanticRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, run_manager=None) -> List[Document]:
            return [
                Document("Semantic result: Deep learning with neural networks", {"source": "semantic_1", "score": 0.9}),
                Document("Semantic result: AI and machine learning concepts", {"source": "semantic_2", "score": 0.8}),
            ]
        
        def invoke(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query)
    
    # Create retrievers
    bm25_retriever = MockBM25Retriever()
    semantic_retriever = MockSemanticRetriever()
    
    print("=== Hybrid Retriever with Cross-Encoder Integration Demo ===")
    
    # Test different configurations
    configs = [
        ("fast", "High-traffic production"),
        ("balanced", "Most use cases"),
    ]
    
    for tier, description in configs:
        print(f"\nTesting {tier.upper()} tier ({description}):")
        
        # Create enhanced retriever
        enhanced_retriever = create_enhanced_hybrid_retriever(
            bm25_retriever=bm25_retriever,
            semantic_retriever=semantic_retriever,
            cross_encoder_tier=tier
        )
        
        # Show configuration
        info = enhanced_retriever.get_retriever_info()
        print(f"  Retriever: {info['retriever_type']}")
        print(f"  Cross-encoder: {info['cross_encoder_enabled']} ({info['cross_encoder_tier']})")
        
        # Test retrieval
        query = "machine learning and AI algorithms"
        results = enhanced_retriever.retrieve(query, k=5)
        
        print(f"  Results: {len(results)} documents")
        for i, doc in enumerate(results[:3], 1):
            score = doc.metadata.get('cross_encoder_score', 'N/A')
            print(f"    {i}. [Score: {score}] {doc.page_content[:50]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_integration_pattern()