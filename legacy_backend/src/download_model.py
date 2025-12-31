from __future__ import annotations

"""Archived stub for `download_model.py`.

Model downloading utilities were archived. Restore full implementation from
Git history if needed.
"""

raise ImportError("backend.src.download_model has been archived to backend/archived/. Restore from Git history if needed.")

import logging
logger = logging.getLogger(__name__)

def download_cross_encoder_model():
    """
    Downloads and caches the HuggingFace Cross-Encoder model.
    This is intended to be run during the Docker image build process.
    """
    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from .config import CROSS_ENCODER_MODEL

        logger.info(f"Downloading cross-encoder model: {CROSS_ENCODER_MODEL}...")
        
        # Instantiating the model class will trigger the download
        # to the default cache location (~/.cache/huggingface/hub).
        _ = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
        
        logger.info(f"Successfully downloaded and cached {CROSS_ENCODER_MODEL}.")
    except Exception as e:
        logger.error(f"Error downloading model {CROSS_ENCODER_MODEL}: {e}", exc_info=True)
        # Exit with a non-zero status code to fail the Docker build if download fails
        exit(1)

if __name__ == "__main__":
    download_cross_encoder_model()
