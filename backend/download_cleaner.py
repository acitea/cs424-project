"""
Script to download the U2NET model
"""
import os
import gdown
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = 'u2net.pth'
MODEL_URL = 'https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ'

def download_model():
    """Download the U2NET model if it doesn't exist"""
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading U2NET model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully")
    else:
        logger.info(f"Model already exists at {MODEL_PATH}")

if __name__ == "__main__":
    download_model()