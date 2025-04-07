"""
Configuration for the backend application
"""
import os
from pydantic import BaseModel
from typing import List, Dict, Optional

class TaskConfig(BaseModel):
    """Configuration for a task"""
    id: str
    name: str
    description: str
    model_path: Optional[str] = None
    model_type: str
    parameters: Dict = {}

class AppConfig(BaseModel):
    """Application configuration"""
    # API settings
    api_title: str = "Image Generation API"
    api_description: str = "API for image generation using Cyclic GAN models"
    api_version: str = "1.0.0"
    
    # CORS settings
    cors_origins: List[str] = ["http://localhost:5173"]
    
    # File settings
    upload_dir: str = "uploads"
    results_dir: str = "results"
    logs_dir: str = "logs"
    allowed_extensions: List[str] = ["png", "jpg", "jpeg", "gif", "webp"]
    max_file_size_mb: int = 5
    
    # Model settings
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Task definitions
    tasks: List[TaskConfig] = [
        TaskConfig(
            id="task1",
            name="Style Transfer",
            description="Transform your photos with artistic style transfer",
            model_type="style_transfer",
            model_path="models/style_transfer.pth",
            parameters={
                "style_weight": 1.0,
                "content_weight": 0.1
            }
        ),
        TaskConfig(
            id="task2",
            name="Photo Enhancement",
            description="Enhance and improve photo quality",
            model_type="photo_enhancement",
            model_path="models/photo_enhancement.pth",
            parameters={
                "enhancement_level": 0.5
            }
        )
    ]

# Load environment-specific config
def load_config() -> AppConfig:
    """
    Load application configuration
    
    Returns:
        AppConfig object
    """
    # Default config
    config = AppConfig()
    
    # Override with environment variables where applicable
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        config.cors_origins = [os.getenv("FRONTEND_URL", "http://localhost:5173")]
        
        # Use GPU if available in production
        if os.getenv("USE_GPU", "false").lower() == "true":
            config.device = "cuda"
    
    # Create necessary directories
    for directory in [config.upload_dir, config.results_dir, config.logs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    return config

# Global config instance
config = load_config()