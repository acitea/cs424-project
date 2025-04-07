"""
Utility functions for the backend application
"""
import os
import uuid
from typing import Tuple, Optional
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_IMAGE_SIZE_MB = 5

def allowed_file(filename: str) -> bool:
    """
    Check if the file extension is allowed
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an image file
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"File size exceeds {MAX_IMAGE_SIZE_MB}MB limit"
    
    # Check if it's a valid image
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"
    
    return True, None

def generate_unique_filename(extension: str = "png") -> str:
    """
    Generate a unique filename
    
    Args:
        extension: File extension (without the dot)
        
    Returns:
        Unique filename with extension
    """
    return f"{uuid.uuid4()}.{extension}"

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure a directory exists, create it if it doesn't
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def get_image_dimensions(file_path: str) -> Tuple[int, int]:
    """
    Get image dimensions
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple of (width, height)
    """
    with Image.open(file_path) as img:
        return img.size

def resize_image_if_needed(file_path: str, max_dimension: int = 1024) -> str:
    """
    Resize an image if either dimension exceeds the maximum
    
    Args:
        file_path: Path to the image file
        max_dimension: Maximum dimension (width or height)
        
    Returns:
        Path to the resized image (same as input if no resize needed)
    """
    with Image.open(file_path) as img:
        width, height = img.size
        
        # Check if resize is needed
        if width <= max_dimension and height <= max_dimension:
            return file_path
        
        # Calculate new dimensions
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save the resized image
        resized_path = f"{os.path.splitext(file_path)[0]}_resized.png"
        resized_img.save(resized_path)
        
        logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return resized_path