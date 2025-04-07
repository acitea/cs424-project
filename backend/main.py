from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
import shutil
import os
import uuid
import time
import logging
from pydantic import BaseModel

# Import our modules
from models import get_model
import utils as app_utils
from config import config
from docs import setup_docs
# Uncomment to use real GAN models when ready
# from gan_integration import get_gan_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.logs_dir, "app.log"))
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
for directory in [config.upload_dir, config.results_dir, config.logs_dir]:
    app_utils.ensure_directory_exists(directory)

app = FastAPI(
    title=config.api_title,
    description=config.api_description,
    version=config.api_version
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images
app.mount("/static", StaticFiles(directory=config.results_dir), name="static")

# Setup custom docs
setup_docs(app)

# Define Pydantic models for API
class Task(BaseModel):
    id: str
    name: str
    description: str

class TaskList(BaseModel):
    tasks: List[Task]

class ImageResponse(BaseModel):
    result_url: str
    width: Optional[int] = None
    height: Optional[int] = None

@app.get("/")
def read_root():
    """Root endpoint that confirms the API is running"""
    return {"message": "Image Generation API is running", "version": "1.0.0"}

@app.post("/api/generate", response_model=ImageResponse)
async def generate_image(
    file: UploadFile = File(...),
    task_id: str = Form(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Generate an image using the specified task
    
    Args:
        file: Input image file
        task_id: ID of the task to perform
        background_tasks: FastAPI background tasks
        
    Returns:
        URL of the generated image
    """
    logger.info(f"Processing request for task: {task_id}")
    
    # Validate file extension
    if not app_utils.allowed_file(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(app_utils.ALLOWED_EXTENSIONS)}"
        )
    
    # Get the appropriate model for the task
    model = get_model(task_id)
    if not model:
        raise HTTPException(status_code=400, detail=f"Invalid task ID: {task_id}")
    
    # Create unique filenames
    file_id = str(uuid.uuid4())
    input_filename = f"uploads/{file_id}.png"
    output_filename = f"results/{file_id}.png"
    
    # Save the uploaded file
    try:
        with open(input_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Validate the image file
    is_valid, error_message = app_utils.validate_image(input_filename)
    if not is_valid:
        # Clean up the invalid file
        if os.path.exists(input_filename):
            os.remove(input_filename)
        raise HTTPException(status_code=400, detail=error_message)
    
    # Resize image if needed
    resized_input = app_utils.resize_image_if_needed(input_filename)
    
    # Process the image with the appropriate model
    try:
        success = model.generate(resized_input, output_filename)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        # Get the dimensions of the output image
        width, height = app_utils.get_image_dimensions(output_filename)
        
        # Return the result image
        return ImageResponse(
            result_url=f"/api/images/{file_id}.png",
            width=width,
            height=height
        )
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up temporary files in background
        if background_tasks:
            background_tasks.add_task(
                lambda: os.remove(input_filename) if os.path.exists(input_filename) else None
            )
            # Only remove the resized file if it's different from the input
            if resized_input != input_filename and background_tasks:
                background_tasks.add_task(
                    lambda: os.remove(resized_input) if os.path.exists(resized_input) else None
                )

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """
    Get a generated image by filename
    
    Args:
        filename: Name of the image file
        
    Returns:
        The image file
    """
    file_path = f"results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        file_path, 
        media_type="image/png",
        filename=f"generated-{filename}"
    )

@app.get("/api/tasks", response_model=TaskList)
def get_tasks():
    """
    Get list of available tasks
    
    Returns:
        List of available tasks
    """
    # Convert from config to API response
    tasks = [
        Task(
            id=task.id,
            name=task.name,
            description=task.description
        )
        for task in config.tasks
    ]
    
    return {"tasks": tasks}

@app.get("/api/health")
def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)