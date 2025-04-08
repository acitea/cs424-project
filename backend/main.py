from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.responses import JSONResponse
import os
import uuid
import shutil
import logging
from models import get_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import download_cleaner
    download_cleaner.download_model()
except Exception as e:
    logging.error(f"Error downloading U2NET model: {str(e)}")
    logging.warning("The application may not function correctly without the U2NET model")

# Directory setup
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(
    title="Image Transformation API",
    description="API for image transformation tasks using GAN models",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Image Transformation API is running"}

@app.get("/tasks")
async def get_tasks():
    """Get a list of available transformation tasks"""
    return {
        "tasks": [
            {
                "id": "task1",
                "name": "Style Transfer",
                "description": "Apply a stylized filter to your image",
                "conversion_key":"Human,Cartoon"
            },
            {
                "id": "task2",
                "name": "Pokemon Generator",
                "description": "Transform your image into Pokemon style",
                "conversion_key":"Animal,Pokemon"
            }
        ]
    }

@app.get("/available-models/{task_id}")
async def get_available_models(task_id: str, reverse: bool):
    """
    Get list of available model files (.pth) for the specified task
    
    Args:
        task_id: The task identifier (task1, task2)
    
    Returns:
        JSON with list of available model files
    """
    try:
        base_path = os.path.join("weights", task_id)
        
        if not os.path.exists(base_path):
            return {"status": "success", "models": []}
        
        models = []
        
        # For task2, scan both forward and reverse directories
        if task_id == "task2":
            weights_key = "forward" if not reverse else "reverse"
            paths = os.path.join(base_path, f"{weights_key}")
            if os.path.exists(paths):
                for file in os.listdir(paths):
                    if file.endswith(".pth"):
                        models.append({
                            "id": os.path.join(weights_key, file),
                            "name": f"{weights_key.capitalize()}: {file}",
                            "description": f"Model for {weights_key} transformation",
                            "reverse": reverse
                        })
        else:
            # For other tasks, just scan the task directory for all .pth files
            for root, _, files in os.walk(base_path):
                rel_path = os.path.relpath(root, base_path)
                for file in files:
                    if file.endswith(".pth"):
                        rel_file_path = os.path.join(rel_path, file) if rel_path != "." else file
                        models.append({
                            "id": rel_file_path,
                            "name": file,
                            "description": f"Model from {rel_path if rel_path != '.' else 'root directory'}",
                            "reverse": "reverse" in rel_file_path.lower()
                        })
                
        return {"status": "success", "models": models}
            
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transform/{task_id}")
async def transform_image(
    task_id: str,
    file: UploadFile = File(...),
    model_path: str = Form(...),  # Path to the specific .pth file
    background_tasks: BackgroundTasks = None
):
    """
    Transform an uploaded image using the specified model file
    
    Args:
        task_id: The task identifier (task1, task2)
        file: The uploaded image file
        model_path: Path to the specific model file (.pth) relative to the task weights directory
    
    Returns:
        JSON with the image URL
    """
    try:
        # Determine if this is a reverse transformation based on the model path
        reverse = "reverse" in model_path.lower()
        logger.info(f"Received request for task {task_id} with model_path={model_path}, reverse={reverse}")
        
        # Get the model with the appropriate constructor
        model = get_model(task_id, reverse=reverse, model_path=model_path)
        if not model:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}"
        file_extension = os.path.splitext(file.filename)[1]
        input_path = os.path.join(UPLOAD_DIR, f"{filename}{file_extension}")
        output_path = os.path.join(RESULTS_DIR, f"{filename}_result{file_extension}")
        
        # Save the uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        logger.info(f"Processing image for task {task_id} with model {model_path}")
        success = model.generate(input_path, output_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Image transformation failed")
        
        # Clean up the input file in the background
        if background_tasks:
            background_tasks.add_task(os.remove, input_path)
        
        # Return the URL to the transformed image
        result_url = f"/results/{os.path.basename(output_path)}"
        return {"status": "success", "result_url": result_url}
        
    except Exception as e:
        logger.error(f"Error processing transformation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{filename}")
async def get_result(filename: str):
    """
    Get a transformed image result
    
    Args:
        filename: The result filename
    
    Returns:
        The image file
    """
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result not found")
    
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)