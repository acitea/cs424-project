# Image Transformation Studio

A web application for applying GAN-based image transformations using a modern React frontend and FastAPI backend.

## Features

- Style Transfer transformation
- Pokemon-style image generation
- Easy-to-use interface with drag-and-drop uploads
- Real-time progress indicators
- Responsive design for all device sizes

## Project Structure

The project is organized into two main parts:

### Frontend (React + Vite + TypeScript)

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── ImageDisplay.tsx
│   │   ├── ImageGeneration.tsx
│   │   ├── ImageUploader.tsx
│   │   ├── LoadingIndicator.tsx
│   │   └── theme-provider.tsx
│   ├── store.ts
│   ├── utils.ts
│   ├── App.tsx
│   └── main.tsx
├── .env
├── index.html
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### Backend (FastAPI + Python)

```
.
├── uploads/              # Temporary storage for uploaded images
├── results/              # Output directory for transformed images
├── weights/              # Pre-trained model weights
│   ├── task1/            # Style transfer weights
│   └── task2/            # Pokemon generator weights
│       ├── forward/      # Real to Pokemon transformation
│       └── reverse/      # Pokemon to real transformation
├── models.py             # Model definitions and factory
├── bg_removal.py         # Background removal utilities
├── download_cleaner.py   # U2NET downloader
└── main.py               # FastAPI application
```

## Setup and Installation

### Using Docker Compose (Recommended)


1. Start the application using Docker Compose:
   ```
   docker-compose up -d
   ```

2. Access the application at `http://localhost:3000`

# Image Transformation API

A FastAPI-based backend for image transformation using GAN models.

## Overview

This API provides endpoints for transforming images using various pre-trained GAN models. It supports multiple transformation tasks like style transfer and Pokemon image generation.

## Features

- **Multiple Transformation Models**: Support for different GAN-based transformations
- **Model Selection**: Users can choose specific model weights for each transformation
- **RESTful API**: Clean, well-documented API endpoints
- **Automatic Background Removal**: Uses U2NET for preprocessing
- **File Management**: Handles file uploads, storage, and cleanup

## Requirements

- Python 3.8+
- FastAPI
- PyTorch
- Pillow
- U2NET (downloaded automatically on startup)

## Directory Structure

```
.
├── uploads/              # Temporary storage for uploaded images
├── results/              # Output directory for transformed images
├── weights/              # Pre-trained model weights
│   ├── task1/            # Style transfer weights
│   └── task2/            # Pokemon generator weights
│       ├── forward/      # Real to Pokemon transformation
│       └── reverse/      # Pokemon to real transformation
├── models.py             # Model definitions and factory
├── bg_removal.py         # Background removal utilities
├── download_cleaner.py   # U2NET downloader
├── run.sh                # Execution script
└── main.py               # FastAPI application
```

## Installation

### Manual Setup (Recommended)

#### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make the run script executable:
   ```bash
   chmod +x run.sh
   ```

5. Run the backend server using the script (this will download the U2NET model if needed):
   ```bash
   ./run.sh
   ```

   Alternatively, you can run the server directly:
   ```bash
   python download_cleaner.py  # Download the model first
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Access the frontend at `http://localhost:5173`

### Alternative Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-transformation-api.git
   cd image-transformation-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download pre-trained models:
   - Place model weights in the appropriate directories under `weights/`
   - U2NET model will be downloaded automatically on first run

## Running the API

Start the server:

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## API Endpoints

### Get Available Tasks

```
GET /tasks
```

Returns a list of available transformation tasks.

### Get Available Models

```
GET /available-models/{task_id}?reverse={boolean}
```

Returns a list of available models for a specific task.

Parameters:
- `task_id`: The task identifier (e.g., task1, task2)
- `reverse`: Boolean indicating whether to fetch reverse transformation models

### Transform Image

```
POST /transform/{task_id}
```

Transforms an uploaded image using the specified model.

Parameters:
- `task_id`: The task identifier
- `file`: The image file to transform (form data)
- `model_path`: Path to the specific model file (form data)

### Get Result

```
GET /results/{filename}
```

Returns a transformed image file.

## Examples

### Transforming an image to Pokemon style

```python
import requests

# Get available models
response = requests.get("http://localhost:8000/available-models/task2?reverse=false")
models = response.json()["models"]

# Upload and transform an image
with open("cat.jpg", "rb") as f:
    files = {"file": f}
    data = {"model_path": models[0]["id"]}
    response = requests.post("http://localhost:8000/transform/task2", files=files, data=data)

# Get the result URL
result_url = response.json()["result_url"]
print(f"Transformed image: http://localhost:8000{result_url}")
```

## Error Handling

The API returns appropriate HTTP status codes and error messages in case of failures:

- `404`: Task or model not found
- `422`: Invalid request parameters
- `500`: Server error during processing

## Additional Information

- The API automatically cleans up temporary files
- U2NET model is used for background removal to improve transformation quality
- All model file paths are relative to the weights directory

## License

[MIT License](LICENSE)

## API Endpoints

- `GET /` - API status
- `GET /tasks` - List available transformation tasks
- `POST /transform/{task_id}` - Transform an image using specified task
   - Parameters:

   - task_id: The task identifier
   - file: The image file to transform (form data)
   - 
model_path: Path to the specific model file (form data)
- `GET /results/{filename}` - Get a transformed image result
- `/available-models/{task_id}?reverse={boolean}` - Get available models for this task
   task_id: The task identifier (e.g., task1, task2)
   reverse: Boolean indicating whether to fetch reverse transformation models

## Example usage if you want to script for tests


## Technologies Used

### Frontend
- React 18
- TypeScript
- Vite
- Jotai (State Management)
- Tailwind CSS
- ShadCN UI Components
- React Dropzone

### Backend
- FastAPI
- Python 3.10+
- PyTorch
- U2NET (Background Removal)
- CycleGAN (Style Transfer)

## License

This project is licensed under the MIT License - see the LICENSE file for details.