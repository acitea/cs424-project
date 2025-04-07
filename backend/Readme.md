ts         # Jotai state management
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── backend/                 # FastAPI backend
│   ├── main.py              # Main FastAPI application
│   ├── models.py            # GAN model implementations
│   └── requirements.txt     # Python dependencies
└── README.md                # This file
```

## Backend Setup

1. Create a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Unix or MacOS
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend:
   ```bash
   uvicorn main:app --reload
   ```

The backend will be available at `http://localhost:8000`.

## Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install axios
   ```

2. Run the frontend:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`.

## Features

- Two image processing tasks (style transfer and photo enhancement)
- Upload images via drag-and-drop
- Real-time progress indication
- Download processed images
- Responsive UI

## Backend API Endpoints

- `GET /api/tasks` - Get list of available tasks
- `POST /api/generate` - Generate an image from uploaded file and task ID
- `GET /api/images/{filename}` - Get generated image file

## Technical Details

### Backend

The backend uses FastAPI to provide REST API endpoints. It includes:

- **FastAPI Application**: Handles HTTP requests, file uploads, and serves generated images
- **Mock GAN Models**: Simulates image processing with filters for demonstration
- **CORS Support**: Configured for local development
- **File Storage**: Manages uploaded and generated images

### Frontend

The frontend is built with React, TypeScript, and modern UI components:

- **State Management**: Uses Jotai for app-wide state management
- **Component Library**: Uses ShadCN UI components 
- **File Handling**: Supports drag-and-drop with previews
- **API Integration**: Communicates with the backend via axios
- **Responsive Design**: Works on mobile and desktop devices

## Extending the Application

To integrate real Cyclic GAN models:

1. Update the `models.py` file with actual PyTorch/TensorFlow model loading and inference
2. Add model weights to a separate directory
3. Implement pre-processing and post-processing functions for the images
4. Update the endpoint to handle model-specific parameters

## Screenshots

(Add screenshots here when available)