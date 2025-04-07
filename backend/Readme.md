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
backend/
├── generators/
│   └── task2/
│       └── PokemonGenerator.py
├── weights/
│   └── task2/
│       └── PokemonCycleGanGAB.pth
├── uploads/
├── results/
├── bg_removal.py
├── models.py
├── main.py
├── utils.py
├── download_cleaner.py
├── run.sh
├── requirements.txt
└── Dockerfile
```

## Setup and Installation

### Using Docker Compose (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/your-username/image-transformation-studio.git
   cd image-transformation-studio
   ```

2. Start the application using Docker Compose:
   ```
   docker-compose up -d
   ```

3. Access the application at `http://localhost:3000`

### Manual Setup

#### Backend

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Make the run script executable:
   ```
   chmod +x run.sh
   ```

5. Run the backend server using the script (this will download the U2NET model if needed):
   ```
   ./run.sh
   ```

   Alternatively, you can run the server directly:
   ```
   python download_cleaner.py  # Download the model first
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Frontend

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```

4. Access the frontend at `http://localhost:5173`

## API Endpoints

- `GET /` - API status
- `GET /tasks` - List available transformation tasks
- `POST /transform/{task_id}` - Transform an image using specified task
- `GET /results/{filename}` - Get a transformed image result

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