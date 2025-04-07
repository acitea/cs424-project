"""
Custom API documentation for the FastAPI application
"""
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

# Create directory for templates if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Create a basic template for the docs page
with open("templates/docs.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Image Generation API Documentation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .endpoint {
            margin-bottom: 2rem;
            padding: 1rem;
            border-radius: 0.25rem;
            background-color: #f8f9fa;
            border-left: 4px solid #6c757d;
        }
        .endpoint-post {
            border-left-color: #198754;
        }
        .endpoint-get {
            border-left-color: #0d6efd;
        }
        .method {
            font-weight: bold;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            display: inline-block;
            min-width: 60px;
            text-align: center;
        }
        .method-post {
            background-color: #198754;
            color: white;
        }
        .method-get {
            background-color: #0d6efd;
            color: white;
        }
        pre {
            background-color: #212529;
            color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.25rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Image Generation API Documentation</h1>
        
        <div class="alert alert-info">
            This documentation provides an overview of the available endpoints and how to use them.
            For interactive API documentation, visit the <a href="/docs">Swagger UI</a>.
        </div>
        
        <h2 class="mt-4 mb-3">Authentication</h2>
        <p>No authentication is required for development. In production, appropriate authentication would be implemented.</p>
        
        <h2 class="mt-4 mb-3">Endpoints</h2>
        
        <div class="endpoint endpoint-get">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3 class="m-0">Get Available Tasks</h3>
                <span class="method method-get">GET</span>
            </div>
            <p><strong>Endpoint:</strong> <code>/api/tasks</code></p>
            <p>Returns a list of available image generation tasks.</p>
            
            <h4 class="mt-3">Response Example:</h4>
            <pre>{
  "tasks": [
    {
      "id": "task1",
      "name": "Style Transfer",
      "description": "Transform your photos with artistic style transfer"
    },
    {
      "id": "task2",
      "name": "Photo Enhancement",
      "description": "Enhance and improve photo quality"
    }
  ]
}</pre>
        </div>
        
        <div class="endpoint endpoint-post">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3 class="m-0">Generate Image</h3>
                <span class="method method-post">POST</span>
            </div>
            <p><strong>Endpoint:</strong> <code>/api/generate</code></p>
            <p>Generate an image based on an input image and a task ID.</p>
            
            <h4 class="mt-3">Request Parameters:</h4>
            <ul>
                <li><code>file</code> (required): The input image file (multipart/form-data)</li>
                <li><code>task_id</code> (required): The ID of the task to perform</li>
            </ul>
            
            <h4 class="mt-3">Response Example:</h4>
            <pre>{
  "result_url": "/api/images/f47ac10b-58cc-4372-a567-0e02b2c3d479.png",
  "width": 512,
  "height": 512
}</pre>
        </div>
        
        <div class="endpoint endpoint-get">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3 class="m-0">Get Generated Image</h3>
                <span class="method method-get">GET</span>
            </div>
            <p><strong>Endpoint:</strong> <code>/api/images/{filename}</code></p>
            <p>Get a previously generated image by filename.</p>
            
            <h4 class="mt-3">Request Parameters:</h4>
            <ul>
                <li><code>filename</code> (required): The filename of the generated image</li>
            </ul>
            
            <h4 class="mt-3">Response:</h4>
            <p>Returns the image file with appropriate content type.</p>
        </div>
        
        <h2 class="mt-4 mb-3">Error Responses</h2>
        <p>All endpoints may return the following error responses:</p>
        
        <div class="table-responsive">
            <table class="table">
                <thead>
                    <tr>
                        <th>Status Code</th>
                        <th>Description</th>
                        <th>Response Example</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>400</td>
                        <td>Bad Request</td>
                        <td><pre>{"detail": "Invalid file type"}</pre></td>
                    </tr>
                    <tr>
                        <td>404</td>
                        <td>Not Found</td>
                        <td><pre>{"detail": "Image not found"}</pre></td>
                    </tr>
                    <tr>
                        <td>500</td>
                        <td>Internal Server Error</td>
                        <td><pre>{"detail": "Failed to process image"}</pre></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
    """)

templates = Jinja2Templates(directory="templates")

def setup_docs(app: FastAPI):
    """
    Set up custom API documentation
    
    Args:
        app: FastAPI app instance
    """
    @app.get("/api-docs", response_class=HTMLResponse)
    async def get_api_docs(request: Request):
        """Get custom API documentation page"""
        return templates.TemplateResponse("docs.html", {"request": request})
    
    # We keep the default Swagger UI but customize its path
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Get Swagger UI"""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
        )