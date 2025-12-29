import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.api_routes import router
from app.frontend_routes import router as frontend_router
from app.auth_routes import router as auth_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resource Management System API",
    description="Multi-department resource allocation through AI negotiation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the original API routes (negotiation system)
app.include_router(router)

# Include the new frontend routes (CRUD operations for frontend)
app.include_router(frontend_router)

# Include authentication routes
app.include_router(auth_router)

BASE_DIR = Path(__file__).parent.parent

@app.get("/", response_class=HTMLResponse)
async def root():
    demo_file = BASE_DIR / "demo.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resource Management System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #667eea; }
            .link { display: inline-block; margin: 10px; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; }
            .link:hover { background: #5568d3; }
        </style>
    </head>
    <body>
        <h1>🤖 Resource Management System</h1>
        <p>Welcome to the Resource Management System API</p>
        <a href="/demo" class="link">🎯 Live Demo</a>
        <a href="/docs" class="link">📖 API Documentation</a>
        <a href="/health" class="link">❤️ Health Check</a>
    </body>
    </html>
    """

@app.get("/demo", response_class=HTMLResponse)
async def demo():
    demo_file = BASE_DIR / "demo.html"
    if demo_file.exists():
        with open(demo_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Demo file not found</h1><p>Please ensure demo.html exists in the project root.</p>"

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
