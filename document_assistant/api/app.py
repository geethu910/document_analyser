"""
FastAPI application for Document Assistant with RAG and FastAPI-Users (async)
"""
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from document_assistant.api.database import Base, create_db_and_tables
from document_assistant.api.auth import auth_backend, auth_manager
from document_assistant.api.schemas import UserCreate, UserRead, UserUpdate
from document_assistant.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup code - create database tables
    await create_db_and_tables()
    yield
    # Shutdown code (if needed)

# Initialize FastAPI app
app = FastAPI(
    title="Document Assistant API",
    description="RAG-based document search with semantic search and LLM integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include FastAPI-Users auth routes
app.include_router(
    auth_manager.get_register_router(UserRead, UserCreate),
    prefix="/api/auth",
    tags=["auth"],
)
# Verify router - handles email verification with token
app.include_router(
    auth_manager.get_verify_router(UserRead),
    prefix="/api/auth",
    tags=["auth"],
)
app.include_router(
    auth_manager.get_auth_router(auth_backend, requires_verification=True),
    prefix="/api/auth/jwt",
    tags=["auth"],
)

app.include_router(
    auth_manager.get_reset_password_router(),
    prefix="/api/auth",
    tags=["auth"],
)

app.include_router(
    auth_manager.get_users_router(UserRead, UserUpdate),
    prefix="/api/users",
    tags=["users"],
)

# Include custom routes
app.include_router(router)

# Serve verify.html explicitly
@app.get("/verify")
async def verify_page():
    """Serve the email verification page"""
    verify_file = Path(__file__).parent.parent.parent / "static" / "verify.html"
    if verify_file.exists():
        return FileResponse(verify_file, media_type="text/html")
    return {"detail": "Verification page not found"}

# Mount static files
# This serves HTML, CSS, JS from the static/ folder
# This tells fastapi : Any request to a path that doesn't match an API route
# Check the static/ folder for a file with that name Serve it automatically
static_dir = Path(__file__).parent.parent.parent / "static"
print(f"Checking for static files in: {static_dir}")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    print("Static files mounted successfully")
else:
    print(f"Warning: Static directory not found at {static_dir}. Static files will not be served.")


# @app.get("/")
# def root():
#     """Root endpoint"""
#     return {
#         "message": "Welcome to Document Assistant API",
#         "docs": "/docs",
#         "health": "/api/health"
#     }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "document_assistant.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
