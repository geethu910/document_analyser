"""
FastAPI application for Document Assistant with RAG and FastAPI-Users (async)
"""
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    auth_manager.get_auth_router(auth_backend),
    prefix="/api/auth/jwt",
    tags=["auth"],
)
app.include_router(
    auth_manager.get_register_router(UserRead, UserCreate),
    prefix="/api/auth",
    tags=["auth"],
)
app.include_router(
    auth_manager.get_reset_password_router(),
    prefix="/api/auth",
    tags=["auth"],
)
app.include_router(
    auth_manager.get_verify_router(UserRead),
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


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Document Assistant API",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "document_assistant.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
