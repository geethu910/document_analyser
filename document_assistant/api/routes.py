"""
FastAPI routes for document assistant API using FastAPI-Users (async)
"""
import os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from document_assistant.api.database import (
    get_async_session, UploadedFile, SearchHistory, User
)
from document_assistant.api.auth import current_active_user
from document_assistant.api.schemas import (
    FileUploadResponse, SearchResponse, UserFilesResponse
)
from document_assistant.services.file_service import FileService

# Router
router = APIRouter()

# File search lazy loading (singleton)
_file_search_instance: None = None
_file_service: None = None

# This is added to ensure that the FileSearch class is only initialized when needed, which allows the API to start up faster without waiting for the model to download. The first time get_file_search() is called, it will import and create an instance of FileSearch. Subsequent calls will return the already initialized instance.
def get_file_search():
    """Lazy load FileSearch to avoid blocking on model download"""
    global _file_search_instance
    if _file_search_instance is None:
        from document_assistant.file_search_embeddings import FileSearch
        _file_search_instance = FileSearch()
    return _file_search_instance

def get_file_service() -> FileService:
    """Get FileService singleton"""
    global _file_service
    if _file_service is None:
        from document_assistant.services.file_service import FileService
        _file_service = FileService(get_file_search())
    return _file_service

# ===== PLACEHOLDER ROUTES =====
# These routes require authentication - will be implemented once FastAPI-Users is working

@router.post("/api/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Upload a document file (PDF, DOCX, TXT)"""
    try:
        # get service
        file_service = get_file_service()
        
        _file_search_instance.load_file(file.file)  # Process the uploaded file and add to vector database
        # raise HTTPException(
        #     status_code=status.HTTP_501_NOT_IMPLEMENTED,
        #     detail="Requires authentication - register at /api/auth/register first"
        # )
    except HTTPException:
        raise  # re-raise the exception

    except Exception as e:
        await session.rollback()  # Rollback any pending transactions in case of error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/api/files/list", response_model=UserFilesResponse)
async def list_user_files(
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """List all files uploaded by current user"""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Requires authentication - register at /api/auth/register first"
    )


@router.delete("/api/files/{file_id}")
async def delete_file(
    file_id: int,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Delete a file and remove from vector database"""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Requires authentication - register at /api/auth/register first"
    )


@router.post("/api/search", response_model=SearchResponse)
async def search_documents(
    query: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Search documents using semantic search"""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Requires authentication - register at /api/auth/register first"
    )


@router.get("/api/search/history")
async def get_search_history(
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
    limit: int = 20
):
    """Get search history for current user"""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Requires authentication - register at /api/auth/register first"
    )


@router.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Document Assistant API"}
