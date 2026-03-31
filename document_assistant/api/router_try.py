import os
import hashlib
import tempfile
import mimetypes
from pathlib import Path
from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from document_assistant.api.database import (
    get_async_session, UploadedFile, SearchHistory, User
)
from document_assistant.api.auth import current_active_user
from document_assistant.api.schemas import (
    FileUploadResponse, SearchResponse, UserFilesResponse
)
from document_assistant.services.file_service import FileService
from document_assistant.file_search_embeddings import FileSearch

router = APIRouter()

_file_search_instance: Optional[FileSearch] = None
_file_service: Optional[FileService] = None


def get_file_search() -> FileSearch:
    """Lazy load FileSearch"""
    global _file_search_instance
    if _file_search_instance is None:
        _file_search_instance = FileSearch()
    return _file_search_instance


def get_file_service() -> FileService:
    """Get FileService singleton"""
    global _file_service
    if _file_service is None:
        _file_service = FileService(get_file_search())
    return _file_service


# ===== FILE MANAGEMENT ROUTES =====

@router.post("/api/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Upload a document file (PDF, DOCX, TXT) - with proper temp file cleanup"""
    
    temp_file_path = None
    
    try:
        # Get service
        file_service = get_file_service()
        
        # ✅ Step 1: Validate filename
        is_valid, error_msg = file_service.validate_filename(file.filename)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        file_ext = Path(file.filename).suffix.lower()
        
        # ✅ Step 2: Create temp file and stream upload
        os.makedirs("./temp_uploads", exist_ok=True)
        
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,  # ← We'll delete manually
                suffix=file_ext,
                dir="./temp_uploads"
            ) as tmp_file:
                temp_file_path = tmp_file.name
                
                # Stream file in chunks (4KB at a time)
                hasher = hashlib.sha256()
                chunk_size = 4096
                total_size = 0
                
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    
                    tmp_file.write(chunk)
                    hasher.update(chunk)
                    total_size += len(chunk)
        
        except Exception as e:
            # ✅ Delete temp file if creation failed
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File upload failed: {str(e)}"
            )
        
        file_hash = hasher.hexdigest()
        file_size_kb = round(total_size / 1024, 2)
        
        # ✅ Step 3: Validate file size
        max_size_kb = 50 * 1024  # 50MB
        if file_size_kb > max_size_kb:
            # ✅ Delete temp file if too large
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max: 50MB. Got: {file_size_kb / 1024:.1f}MB"
            )
        
        # ✅ Step 4: Check for duplicates (before processing)
        existing_query = select(UploadedFile).where(
            UploadedFile.file_hash == file_hash,
            UploadedFile.user_id == current_user.id
        )
        existing = await session.execute(existing_query)
        if existing.scalars().first():
            # ✅ Delete temp file - duplicate found
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass
            
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"File '{file.filename}' was already uploaded"
            )
        
        try:
            # ✅ Step 5: Process file (vectorize from temp location)
            file_search = get_file_search()
            vector_count = file_search.load_file(temp_file_path)
            
            # ✅ Step 6: Extract metadata
            mime_type, _ = mimetypes.guess_type(file.filename)
            if not mime_type:
                mime_type = "application/octet-stream"
            
            # ✅ Step 7: Save to database
            db_file = UploadedFile(
                user_id=current_user.id,
                file_name=file.filename,
                file_hash=file_hash,
                file_size_kb=file_size_kb,
                file_type=file_ext[1:],
                mime_type=mime_type,
                status="indexed",
                vector_count=vector_count or 0
            )
            session.add(db_file)
            await session.commit()
            await session.refresh(db_file)
            
            return FileUploadResponse(
                file_id=str(db_file.id),
                file_name=db_file.file_name,
                file_size_kb=db_file.file_size_kb,
                file_type=db_file.file_type,
                vector_count=db_file.vector_count,
                status=db_file.status,
                upload_time=db_file.upload_time.isoformat()
            )
        
        except Exception as e:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File processing failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
    
    finally:
        # ✅ CRITICAL: Always clean up temp file
        # This runs regardless of success or failure
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"🗑️ Cleaned up temp file: {temp_file_path}")
            except OSError as e:
                print(f"⚠️ Failed to delete temp file {temp_file_path}: {e}")
                # Log this but don't crash - temp can be cleaned later


@router.get("/api/files/list", response_model=UserFilesResponse)
async def list_user_files(
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """List all files uploaded by current user"""
    try:
        query = select(UploadedFile).where(
            UploadedFile.user_id == current_user.id
        ).order_by(UploadedFile.upload_time.desc())
        
        result = await session.execute(query)
        user_files = result.scalars().all()
        
        total_size_kb = sum(f.file_size_kb for f in user_files)
        total_vectors = sum(f.vector_count for f in user_files)
        
        return UserFilesResponse(
            files=[
                {
                    "id": str(f.id),
                    "file_name": f.file_name,
                    "file_size_kb": f.file_size_kb,
                    "file_type": f.file_type,
                    "status": f.status,
                    "vector_count": f.vector_count,
                    "upload_time": f.upload_time.isoformat()
                }
                for f in user_files
            ],
            total_files=len(user_files),
            total_size_kb=round(total_size_kb, 2),
            total_vectors=total_vectors
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {str(e)}"
        )


@router.delete("/api/files/{file_id}")
async def delete_file(
    file_id: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Delete a file and remove from vector database"""
    
    try:
        # Validate UUID
        try:
            file_uuid = UUID(file_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file ID format"
            )
        
        # Query database
        query = select(UploadedFile).where(
            UploadedFile.id == file_uuid,
            UploadedFile.user_id == current_user.id
        )
        result = await session.execute(query)
        db_file = result.scalars().first()
        
        if not db_file:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        try:
            # Delete from ChromaDB
            file_search = get_file_search()
            file_search.delete_file(db_file.file_hash)
        except Exception as e:
            print(f"⚠️ Warning: Failed to delete from ChromaDB: {e}")
            # Continue anyway - still delete from DB
        
        # Delete from database
        await session.delete(db_file)
        await session.commit()
        
        return {"message": f"File '{db_file.file_name}' deleted successfully"}
    
    except HTTPException:
        raise
    
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )


# ===== SEARCH ROUTES =====

@router.post("/api/search", response_model=SearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Search documents using semantic search"""
    
    try:
        file_search = get_file_search()
        
        # Perform semantic search
        search_results = file_search.search(query=query, top_k=limit)
        
        # Save search history
        search_history = SearchHistory(
            user_id=current_user.id,
            query=query,
            results_count=len(search_results) if search_results else 0
        )
        session.add(search_history)
        await session.commit()
        
        return SearchResponse(
            query=query,
            results_count=len(search_results) if search_results else 0,
            results=search_results or [],
            search_id=str(search_history.id)
        )
    
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/api/search/history")
async def get_search_history(
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
    limit: int = Query(20, ge=1, le=100)
):
    """Get search history for current user"""
    
    try:
        query = select(SearchHistory).where(
            SearchHistory.user_id == current_user.id
        ).order_by(SearchHistory.search_time.desc()).limit(limit)
        
        result = await session.execute(query)
        history = result.scalars().all()
        
        return {
            "history": [
                {
                    "id": str(h.id),
                    "query": h.query,
                    "results_count": h.results_count,
                    "search_time": h.search_time.isoformat()
                }
                for h in history
            ],
            "total": len(history)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get search history: {str(e)}"
        )


# ===== HEALTH CHECK =====

@router.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Document Assistant API",
        "version": "1.0.0"
    }