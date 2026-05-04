"""
FastAPI routes for document assistant API using FastAPI-Users (async)
"""
import datetime
import hashlib
import os
from pathlib import Path
import tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from document_assistant.api.database import (
    get_async_session, UploadedFile, SearchHistory, User
)
from document_assistant.api.auth import current_active_user
from document_assistant.api.schemas import (
    FileUploadResponse, SearchResponse, UserFilesResponse
)
from document_assistant.services.file_service import FileService
from document_assistant.services.cache_service import *

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
        # validate file type
        if not file_service.validate_file_type(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Only PDF, DOCX, and TXT are allowed."
            )
        file_ext = Path(file.filename).suffix.lower()
        # Step 2: Create temp file and stream upload
        os.makedirs("./temp_uploads", exist_ok=True)
        
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
                # read a chunks untill eof, await is used to ensure that the event loop can handle other requests while waiting for file I/O
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                
                tmp_file.write(chunk)
                hasher.update(chunk)
                total_size += len(chunk)
        file_hash = hasher.hexdigest()
        # Step 3: Validate file size
        file_size_kb = round(total_size / 1024, 2)
        max_size_kb = 50 * 1024  # 50MB
        if file_size_kb > max_size_kb:
            # Delete temp file if too large
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max: 50MB. Got: {file_size_kb / 1024:.1f}MB"
            )
        # step 4: check if file is already loaded for the current user
        existing = await session.execute(
            select(UploadedFile).where(
                UploadedFile.file_name == file.filename,
                UploadedFile.user_id == current_user.id
            )
        )  
        existing_file = existing.scalars().first() # here scalars() is used to extract the UploadedFile objects from the result, and first() retrieves the first one (or None if there are no matches). This allows us to check if a file with the same hash has already been uploaded by the current user, which helps prevent duplicate entries in the database and redundant processing in the vector database.
        if existing_file:
            if existing_file.file_hash == file_hash:
                print(f'File {file.filename} already exists with same hash, skipping re-indexing')
                # SAME FILE: Skip (already indexed)
                return FileUploadResponse( 
                    file_id= str(existing_file.id),
                    file_name = file.filename,
                    file_size = existing_file.file_size,
                    vector_count= existing_file.vector_count,
                    upload_time=existing_file.upload_time,
                    status= "FFile is already uploaded"

                )
            else:
                # different file, same name:: replace
                print(f'Replacing older version of {file.filename}')
                # delete file from both places
                # delete from vector db
                get_file_service().delete_file(existing_file.file_hash, str(current_user.id))
                # delete from database
                await session.delete(existing_file)
                await session.commit()
            
        # create embeddings and add to vector database
        chunk_count = file_service.upload_file(
            temp_file_path, user_id=str(current_user.id), 
            file_hash=file_hash,
            original_file_name = file.filename
            )  # Process the uploaded file and add to vector database

        # save file metadata to database
        new_file = UploadedFile(
            user_id=current_user.id,
            file_name=file.filename,
            file_hash=file_hash,
            file_size=total_size,
            file_type=file_ext[1:],  # remove dot from extension
            status="indexed",
            vector_count=chunk_count or 0
        )
        session.add(new_file)
        await session.commit()  # Commit the transaction to save the file metadata in the database
        await session.refresh(new_file)

        # increment cache version of user inorder to exclude stale versions
        await invalidate_user_cache(current_user.id)
        return FileUploadResponse(
            file_id=str(new_file.id),
            file_name=new_file.file_name,
            file_size=new_file.file_size,
            vector_count=chunk_count,
            status=new_file.status,
            upload_time=new_file.upload_time.isoformat()
            )
        
    except HTTPException:
        raise  # re-raise the exception

    except Exception as e:
        await session.rollback()  # Rollback any pending transactions in case of error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        # clean up temp file
        # This runs regardless of success or failure
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temp file: {temp_file_path}")
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
        result = await session.execute(
            select(UploadedFile).where(UploadedFile.user_id == current_user.id))
        files = result.scalars().all()
        total_size_kb = sum(f.file_size for f in files)

        return UserFilesResponse(
            files=[
                FileUploadResponse(
                    file_id=str(f.id),
                    file_name=f.file_name,
                    file_size=f.file_size,
                    vector_count=f.vector_count,
                    status=f.status,
                    upload_time=f.upload_time.isoformat()
                )
                for f in files
            ],
            total_files=len(files),
            total_size_kb=round(total_size_kb, 2),
        )

    except Exception as e:
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error retrieving files: {str(e)}"
            )

@router.delete("/api/files/{file_id}")
async def delete_file(
    file_id: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Delete a file and remove from vector database"""
    try:
        from uuid import UUID
        try:
            file_uuid = UUID(file_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file ID provided"
            )
        result = await session.execute(
                select(UploadedFile).where(UploadedFile.id == file_uuid, UploadedFile.user_id == current_user.id))

        db_file = result.scalars().first()
        if not db_file:
            raise HTTPException(
                status_code= status.HTTP_404_NOT_FOUND,
                detail = "File not found"
            )
        # delete from vector db
        get_file_service().delete_file(db_file.file_hash, str(current_user.id))
        # delete from database
        await session.delete(db_file)
        await session.commit()
        return {"message": f"File '{db_file.file_name}' deleted successfully"}
    
    except HTTPException:
        raise  # re-raise the exception

    except Exception as e:
        await session.rollback()  # Rollback any pending transactions in case of error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete the file: {str(e)}"
        )

@router.post("/api/search", response_model=SearchResponse)
async def search_documents(
    query: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Search documents using semantic search"""
    try:
        # check cache
        cached = await get_cached_search(query, current_user.id)
        if cached:
            print('########### cache hit')
            search_record = SearchHistory(
                user_id = current_user.id,
                query = query,
                answer = cached['answer'],
                search_time = datetime.datetime.utcnow()
            )
            session.add(search_record)
            await session.commit()
            await session.refresh(search_record)
            return SearchResponse(
                **cached, search_id=str(search_record.id),
                created_at=search_record.search_time
            )

        # search
        answer, search_results = get_file_service().keyword_search(query, str(current_user.id))
        # Format matches for response
        matches = [
            {
                "file": result["file"],
                "page": result["page"],
                "chunk": result["chunk"],
                "score": result["score"],
                "text": result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"]
            }
            for result in search_results
        ]
        search_record = SearchHistory(
            user_id = current_user.id,
            query = query,
            answer = answer,
            search_time = datetime.datetime.utcnow()
        )
        session.add(search_record)
        await session.commit()
        await session.refresh(search_record)

        # cache results
        response_payload = {
            "query": query,
            "answer": answer,
            "matches": matches,
            "match_count": len(matches)
            }
        print('###### cache miss:: adding to db')
        await cache_search(query, current_user.id, response_payload)
        return SearchResponse(
            query=query, answer=answer, search_id=str(search_record.id),
            created_at=search_record.search_time, matches=matches,
            match_count=len(matches)
        )
    except HTTPException:
        raise
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
    limit: int = 20
):
    """Get search history for current user"""
    try:
        """
         result = await session.execute(
                select(UploadedFile).where(UploadedFile.id == file_uuid, UploadedFile.user_id == current_user.id))

        db_file = result.scalars().first()
        """
        search_result = await session.execute(
            select(SearchHistory).where(SearchHistory.user_id == current_user.id)
            .order_by(SearchHistory.search_time.desc()).limit(limit))
        _history = search_result.scalars().all()
        return {
            "searches": [
                {
                    "search_id": str(h.id),
                    "query": h.query,
                    "answer": h.answer[:200] + "..." if len(h.answer) > 200 else h.answer,  # Preview
                    "created_at": h.search_time.isoformat()
                }
                for h in _history
            ],
            "total": len(_history),
            "limit": limit
        }
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"get Search history failed: {str(e)}"
        )


@router.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Document Assistant API"}
