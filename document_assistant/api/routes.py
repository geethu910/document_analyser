"""
FastAPI routes for document assistant API using FastAPI-Users (async)
"""
import asyncio
import datetime
import time
import json
import hashlib
import os
from pathlib import Path
import tempfile
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import StreamingResponse
from fastapi import Request
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
from document_assistant.tasks import process_file_task
from document_assistant import logger

BASE_UPLOAD_DIRECTORY = '/app/uploads'
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
        logger.debug("Lazy-loading FileSearch instance")
        start = time.perf_counter()
        from document_assistant.file_search_embeddings import FileSearch
        _file_search_instance = FileSearch()
        logger.info("FileSearch initialized in %.3fs", time.perf_counter() - start)
    return _file_search_instance

def get_file_service() -> FileService:
    """Get FileService singleton"""
    global _file_service
    if _file_service is None:
        logger.debug("Creating FileService singleton")
        start = time.perf_counter()
        from document_assistant.services.file_service import FileService
        _file_service = FileService(get_file_search())
        logger.info("FileService created in %.3fs", time.perf_counter() - start)
    return _file_service

@router.post("/api/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Upload a document file (PDF, DOCX, TXT)"""
    try:
        start = time.perf_counter()
        # get service
        file_service = get_file_service()
        logger.info("Upload requested: user=%s filename=%s", current_user.id, file.filename)
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
        logger.debug("Wrote temp file %s size=%d hash=%s", temp_file_path, total_size, file_hash)
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
                logger.debug(f'File {file.filename} already exists with same hash, skipping re-indexing')
                # SAME FILE: Skip (already indexed)
                return FileUploadResponse( 
                    file_id= str(existing_file.id),
                    file_name = file.filename,
                    file_size = existing_file.file_size,
                    vector_count= existing_file.vector_count,
                    upload_time=existing_file.upload_time,
                    status="File is already uploaded",
                )
            else:
                # different file, same name:: replace
                logger.debug(f'Replacing older version of {file.filename}')
                # delete file from both places
                # delete from vector db
                get_file_service().delete_file(existing_file.file_hash, str(current_user.id))
                # delete from database
                await session.delete(existing_file)
                await session.commit()
            
        # create embeddings and add to vector database
        idx_start = time.perf_counter()
        chunk_count = file_service.upload_file(
            temp_file_path, user_id=str(current_user.id), 
            file_hash=file_hash,
            original_file_name = file.filename
            )  # Process the uploaded file and add to vector database
        logger.info("Indexing complete: user=%s file=%s vectors=%s time=%.3fs", current_user.id, file.filename, chunk_count, time.perf_counter() - idx_start)

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
        elapsed = time.perf_counter() - start
        logger.info("Upload finished: user=%s file=%s size_kb=%.2f vectors=%s total_time=%.3fs", current_user.id, file.filename, file_size_kb, chunk_count, elapsed)
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

    except Exception:
        await session.rollback()  # Rollback any pending transactions in case of error

        logger.error("Error processing file upload user=%s filename=%s", getattr(current_user, 'id', None), getattr(file, 'filename', None), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process file upload. Please try again later."
        )
    finally:
        # clean up temp file
        # This runs regardless of success or failure
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up temp file: {temp_file_path}")
            except OSError as e:
                logger.warning(f" Failed to delete temp file {temp_file_path}", exc_info=True)
                # Log this but don't crash - temp can be cleaned later

@router.get("/api/files/list", response_model=UserFilesResponse)
async def list_user_files(
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """List all files uploaded by current user"""
    try:
        start = time.perf_counter()
        result = await session.execute(
            select(UploadedFile).where(UploadedFile.user_id == current_user.id))
        files = result.scalars().all()
        total_size_kb = sum(f.file_size for f in files)

        elapsed = time.perf_counter() - start
        logger.info("List files: user=%s total_files=%s total_kb=%.2f time=%.3fs", current_user.id, len(files), round(total_size_kb, 2), elapsed)

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
        logger.error("Error retrieving user files", exc_info=True)
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unable to process file retrieval. Please try again later."
            )

@router.delete("/api/files/{file_id}")
async def delete_file(
    file_id: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Delete a file and remove from vector database"""
    try:
        start = time.perf_counter()
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

        elapsed = time.perf_counter() - start
        logger.info("Deleted file: user=%s file=%s time=%.3fs", current_user.id, db_file.file_name, elapsed)
        return {"message": f"File '{db_file.file_name}' deleted successfully"}

    except HTTPException:
        raise  # re-raise the exception

    except Exception:
        await session.rollback()  # Rollback any pending transactions in case of error
        logger.error("Error deleting file", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to delete file. Please try again later."
        )

@router.post("/api/search", response_model=SearchResponse)
async def search_documents(
    query: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Search documents using semantic search"""
    try:
        start = time.perf_counter()
        # check cache
        cached = await get_cached_search(query, current_user.id)
        if cached:
            logger.debug('cache hit')
            search_record = SearchHistory(
                user_id = current_user.id,
                query = query,
                answer = cached['answer'],
                search_time = datetime.datetime.utcnow()
            )
            session.add(search_record)
            await session.commit()
            await session.refresh(search_record)
            elapsed = time.perf_counter() - start
            logger.info("Search cache hit: user=%s query_len=%d time=%.3fs", current_user.id, len(query), elapsed)
            return SearchResponse(
                **cached, search_id=str(search_record.id),
                created_at=search_record.search_time
            )

        # cache miss — perform search
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
        logger.debug('cache miss:: adding to db')
        await cache_search(query, current_user.id, response_payload)
        elapsed = time.perf_counter() - start
        logger.info("Search: user=%s query_len=%d matches=%d time=%.3fs cache_miss=True", current_user.id, len(query), len(matches), elapsed)
        return SearchResponse(
            query=query, answer=answer, search_id=str(search_record.id),
            created_at=search_record.search_time, matches=matches,
            match_count=len(matches)
        )
    except HTTPException:
        raise
    except Exception:
        logger.error("Error during search", exc_info=True)
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed: Unable to process your request. Please try again later."
        )

@router.get("/api/search/history")
async def get_search_history(
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
    limit: int = 20
):
    """Get search history for current user"""
    try:
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
        logger.error("Error retrieving search history", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process your request. Please try again later."
        )


@router.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Document Assistant API"}

@router.post("/api/hybrid_search", response_model=SearchResponse)
async def hybrid_search(
    query: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Search documents using semantic search"""
    try:
        start = time.perf_counter()
        # check cache
        cached = await get_cached_search(query, current_user.id)
        if cached:
            logger.debug('########### cache hit')
            search_record = SearchHistory(
                user_id = current_user.id,
                query = query,
                answer = cached['answer'],
                search_time = datetime.datetime.utcnow()
            )
            session.add(search_record)
            await session.commit()
            await session.refresh(search_record)
            elapsed = time.perf_counter() - start
            logger.info("Hybrid search cache hit: user=%s query_len=%d time=%.3fs", current_user.id, len(query), elapsed)
            return SearchResponse(
                **cached, search_id=str(search_record.id),
                created_at=search_record.search_time
            )

        # cache miss — perform hybrid search
        answer, search_results = get_file_service().hybrid_search(query, str(current_user.id))
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
        logger.debug(f'search results retrieved at {datetime.datetime.utcnow().isoformat()} with {len(matches)} matches')
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
        logger.debug('cache miss:: adding to db')
        await cache_search(query, current_user.id, response_payload)
        elapsed = time.perf_counter() - start
        logger.info("Hybrid search: user=%s query_len=%d matches=%d time=%.3fs cache_miss=True", current_user.id, len(query), len(matches), elapsed)
        return SearchResponse(
            query=query, answer=answer, search_id=str(search_record.id),
            created_at=search_record.search_time, matches=matches,
            match_count=len(matches)
        )
    except HTTPException:
        raise
    except Exception:
        await session.rollback()
        logger.error("Error during hybrid search", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process your request. Please try again later."
        )

@router.post("/api/search/stream")
async def stream_search(
    request: Request,
    query: str,
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Hybrid search with SSE streaming"""

    async def event_generator():
        tokens = []
        start = time.perf_counter()
        try:
            # Step 1 — check cache first
            cached = await get_cached_search(query, current_user.id)
            if cached:
                # cache hit — send everything at once (already fast)
                yield f"event: matches\ndata: {json.dumps(cached['matches'])}\n\n"
                yield f"event: token\ndata: {json.dumps(cached['answer'])}\n\n"
                yield f"event: done\ndata: {json.dumps({'search_id': None, 'match_count': cached['match_count'], 'cached': True})}\n\n"
                elapsed = time.perf_counter() - start
                logger.info("Stream search cache hit: user=%s query_len=%d time=%.3fs", current_user.id, len(query), elapsed)
                return

            # Step 2 — retrieve chunks (fast ~10ms)
            search_results = get_file_service().hybrid_retrieve(
                query, str(current_user.id)
            )
            matches = [
                {
                    "file": r["file"],
                    "page": r["page"],
                    "chunk": r["chunk"],
                    "score": r["score"],
                    "text": r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
                }
                for r in search_results
            ]

            # Send matches immediately — frontend renders sources right away
            yield f"event: matches\ndata: {json.dumps(matches)}\n\n"

            # Step 3 — stream tokens from Groq
            if not search_results:
                yield f"event: token\ndata: {json.dumps('Not found in the provided documents.')}\n\n"
                tokens.append("Not found in the provided documents.")
            else:
                async for token in get_file_service().hybrid_generate_stream(
                    query, search_results
                ):
                    if await request.is_disconnected():
                        logger.info("Client disconnected, stopping stream")
                        break
                    if token is None:
                        continue  # skip any empty tokens
                    tokens.append(token)
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            # Step 4 — save to DB + cache after streaming completes
            search_record = SearchHistory(
                user_id=current_user.id,
                query=query,
                answer=''.join(tokens),
                search_time=datetime.datetime.utcnow()
            )
            session.add(search_record)
            await session.commit()
            await session.refresh(search_record)

            response_payload = {
                "query": query,
                "answer": ''.join(tokens),
                "matches": matches,
                "match_count": len(matches)
            }
            await cache_search(query, current_user.id, response_payload)

            # Step 5 — send completion signal with search_id
            elapsed = time.perf_counter() - start
            logger.info("Stream search completed: user=%s query_len=%d matches=%d time=%.3fs", current_user.id, len(query), len(matches), elapsed)
            yield f"event: done\ndata: {json.dumps({'search_id': str(search_record.id), 'match_count': len(matches), 'cached': False})}\n\n"
        except asyncio.CancelledError:
            logger.info("Search cancelled by client")
            raise
        except Exception:
            await session.rollback()
            logger.error("Error during hybrid search", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'detail': 'Internal server error'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

@router.post("/api/files/hybrid_upload", response_model=FileUploadResponse)
async def hybrid_upload(
    file: UploadFile = File(...),
    current_user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Upload a document file (PDF, DOCX, TXT)"""
    try:
        start = time.perf_counter()
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
        base_dir = os.path.join(BASE_UPLOAD_DIRECTORY, str(current_user.id))
        os.makedirs(base_dir, exist_ok=True)
        temp_path = os.path.join(base_dir, f"{uuid.uuid4()}.tmp")
        # Stream file in chunks (4KB at a time)
        hasher = hashlib.sha256()
        chunk_size = 4096
        total_size = 0

        with open(temp_path, "wb") as out_file:
            while True:
                # read a chunks untill eof, await is used to ensure that the event loop can handle other requests while waiting for file I/O
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                hasher.update(chunk)
                total_size += len(chunk)

                if total_size > 50 * 1024 * 1024:  # 50MB limit
                    out_file.close()  # Close the file before deleting
                    os.unlink(temp_path)  # Delete temp file if too large
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Max: 50MB. Got: {total_size / (1024 * 1024):.1f}MB"
                    )

        file_hash = hasher.hexdigest()
        final_path = os.path.join(base_dir, f"{file_hash}{file_ext}")
        os.replace(temp_path, final_path)  # Move temp file to final location with hash
     
        # step 4: check if file is already loaded for the current user
        existing = await session.execute(
            select(UploadedFile).where(
                UploadedFile.file_name == file.filename,
                UploadedFile.user_id == current_user.id,
                UploadedFile.status.in_(['indexed', 'processing'])
        ))  
        existing_file = existing.scalars().first() # here scalars() is used to extract the UploadedFile objects from the result, and first() retrieves the first one (or None if there are no matches). This allows us to check if a file with the same hash has already been uploaded by the current user, which helps prevent duplicate entries in the database and redundant processing in the vector database.
        if existing_file:
            if existing_file.file_hash == file_hash:
                logger.info(f'File {file.filename} already exists with same hash, skipping re-indexing')
                # SAME FILE: Skip (already indexed)
                return FileUploadResponse( 
                    file_id= str(existing_file.id),
                    file_name = file.filename,
                    file_size = existing_file.file_size,
                    vector_count= existing_file.vector_count,
                    upload_time=existing_file.upload_time,
                    status= "File is already uploaded"

                )
            else:
                # different file, same name:: replace
                logger.info(f'Replacing older version of {file.filename}')
                # delete file from both places
                # delete from vector db
                get_file_service().delete_file(existing_file.file_hash, str(current_user.id))
                # delete from database
                await session.delete(existing_file)
                await session.commit()
     
        # save file metadata to database
        new_file = UploadedFile(
            user_id=current_user.id,
            file_name=file.filename,
            file_hash=file_hash,
            file_size=total_size,
            file_type=file_ext[1:],  # remove dot from extension
            status="processing",
            vector_count= 0
        )
        session.add(new_file)
        await session.commit()  # Commit the transaction to save the file metadata in the database
        await session.refresh(new_file)

        # Process the uploaded file and add to vector database as a background task using celery.
        task = process_file_task.delay(
            temp_file_path=final_path,  # passing persistant file path 
            user_id=str(current_user.id),
            file_hash=file_hash,
            original_file_name=file.filename
        )
        # TODO: may also add task.id to database
        # increment cache version of user inorder to exclude stale versions
        await invalidate_user_cache(current_user.id)
        elapsed = time.perf_counter() - start
        logger.info("Hybrid upload queued: user=%s file=%s size_kb=%.2f time=%.3fs", current_user.id, file.filename, total_size / 1024, elapsed)
        return FileUploadResponse(
            file_id=str(new_file.id),
            file_name=new_file.file_name,
            file_size=new_file.file_size,
            vector_count=0,
            status=new_file.status,
            upload_time=new_file.upload_time.isoformat()
            )
        
    except HTTPException:
        raise  # re-raise the exception

    except Exception as e:
        await session.rollback()  # Rollback any pending transactions in case of error
        for path in [temp_path, final_path]:    # clean up files
            if path and os.path.exists(path):
                os.unlink(path)
        logger.error("Error processing file upload", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process your request. Please try again later."
        )
