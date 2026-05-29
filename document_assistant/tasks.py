import os
from celery import Celery
from document_assistant.services.file_service import FileService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from document_assistant.api.database import UploadedFile
from document_assistant import logger
import time

DB_URL = os.getenv('SYNC_DATABASE_URL', 'postgresql://doc_analyser_user:doc_analyser_password@localhost:5432/doc_analyser_db')

engine = create_engine(DB_URL)
SessionLocal = sessionmaker( bind=engine, autocommit=False, autoflush=False )
app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)
# File search lazy loading (singleton)
_file_search_instance: None = None
_file_service: None = None

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

@app.task
def process_file_task(
    temp_file_path: str,
    user_id: str,
    file_hash: str,
    original_file_name: str
):  
    db = SessionLocal()
    try:
        start = time.perf_counter()
        logger.info("Celery task start: process_file task user=%s file=%s", user_id, original_file_name)
        chunk_count = get_file_service().upload_file(
            temp_file_path,
            user_id=user_id,
            file_hash=file_hash,
            original_file_name=original_file_name
        )
        # returns a persistent ORM object that is alredy attached to the session, so we can update it directly and commit once at the end
        # only use db.add() for new records
        file_record = db.query(UploadedFile).filter(
            UploadedFile.file_hash==file_hash, UploadedFile.user_id==user_id).first()
        if not file_record:
            raise ValueError("File record not found")
        file_record.status = "indexed"
        file_record.vector_count = chunk_count
        db.commit()
        elapsed = time.perf_counter() - start
        logger.info("File processing completed for %s hash=%s chunks=%d time=%.3fs", original_file_name, file_hash, chunk_count, elapsed)
    except Exception as e:
        db.rollback()
        failed_record = db.query(UploadedFile).filter(
            UploadedFile.file_hash == file_hash, 
            UploadedFile.user_id == user_id).first()
        if failed_record:
            failed_record.status = "failed"
            db.commit()
        logger.error("Error processing file task user=%s file=%s", user_id, original_file_name, exc_info=True)
        raise
    finally:
        db.close()
