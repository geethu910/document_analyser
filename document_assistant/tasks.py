import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from celery import Celery
from celery.exceptions import Ignore
from document_assistant.services.file_service import FileService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from document_assistant.api.database import UploadedFile
from document_assistant import logger


DB_URL = (
    f"postgresql://"
    f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 5432)}"
    f"/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)
session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)

redis_url = f'redis://{os.getenv("REDIS_HOST", "localhost")}:{os.getenv("REDIS_PORT", 6379)}'

app = Celery(
    "worker",
    broker=f"{redis_url}/0",
    backend=f"{redis_url}/1"
)
# Celery configuration for task reliability(Celery worker + broker/backend defaults.)
app.conf.update(
    # task_acks_late: ensures that tasks are acknowledged only after they have been executed, 
    # if worker crashes mid-task, the task goes back to the queue instead of being lost.
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # ensures that each worker fetches only one task at a time, reducing the chance of losing tasks if a worker crashes.
    task_reject_on_worker_lost=True,  # ensures that if a worker crashes while processing a task, the task is re-queued and not lost.
    broker_transport_options={'visibility_timeout': 3600},  # If a worker doesn’t finish and acknowledge within 3600 seconds, the task is put back into the queue.
    result_expires=3600,  # results(from redis backend) will be deleted after 3600 seconds, preventing the backend from growing indefinitely.
)
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', '')

# File search lazy loading (singleton)
_file_search_instance: None = None
_file_service: None = None
_gcs_bucket = None

def get_gcs_bucket():
    """Lazy load the GCS bucket handle used to fetch raw uploads handed off by the API"""
    global _gcs_bucket
    if _gcs_bucket is None:
        from google.cloud import storage
        _gcs_bucket = storage.Client().bucket(GCS_BUCKET_NAME)
    return _gcs_bucket


def _download_from_gcs(file_uri: str) -> str:
    """Download a gs:// upload to a local temp file and return its path"""
    blob_name = urlparse(file_uri).path.lstrip('/')
    suffix = Path(blob_name).suffix
    fd, local_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    get_gcs_bucket().blob(blob_name).download_to_filename(local_path)
    return local_path


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


# classify errors to determine if they are transient (retryable) or permanent (not retryable)
class TransientProcessingError(Exception):
    """Network/timeout/rate-limit type errors — safe to retry."""
    pass

class PermanentProcessingError(Exception):
    """Corrupt file, unsupported format — retrying won't help."""
    pass


@app.task(
    # per-task retry/error handling rules.
    bind=True,  # allows the task to access its own state and request information
    max_retries=3,
    retry_backoff=True, # exponential backoff for retries instead of being constant: ~1s, 2s, 4s...
    retry_backoff_max=300,  # maximum retry delay of 5 minutes (300 seconds)
    retry_jitter=True,  # instead of exactly 4s delay, it might be 3.7s or 4.3s delay
    autoretry_for=(TransientProcessingError,),
)
def process_file_task(self,
    file_uri: str,
    user_id: str,
    file_hash: str,
    original_file_name: str
):
    db = session_local()
    is_final_attempt = self.request.retries >= self.max_retries
    temp_file_path = None
    try:
        start = time.perf_counter()
        logger.info("Celery task start: process_file task user=%s file=%s", user_id, original_file_name)

        # idempotency guard: don't reprocess an already-indexed file
        precheck = db.query(UploadedFile).filter(
            UploadedFile.file_hash == file_hash, UploadedFile.user_id == user_id
        ).first()
        if precheck and precheck.status == "indexed":
             logger.info("File %s already indexed, skipping (redelivered task)", original_file_name)
             return
        try:
            temp_file_path = _download_from_gcs(file_uri)
            chunk_count = get_file_service().upload_file(
                temp_file_path,
                user_id=user_id,
                file_hash=file_hash,
                original_file_name=original_file_name
            )
        except (ConnectionError, TimeoutError) as exc:
            # retry incase of network error or something
            raise TransientProcessingError(str(exc)) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            # corrupt/unsupported file -> don't retry
            raise PermanentProcessingError(str(exc)) from exc

        # returns a persistent ORM object that is alredy attached to the session, so we can update it directly and commit once at the end
        # only use db.add() for new records
        file_record = db.query(UploadedFile).filter(
            UploadedFile.file_hash==file_hash, UploadedFile.user_id==user_id).first()
        if not file_record:
            raise ValueError("File record not found")
        file_record.status = "indexed"
        file_record.vector_count = chunk_count
        db.commit()
        _delete_from_gcs(file_uri)
        elapsed = time.perf_counter() - start
        logger.info("File processing completed for %s hash=%s chunks=%d time=%.3fs", original_file_name, file_hash, chunk_count, elapsed)

    except PermanentProcessingError as e:
        db.rollback()  # clear failed commit + dirty session state
        _mark_failed(db, file_hash, user_id)
        db.commit()
        logger.error("Permanent failure for %s: %s", original_file_name, e, exc_info=True)
        # don't retry — stop here, no backend result will be stored, and the task will be marked as failed in the Celery backend.
        raise Ignore()

    except TransientProcessingError as e:
        db.rollback()
        if is_final_attempt:
            _mark_failed(db, file_hash, user_id)
            db.commit()
            logger.error("Giving up on %s after %d retries", original_file_name, self.request.retries)
        else:
            logger.warning("Transient error on %s, will retry: %s", original_file_name, e)
        raise  # autoretry_for handles the actual retry
    
    except Exception as e:
        db.rollback()
        _mark_failed(db, file_hash, user_id)
        db.commit()
        logger.error("Error processing file task user=%s file=%s", user_id, original_file_name, exc_info=True)
        raise  # propagate the exception to Celery, which will mark the task as failed and store the exception in the backend for inspection.
    finally:
        db.close()
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def _delete_from_gcs(file_uri: str) -> None:
    """Remove the raw upload from GCS once it's been indexed — no further use for it"""
    blob_name = urlparse(file_uri).path.lstrip('/')
    get_gcs_bucket().blob(blob_name).delete()


def _mark_failed(db, file_hash, user_id):
    failed_record = db.query(UploadedFile).filter(
        UploadedFile.file_hash == file_hash,
        UploadedFile.user_id == user_id
    ).first()
    if failed_record:
        failed_record.status = "failed"
