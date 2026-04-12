"""
API Request/Response schemas using Pydantic and FastAPI-Users
"""
import uuid
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from fastapi_users import schemas


class UserRead(schemas.BaseUser[uuid.UUID]):
    """User read schema (response)"""

    class Config:
        from_attributes = True


class UserCreate(schemas.BaseUserCreate):
    """User create schema (request for registration)"""
    pass


class UserUpdate(schemas.BaseUserUpdate):
    """User update schema"""
    pass


# File and Search schemas remain the same
class FileUploadResponse(BaseModel):
    """File upload response"""
    file_id: str
    file_name: str
    file_size: int
    vector_count: int
    status: str
    upload_time: datetime


class SearchResult(BaseModel):
    """Single search result"""
    file: str
    page: Optional[int] = None
    chunk: int
    score: float
    text: str

class SearchMatch(BaseModel):
    """Individual search match result"""
    file: str
    page: Optional[int]
    chunk: int
    score: float  # ✅ ADD: Similarity score
    text: str

class SearchResponse(BaseModel):
    """Search query response"""
    query: str
    answer: str
    search_id: str
    created_at: datetime
    matches: List[SearchMatch] = []  # ✅ ADD: Top matching chunks
    match_count: int = 0  # ✅ ADD: Number of relevant matches


# class FileInfo(BaseModel):
#     """File information"""
#     file_name: str
#     upload_time: datetime
#     file_size: int
#     status: str


class UserFilesResponse(BaseModel):
    """List of files uploaded by user"""
    files: List[FileUploadResponse]
    total_files: int
    total_size_kb: float
