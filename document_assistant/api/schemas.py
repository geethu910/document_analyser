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
    file_name: str
    file_size: int
    upload_time: datetime
    status: str
    message: str


class SearchResult(BaseModel):
    """Single search result"""
    file: str
    page: Optional[int]
    chunk: int
    score: float
    text: str


class SearchResponse(BaseModel):
    """Search query response"""
    query: str
    results: List[SearchResult]
    answer: str
    total_results: int


class FileInfo(BaseModel):
    """File information"""
    file_name: str
    upload_time: datetime
    file_size: int
    status: str


class UserFilesResponse(BaseModel):
    """List of files uploaded by user"""
    files: List[FileInfo]
    total_files: int
