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
    email: EmailStr


class UserUpdate(schemas.BaseUserUpdate):
    """User update schema"""
    email: EmailStr


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
    score: float
    text: str
    text_full: Optional[str] = None  # Untruncated chunk text (used for evaluation/grounding)

class SearchResponse(BaseModel):
    """Search query response"""
    query: str
    answer: str
    search_id: str
    created_at: datetime
    matches: List[SearchMatch] = []  # ✅ ADD: Top matching chunks
    match_count: int = 0  # ✅ ADD: Number of relevant matches
    conversation_id: Optional[str] = None  # Set when the search belongs to a multi-turn conversation


class ConversationCreate(BaseModel):
    """Request body to start a new conversation"""
    title: Optional[str] = None


class ConversationRead(BaseModel):
    """Conversation summary (list/create response)"""
    conversation_id: str
    title: Optional[str] = None
    created_at: datetime


class ConversationTurn(BaseModel):
    """A single message within a conversation"""
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime


class ConversationDetail(BaseModel):
    """Full conversation with its ordered turns"""
    conversation_id: str
    title: Optional[str] = None
    created_at: datetime
    turns: List[ConversationTurn] = []


class ConversationListResponse(BaseModel):
    """List of a user's conversations"""
    conversations: List[ConversationRead] = []
    total: int = 0


class UserFilesResponse(BaseModel):
    """List of files uploaded by user"""
    files: List[FileUploadResponse]
    total_files: int
    total_size_kb: float
