"""
Database models and initialization using FastAPI-Users (async)
"""
import os
from pathlib import Path
import uuid
from uuid import uuid4
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
from fastapi import Depends
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase, SQLAlchemyBaseUserTableUUID


# Database URL - use async driver
# DB_URL = os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///./results/app_db.sqlite')
DB_URL = (
    f"postgresql+asyncpg://"
    f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 5432)}"
    f"/{os.getenv('DB_NAME')}"
)


DB_URL_SYNC = (
    f"postgresql://"
    f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 5432)}"
    f"/{os.getenv('DB_NAME')}"
)
# Create async engine
engine = create_async_engine(
    DB_URL,
    echo=False,
    future=True
)

# Async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    future=True
)

# Base class for models
class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    """User model for FastAPI-Users with UUID"""
    # relationships creates virtual attributes for accessing data, so that user.uploaded_files and user.search_history work seamlessly  
    uploaded_files = relationship("UploadedFile", back_populates="user")
    search_history = relationship("SearchHistory", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class UploadedFile(Base):
    """Track uploaded files per user"""
    __tablename__ = "uploaded_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)  # Use UUID for file ID
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    file_name = Column(String, index=True, nullable=False)  # ✅ USEFUL - you'll search by name
    file_hash = Column(String, nullable=False)  # SHA-256 for deduplication
    # file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="indexed")  # indexed, processing, failed
    file_type = Column(String(20))  # pdf, docx, txt (derived from extension)
    user = relationship("User", back_populates="uploaded_files")
    vector_count = Column(Integer, nullable=True)  # Store total chunks indexed for analytics
    celery_task_id = Column(String(255), nullable=True, index=True)
    __table_args__ = (UniqueConstraint('user_id', 'file_hash', name='uq_user_file_hash'),)
    # vector_count / num_chunks can be added later for analytics


class SearchHistory(Base):
    """Track search queries for analytics"""
    __tablename__ = "search_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)  # Use UUID for search history ID
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    query = Column(String, nullable=False, index=True)
    answer = Column(String, nullable=False)
    search_time = Column(DateTime, default=datetime.utcnow)
    # Nullable so existing single-turn searches keep working; set when the turn belongs to a conversation
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True, index=True)
    user = relationship("User", back_populates="search_history")
    conversation = relationship("Conversation", back_populates="turns")


class Conversation(Base):
    """A multi-turn conversation that groups related search turns (SearchHistory rows)"""
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="conversations")
    turns = relationship(
        "SearchHistory",
        back_populates="conversation",
        order_by="SearchHistory.search_time",
    )


# Async database initialization
async def create_db_and_tables():
    """Create database tables (async)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session"""
    async with async_session_maker() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    """Get FastAPI-Users database session"""
    yield SQLAlchemyUserDatabase(session, User)

