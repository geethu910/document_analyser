"""
Authentication module using FastAPI-Users (async)
"""
import os
import uuid
from typing import Optional
from fastapi import Depends, Request
from fastapi_users import FastAPIUsers
from fastapi_users import BaseUserManager, UUIDIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from dotenv import load_dotenv

# Import User model
from document_assistant.api.database import User, get_user_db

load_dotenv()

# Security constants
SECRET_KEY = os.getenv('SECRET_KEY', 'sakjdhkjad872323')


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """User manager for FastAPI-Users with UUID support"""
    reset_password_token_secret = SECRET_KEY
    verification_token_secret = SECRET_KEY

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has forgot their password.")

    async def on_after_request_verify(self, user: User, request: Optional[Request] = None):
        print(f"Verification requested for user {user.id}.")


# Configure Bearer transport
# This defines how the authentication token is sent in requests. We use the standard Bearer token in the Authorization header, and specify the login URL for obtaining tokens.
bearer_transport = BearerTransport(tokenUrl="api/auth/jwt/login")


# Configure JWT strategy
# This defines how JWT tokens are created and validated. We use a secret key and set a token lifetime of 30 minutes.
def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET_KEY, lifetime_seconds=1800)  # 30 minutes


# Configure authentication backend
# This ties together the transport and strategy to define how authentication works in our app. We name it "jwt" since we're using JWT tokens.
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# Get user manager dependency
# T
async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    """Dependency to get user manager"""
    yield UserManager(user_db)


# Initialize FastAPI-Users instance with UUID (single source of truth)
# This creates the main FastAPI-Users instance that we will use to manage users and authentication. We specify our User model, UUID type, user manager dependency, and authentication backend.
auth_manager = FastAPIUsers[User, uuid.UUID](
    get_user_manager=get_user_manager,
    auth_backends=[auth_backend],
)

# Dependency for getting current active user
current_active_user = auth_manager.current_user(active=True)

# Dependency for getting current superuser
# this is required for any admin-only endpoints. It will check that the user is active and has superuser privileges before allowing access.
current_superuser = auth_manager.current_user(active=True, superuser=True)


"""
users:  "email": "gtk1@gmail.com"/ "password1", "gtk2@gmail.com"/ "password2"

"""