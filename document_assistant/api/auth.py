"""
Authentication module using FastAPI-Users (async)
"""
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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
from document_assistant import logger

load_dotenv()

# Security constants
SECRET_KEY = os.getenv('SECRET_KEY', '') # used to sign jwt tokens and verification/ reset tokens. must be random secret
# can be created using:: python -c "import secrets; print(secrets.token_urlsafe(32))" or openssl rand -hex 32
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com") # smtp server hostname used to send emails.
# eg: smtp.gmail.com for Gmail, or your own SMTP server if you have one. This is used for sending verification emails and password reset emails to users.

SMTP_PORT = int(os.getenv("SMTP_PORT", 587)) # SMTP port to connect to
# common values: 587 for STARTTLS (), 465 for SMTPS, 25 for plain SMTP
SMTP_USER = os.getenv("SMTP_USER", "")  # the username/login for the SMTP account, often an email address or API user name from your mail provider
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # the password or API key for the SMTP account
EMAIL_FROM = os.getenv("EMAIL_FROM", "no-reply@example.com")  # the sender address shown in verification emails
# should be a valid address you control, e.g. no-reply@yourdomain.com


async def send_verification_email(email: str, token: str, request: Request):
    """Send verification email with token link"""
    logger.info('Preparing to send verification email...')
    # verification_link = f"{request.base_url}api/auth/verify?token={token}"
    # Link goes to /verify (static file), not /api/auth/verify (API endpoint)
    base_url = os.getenv("APP_BASE_URL", str(request.base_url))
    verification_link = f"{base_url}verify?token={token}"
    message = MIMEMultipart("alternative")
    message["Subject"] = "Verify your email"
    message["From"] = EMAIL_FROM
    message["To"] = email
    
    text = f"Click the link to verify your account: {verification_link}"
    html = f"""\
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h2>Email Verification</h2>
        <p>Click the link below to verify your account:</p>
        <a href="{verification_link}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">
          Verify Email
        </a>
        <p>Or copy and paste this link in your browser:</p>
        <p>{verification_link}</p>
        <p style="color: #666; font-size: 12px;">This link expires in 24 hours.</p>
      </body>
    </html>
    """
    
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, email, message.as_string())
        logger.info("Verification email sent to the user")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """User manager for FastAPI-Users with UUID support"""
    reset_password_token_secret = SECRET_KEY
    verification_token_secret = SECRET_KEY

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        logger.info(f"User registered")

    async def on_after_forgot_password(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} forgot their password.")

    async def on_after_verify(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} verified.")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info(f"Verification requested for user {user.id}")
        await send_verification_email(user.email, token, request)

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
gtk1@mailinator.com/ password1
"""

# ooauth2 flow::
"""
User clicks “Login with Google”
Redirect to Google
User logs in + consents
Google sends back:authorization code
Your backend sends this code to Google’s token endpoint
Google returns:
    access token
    refresh token (optional)
    ID token (JWT) (if using OpenID Connect)
"""