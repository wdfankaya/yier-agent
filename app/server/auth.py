"""HTTP identity comes from a server-issued bearer token, never the request body."""
import secrets
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.config.settings import settings
from app.server.identifiers import validate_id

_bearer = HTTPBearer(auto_error=False)

def authenticated_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    tokens = settings.api_tokens
    if not tokens:
        raise HTTPException(503, "Configure API_TOKENS before using the HTTP API")
    values = list(tokens.values())
    if len(values) != len(set(values)) or any(len(v) < 32 for v in values):
        raise HTTPException(503, "API_TOKENS must contain unique tokens of at least 32 characters")
    if credentials is not None and credentials.scheme.lower() == "bearer":
        for user, token in tokens.items():
            if secrets.compare_digest(credentials.credentials.encode(), token.encode()):
                try:
                    return validate_id(user)
                except ValueError:
                    raise HTTPException(503, "Invalid configured user ID") from None
    raise HTTPException(401, "Invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})

def check_claim(claim: str | None, user: str) -> None:
    if claim is not None and claim != user:
        raise HTTPException(403, "user_id does not match authenticated identity")
