"""Accounts: password hashing, login sessions and the FastAPI dependencies that use them.

Passwords are hashed with scrypt from the standard library (hashlib.scrypt): a
memory-hard KDF recommended for password storage, so no bcrypt/argon2 dependency is
needed. Each hash carries its own random salt and parameters (`scrypt$n$r$p$salt$key`),
so the cost can be raised later without invalidating old hashes. Hashing takes tens
of milliseconds of CPU, so it runs in a thread and does not stall the event loop that
streams other users' routes.

A session is a random token in an HttpOnly cookie; the database keeps only its
sha256. Unlike a signed JWT, a server-side session can be revoked (logout really
logs out, a password reset signs everyone out) and there is no signing key to manage.
CSRF: SameSite=Lax keeps the cookie off cross-site POST/PUT/DELETE, and every change
is a JSON POST/PUT or a DELETE, which another origin cannot send without a CORS
preflight this app never grants (that also covers a sibling subdomain, which counts as
the same site). Only logout is a bare POST, and a forced logout is harmless.
"""

import asyncio
import hashlib
import hmac
import secrets
from datetime import timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response

from app.db import utcnow
from app.profile import Profile
from app.store import Store, UserOut

COOKIE_NAME = "progulka_session"
SESSION_TTL = timedelta(days=30)
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    # scrypt needs about 128·r·(n + p) bytes; OpenSSL's default cap (32 MiB) would
    # refuse n = 2**15, so the cap follows the parameters and the cost can be raised.
    return hashlib.scrypt(
        password.encode(), salt=salt, n=n, r=r, p=p, maxmem=256 * r * (n + p), dklen=32
    )


async def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = await asyncio.to_thread(_derive, password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${key.hex()}"


async def verify_password(password: str, stored: str) -> bool:
    _scheme, n, r, p, salt, key = stored.split("$")
    derived = await asyncio.to_thread(
        _derive, password, bytes.fromhex(salt), int(n), int(r), int(p)
    )
    return hmac.compare_digest(derived, bytes.fromhex(key))


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def authenticate(store: Store, email: str, password: str) -> UserOut | None:
    found = await store.credentials(email)
    if found is None:
        return None
    user, stored = found
    return user if await verify_password(password, stored) else None


async def start_session(store: Store, response: Response, user_id: int, secure: bool) -> None:
    token = secrets.token_urlsafe(32)
    await store.create_session(user_id, token_hash(token), utcnow() + SESSION_TTL)
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


async def end_session(store: Store, request: Request, response: Response, secure: bool) -> None:
    if token := request.cookies.get(COOKIE_NAME):
        await store.delete_session(token_hash(token))
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=secure)


async def seed_account(
    store: Store,
    email: str,
    password: str,
    name: str,
    *,
    is_admin: bool,
    profile: Profile | None = None,
) -> None:
    """Make sure an operator-configured account exists with the password from .env.

    Created when missing (already onboarded, so it skips the questionnaire); when the
    password in .env no longer matches, the stored one is reset to it. That is how an
    operator rotates the demo and admin passwords without a password-change form.
    """
    found = await store.credentials(email)
    if found is None:
        await store.create_user(
            email,
            await hash_password(password),
            name,
            is_admin=is_admin,
            onboarded=True,
            profile=profile,
        )
        return
    user, stored = found
    new_hash = None if await verify_password(password, stored) else await hash_password(password)
    if new_hash or user.is_admin != is_admin:
        await store.update_user(user.id, is_admin=is_admin, password_hash=new_hash)


def get_store(request: Request) -> Store:
    return request.app.state.store


StoreDep = Annotated[Store, Depends(get_store)]


async def current_user(request: Request, store: StoreDep) -> UserOut | None:
    token = request.cookies.get(COOKIE_NAME)
    return await store.session_user(token_hash(token)) if token else None


async def require_user(user: Annotated[UserOut | None, Depends(current_user)]) -> UserOut:
    if user is None:
        raise HTTPException(status_code=401, detail="Нужно войти")
    return user


UserDep = Annotated[UserOut, Depends(require_user)]


async def require_admin(user: UserDep) -> UserOut:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Только для администратора")
    return user


AdminDep = Annotated[UserOut, Depends(require_admin)]
