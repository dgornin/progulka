from datetime import timedelta

from fastapi import Response
from starlette.requests import Request

from app import auth
from app.db import utcnow


async def test_password_hash_is_salted_and_verifies():
    first = await auth.hash_password("correct horse")
    second = await auth.hash_password("correct horse")

    assert first != second  # a fresh salt every time
    assert first.startswith(f"scrypt${auth.SCRYPT_N}${auth.SCRYPT_R}${auth.SCRYPT_P}$")
    assert "correct horse" not in first
    assert await auth.verify_password("correct horse", first)
    assert not await auth.verify_password("correct horsE", first)


async def test_the_cost_can_be_raised_without_locking_anyone_out(monkeypatch):
    stored = await auth.hash_password("пароль-пароль")
    monkeypatch.setattr(auth, "SCRYPT_N", 2**15)  # above OpenSSL's default memory cap

    raised = await auth.hash_password("пароль-пароль")

    assert raised.startswith(f"scrypt${2**15}$")
    assert await auth.verify_password("пароль-пароль", raised)
    assert await auth.verify_password("пароль-пароль", stored)  # old hashes keep working


async def test_authenticate_checks_email_case_insensitively(store):
    await store.create_user("Anna@Example.com", await auth.hash_password("secret-123"), "Анна")

    assert (await auth.authenticate(store, "anna@example.com", "secret-123")).name == "Анна"
    assert await auth.authenticate(store, "anna@example.com", "wrong-pass") is None
    assert await auth.authenticate(store, "nobody@example.com", "secret-123") is None


def _request(cookie: str | None = None) -> Request:
    headers = [(b"cookie", f"{auth.COOKIE_NAME}={cookie}".encode())] if cookie else []
    return Request({"type": "http", "headers": headers})


async def test_session_cookie_holds_a_token_and_the_database_only_its_hash(store):
    user = await store.create_user("a@example.com", "x", "А")
    response = Response()

    await auth.start_session(store, response, user.id, secure=True)

    cookie = response.headers["set-cookie"]
    token = cookie.split(";")[0].split("=", 1)[1]
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Secure" in cookie
    assert f"Max-Age={30 * 24 * 3600}" in cookie
    assert await store.session_user(token) is None  # the raw token is not a key
    assert (await store.session_user(auth.token_hash(token))).id == user.id


async def test_logout_revokes_the_session(store):
    user = await store.create_user("a@example.com", "x", "А")
    response = Response()
    await auth.start_session(store, response, user.id, secure=False)
    token = response.headers["set-cookie"].split(";")[0].split("=", 1)[1]

    out = Response()
    await auth.end_session(store, _request(token), out, secure=False)

    assert await store.session_user(auth.token_hash(token)) is None
    assert 'progulka_session=""' in out.headers["set-cookie"]


async def test_expired_session_does_not_sign_in(store):
    user = await store.create_user("a@example.com", "x", "А")
    await store.create_session(user.id, "old", utcnow() - timedelta(seconds=1))
    await store.create_session(user.id, "new", utcnow() + timedelta(days=1))

    assert await store.session_user("old") is None
    assert await store.session_user("new") is not None


async def test_seeded_account_gets_the_password_from_env(store):
    await auth.seed_account(store, "demo@progulka.app", "first-pass", "Демо", is_admin=False)
    user, _ = await store.credentials("demo@progulka.app")
    assert user.onboarded and not user.is_admin

    await auth.seed_account(store, "demo@progulka.app", "second-pass", "Демо", is_admin=True)

    assert await auth.authenticate(store, "demo@progulka.app", "first-pass") is None
    rotated = await auth.authenticate(store, "demo@progulka.app", "second-pass")
    assert rotated.id == user.id and rotated.is_admin
