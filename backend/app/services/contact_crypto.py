"""Encrypt emergency phone numbers before storing them in MongoDB."""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from ..config import settings


def _fernet() -> Fernet:
    key = settings.emergency_contact_encryption_key
    if not key:
        raise RuntimeError("EMERGENCY_CONTACT_ENCRYPTION_KEY is not configured; generate a Fernet key and add it to the project-root .env file.")
    try:
        return Fernet(key.encode("ascii"))
    except Exception as exc:
        raise RuntimeError("EMERGENCY_CONTACT_ENCRYPTION_KEY is invalid; generate a valid Fernet key.") from exc


def encrypt_value(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_value(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError("Emergency contact cannot be decrypted; verify the encryption key matches the key used to save it.") from exc


def encrypt_phone(phone: str) -> str:
    return encrypt_value(phone)


def decrypt_phone(token: str) -> str:
    return decrypt_value(token)
