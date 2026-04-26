import re
from typing import Optional

from fastapi import HTTPException

SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")


def validate_safe_id(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} is required")
    if not SAFE_IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(
            f"Invalid {name}. Allowed characters: letters, numbers, dot, underscore, colon, hyphen."
        )
    return normalized


def validate_optional_safe_id(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    return validate_safe_id(value, name)


def require_safe_id_or_400(value: str, name: str) -> str:
    try:
        return validate_safe_id(value, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
