"""Identifiers used by HTTP and filesystem-backed storage."""
import re
from pathlib import Path

_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

def validate_id(value: str) -> str:
    if not isinstance(value, str) or not _PATTERN.fullmatch(value) or value.upper() in _RESERVED:
        raise ValueError("ID must be 1-64 ASCII letters, digits, underscores or hyphens; reserved names are forbidden")
    return value

def safe_path(root: Path, *parts: str) -> Path:
    base = root.resolve()
    target = base.joinpath(*parts).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Storage path escapes its root")
    return target
