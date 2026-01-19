from __future__ import annotations

from pathlib import Path


def require_file(path: str | Path, hint: str = "") -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file() or p.stat().st_size == 0:
        msg = f"Missing file: {p}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)
    return p


def require_dir(path: str | Path, hint: str = "") -> Path:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        msg = f"Missing directory: {p}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)
    return p
