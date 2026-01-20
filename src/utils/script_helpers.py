from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional


def parse_device(value: object) -> object:
    if isinstance(value, int):
        return value
    if value is None:
        return "auto"
    v = str(value).strip()
    if v.lower() in {"auto", "mps"}:
        return v.lower()
    try:
        return int(v)
    except Exception:
        return v


def parse_top_k(value: Optional[str | int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    v = str(value).strip().lower()
    if v in {"all", "none"}:
        return None
    return int(v)


def normalize_splits(splits: Iterable[str] | str) -> list[str]:
    if isinstance(splits, str):
        return [s.strip() for s in splits.split(",") if s.strip()]
    return [str(s).strip() for s in splits if str(s).strip()]


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
