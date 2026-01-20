from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def write_progress(progress_path: Optional[Path], payload: Any) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

