from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from utils.ensure import require_dir, require_file

MOVIE_TAG_RE = re.compile(r"@(\d+)")


def safe_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))[:200] or "unknown"


def get_speaker(msg: Dict[str, Any], idx: int) -> str:
    """
    ReDial messages alternate by index:
      even index -> Seeker, odd index -> Recommender.
    """
    _ = msg
    return "Seeker" if (idx % 2 == 0) else "Recommender"


def extract_movie_ids(text: str) -> List[str]:
    return MOVIE_TAG_RE.findall(text or "")


def replace_movie_tags_with_titles(text: str, mentions_map: Dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        k = m.group(1)
        return mentions_map.get(k, m.group(0))

    return MOVIE_TAG_RE.sub(repl, text or "")


@dataclass(frozen=True)
class ReDialConversation:
    dataset: "ReDialDataset"
    split: str
    conversation_id: str
    messages: List[Dict[str, Any]]
    movie_mentions: Dict[str, str]

    emotion_cache: Any = None
    bias_cache: Any = None

    @property
    def emotion(self) -> Dict[str, Any]:
        if self.emotion_cache is None:
            raise RuntimeError("No emotion_cache attached to this conversation.")
        return self.emotion_cache.get(self)

    @property
    def bias(self) -> Dict[str, Any]:
        if self.bias_cache is None:
            raise RuntimeError("No bias_cache attached to this conversation.")
        return self.bias_cache.get(self)


class ReDialDataset:
    """
    Pure loader for ReDial.

    Expected layout (created by scripts/download_data.py):
      <cache_root>/datasets/redial/data/train_data.jsonl
      <cache_root>/datasets/redial/data/test_data.jsonl
      <cache_root>/datasets/redial/data/movies_with_mentions.csv
    """

    def __init__(self, cache_root: Path | str = "./cache"):
        self.cache_root = Path(cache_root)
        self.root = require_dir(
            self.cache_root / "datasets" / "redial",
            hint="Run: python3 scripts/download_data.py",
        )
        self.data_dir = require_dir(
            self.root / "data",
            hint="Run: python3 scripts/download_data.py",
        )

        # Validate required files early
        _ = require_file(self.path_for_split("train"), hint="Run: python3 scripts/download_data.py")
        _ = require_file(self.path_for_split("test"), hint="Run: python3 scripts/download_data.py")
        _ = require_file(self.mentions_csv_path(), hint="Run: python3 scripts/download_data.py")

        self._mentions_map: Optional[Dict[str, str]] = None

    def path_for_split(self, split: str) -> Path:
        if split == "train":
            return self.data_dir / "train_data.jsonl"
        if split == "test":
            return self.data_dir / "test_data.jsonl"
        raise ValueError("split must be 'train' or 'test'")

    def mentions_csv_path(self) -> Path:
        return self.data_dir / "movies_with_mentions.csv"

    def movie_mentions_map(self) -> Dict[str, str]:
        if self._mentions_map is None:
            self._mentions_map = self._load_mentions_map(self.mentions_csv_path())
        return dict(self._mentions_map)

    def iter(
        self,
        split: str = "train",
        start: int = 0,
        max_convos: Optional[int] = None,
        emotion=None,
        bias=None,
    ) -> Iterator[ReDialConversation]:
        """
        Streams JSONL and yields ReDialConversation objects.
        Attach caches by passing emotion=..., bias=...
        """
        path = require_file(self.path_for_split(split), hint="Run: python3 scripts/download_data.py")

        n = 0
        yielded = 0
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                if n < start:
                    n += 1
                    continue

                obj = json.loads(ln)
                cid = str(obj.get("conversationId", "")) or f"line_{n}"
                msgs = obj.get("messages", []) or []
                mentions = obj.get("movieMentions", {}) or {}
                mentions = {str(k): str(v) for k, v in mentions.items()}

                yield ReDialConversation(
                    dataset=self,
                    split=split,
                    conversation_id=cid,
                    messages=list(msgs) if isinstance(msgs, list) else [],
                    movie_mentions=mentions,
                    emotion_cache=emotion,
                    bias_cache=bias,
                )

                n += 1
                yielded += 1
                if max_convos is not None and yielded >= max_convos:
                    return

    @staticmethod
    def _load_mentions_map(csv_path: Path) -> Dict[str, str]:
        csv_path = require_file(csv_path)
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {csv_path}")

            lower = {h.lower(): h for h in reader.fieldnames}

            def pick(col_candidates: Sequence[str]) -> str:
                for c in col_candidates:
                    if c.lower() in lower:
                        return lower[c.lower()]
                raise KeyError(f"Missing expected columns {col_candidates} in {reader.fieldnames}")

            id_col = pick(["movieid", "movie_id", "movieId", "movieID"])
            title_col = pick(["moviename", "movie_name", "title", "name", "movieTitle", "movie_title"])

            out: Dict[str, str] = {}
            for row in reader:
                mid = str(row.get(id_col, "")).strip()
                title = str(row.get(title_col, "")).strip()
                if mid:
                    out[mid] = title if title else out.get(mid, f"@{mid}")
            return out
