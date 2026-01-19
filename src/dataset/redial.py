from __future__ import annotations

import csv
import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

MOVIE_TAG_RE = re.compile(r"@(\d+)")


def safe_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))[:200] or "unknown"


def get_speaker(msg: Dict[str, Any], idx: int) -> str:
    for key in ("sender", "role", "from", "author", "authorType"):
        if key in msg and isinstance(msg[key], str):
            v = msg[key].lower()
            if "seek" in v or v in {"user", "customer", "client"}:
                return "Seeker"
            if "recom" in v or v in {"agent", "system", "bot", "assistant"}:
                return "Recommender"
    return "Seeker" if (idx % 2 == 0) else "Recommender"


def extract_movie_ids(text: str) -> List[str]:
    return MOVIE_TAG_RE.findall(text or "")


def replace_movie_tags_with_titles(text: str, mentions_map: Dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        k = m.group(1)
        return mentions_map.get(k, m.group(0))
    return MOVIE_TAG_RE.sub(repl, text or "")


@dataclass
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
    ReDial dataset download + streaming iterator.

    Stores under:
      <cache_root>/datasets/redial/{repo/, data/}
    """

    REPO_URL = "https://github.com/ReDialData/website"
    BRANCH = "data"
    ZIP_NAME = "redial_dataset.zip"

    def __init__(self, cache_root: Path | str = "./cache", quiet_download: bool = True):
        self.cache_root = Path(cache_root)
        self.root = self.cache_root / "datasets" / "redial"
        self.repo_dir = self.root / "repo"
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._mentions_map: Optional[Dict[str, str]] = None
        self._ensure_dataset(quiet=quiet_download)

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
        return self._mentions_map

    def iter(
        self,
        split: str = "train",
        start: int = 0,
        max_convos: Optional[int] = None,
        emotion=None,
        bias=None,
    ) -> Iterator[ReDialConversation]:
        """
        Streams JSONL line-by-line and yields ReDialConversation objects.
        Attach caches by passing emotion=..., bias=...
        """
        path = self.path_for_split(split)
        n = 0
        yielded = 0
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                if n < start:
                    n += 1
                    continue

                conv = json.loads(ln)
                cid = str(conv.get("conversationId", "")) or f"line_{n}"
                msgs = conv.get("messages", []) or []
                movie_mentions = conv.get("movieMentions", {}) or {}
                movie_mentions = {str(k): str(v) for k, v in movie_mentions.items()}

                yield ReDialConversation(
                    dataset=self,
                    split=split,
                    conversation_id=cid,
                    messages=msgs,
                    movie_mentions=movie_mentions,
                    emotion_cache=emotion,
                    bias_cache=bias,
                )

                n += 1
                yielded += 1
                if max_convos is not None and yielded >= max_convos:
                    return

    def _ensure_dataset(self, quiet: bool) -> None:
        train_p = self.data_dir / "train_data.jsonl"
        test_p = self.data_dir / "test_data.jsonl"
        csv_p = self.data_dir / "movies_with_mentions.csv"
        if train_p.exists() and test_p.exists() and csv_p.exists():
            return

        self.root.mkdir(parents=True, exist_ok=True)

        def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
            if quiet:
                subprocess.check_call(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

        if not self.repo_dir.exists():
            run(["git", "clone", self.REPO_URL, str(self.repo_dir)])

        run(["git", "fetch", "--all"], cwd=self.repo_dir)
        run(["git", "checkout", self.BRANCH], cwd=self.repo_dir)

        zip_path = self.repo_dir / self.ZIP_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"Expected {zip_path} after checkout '{self.BRANCH}'")

        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(self.data_dir))

        for p in (train_p, test_p, csv_p):
            if not p.exists():
                raise FileNotFoundError(f"Missing expected dataset file: {p}")

    @staticmethod
    def _load_mentions_map(csv_path: Path) -> Dict[str, str]:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {csv_path}")

            lower = {h.lower(): h for h in reader.fieldnames}

            def pick(col_candidates: Sequence[str]) -> str:
                for c in col_candidates:
                    if c.lower() in lower:
                        return lower[c.lower()]
                raise KeyError(f"Could not find any of {col_candidates} in {reader.fieldnames}")

            id_col = pick(["movieid", "movie_id", "movieId", "movieID"])
            title_col = pick(["moviename", "movie_name", "title", "name", "movieTitle", "movie_title"])

            out: Dict[str, str] = {}
            for row in reader:
                mid = str(row.get(id_col, "")).strip()
                title = str(row.get(title_col, "")).strip()
                if mid:
                    out[mid] = title if title else out.get(mid, f"@{mid}")
            return out
