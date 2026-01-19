from __future__ import annotations

import bisect
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.ensure import require_dir, require_file

YEAR_RE = re.compile(r"\((\d{4})\)")
PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")


def _parse_title_year(s: str) -> Tuple[str, Optional[int]]:
    year: Optional[int] = None
    last = None
    for m in YEAR_RE.finditer(s or ""):
        last = m
    if last is not None:
        try:
            year = int(last.group(1))
        except Exception:
            year = None

    base = (s or "").strip()
    if last is not None:
        base = (base[: last.start()] + base[last.end() :]).strip()
    base = re.sub(r"\([^)]*\)", "", base).strip()
    return base, year


def _norm_title(s: str) -> str:
    s = (s or "").lower().replace("&", " and ")
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


class MovieLens25M:
    """
    Pure loader + derived caches for MovieLens 25M.

    Expected layout (created by scripts/download_data.py):
      <cache_root>/datasets/movielens/ml-25m/ml-25m/movies.csv
      <cache_root>/datasets/movielens/ml-25m/ml-25m/ratings.csv

    Derived caches written next to the dataset:
      rating_counts.json
      genre_baseline.json
    """

    def __init__(self, cache_root: Path | str = "./cache"):
        self.cache_root = Path(cache_root)
        self.root = require_dir(
            self.cache_root / "datasets" / "movielens" / "ml-25m",
            hint="Run: python3 scripts/download_data.py",
        )

        # Validate files exist (no downloading here)
        _ = require_file(self.movies_csv(), hint="Run: python3 scripts/download_data.py")
        _ = require_file(self.ratings_csv(), hint="Run: python3 scripts/download_data.py")

        self._movies_loaded = False
        self._title_to_ids: Dict[str, List[int]] = {}
        self._title_year_to_id: Dict[Tuple[str, int], int] = {}
        self._id_to_genres: Dict[int, List[str]] = {}

        self._counts_loaded = False
        self._rating_counts: Dict[int, int] = {}
        self._sorted_counts: List[int] = []
        self._top10_threshold: int = 0

        self._baseline_loaded = False
        self._genre_baseline: Dict[str, float] = {}

    def movies_csv(self) -> Path:
        return self.root / "ml-25m" / "movies.csv"

    def ratings_csv(self) -> Path:
        return self.root / "ml-25m" / "ratings.csv"

    # ----------------- movies -----------------
    def _load_movies(self) -> None:
        if self._movies_loaded:
            return

        self._title_to_ids.clear()
        self._title_year_to_id.clear()
        self._id_to_genres.clear()

        p = require_file(self.movies_csv(), hint="Run: python3 scripts/download_data.py")
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = int(row["movieId"])
                title = row["title"]
                genres = row["genres"].split("|") if row.get("genres") else []

                base, year = _parse_title_year(title)
                key = _norm_title(base)

                self._title_to_ids.setdefault(key, []).append(mid)
                if year is not None:
                    self._title_year_to_id[(key, year)] = mid

                self._id_to_genres[mid] = [g for g in genres if g and g != "(no genres listed)"]

        self._movies_loaded = True

    def genres(self, movie_id: int) -> List[str]:
        self._load_movies()
        return list(self._id_to_genres.get(int(movie_id), []))

    # ----------------- rating counts (derived cache) -----------------
    def ensure_rating_counts(self) -> None:
        if self._counts_loaded:
            return

        counts_path = self.root / "rating_counts.json"
        if counts_path.exists() and counts_path.stat().st_size > 0:
            with open(counts_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._rating_counts = {int(k): int(v) for k, v in raw.items()}
        else:
            ratings_p = require_file(self.ratings_csv(), hint="Run: python3 scripts/download_data.py")
            counts = defaultdict(int)
            with open(ratings_p, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    counts[int(row["movieId"])] += 1
            self._rating_counts = dict(counts)
            with open(counts_path, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self._rating_counts.items()}, f)

        self._sorted_counts = sorted(self._rating_counts.values())
        n = len(self._sorted_counts)
        self._top10_threshold = self._sorted_counts[int(0.9 * (n - 1))] if n > 0 else 0
        self._counts_loaded = True

    def rating_count(self, movie_id: int) -> int:
        self.ensure_rating_counts()
        return int(self._rating_counts.get(int(movie_id), 0))

    def top10_threshold(self) -> int:
        self.ensure_rating_counts()
        return int(self._top10_threshold)

    def popularity_percentile(self, movie_id: int) -> float:
        self.ensure_rating_counts()
        c = self.rating_count(movie_id)
        if not self._sorted_counts:
            return 0.0
        i = bisect.bisect_left(self._sorted_counts, c)
        return float(i) / float(len(self._sorted_counts))

    # ----------------- genre baseline (derived cache) -----------------
    def ensure_genre_baseline(self) -> None:
        if self._baseline_loaded:
            return

        path = self.root / "genre_baseline.json"
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as f:
                self._genre_baseline = {str(k): float(v) for k, v in json.load(f).items()}
            self._baseline_loaded = True
            return

        self._load_movies()
        counts = defaultdict(int)
        total = 0
        for gs in self._id_to_genres.values():
            for g in gs:
                counts[g] += 1
                total += 1

        self._genre_baseline = {g: counts[g] / total for g in counts} if total else {}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._genre_baseline, f, ensure_ascii=False)
        self._baseline_loaded = True

    def genre_baseline(self) -> Dict[str, float]:
        self.ensure_genre_baseline()
        return dict(self._genre_baseline)

    # ----------------- title mapping -----------------
    def map_title(self, title: str) -> Optional[int]:
        self._load_movies()
        base, year = _parse_title_year(title)
        key = _norm_title(base)

        if year is not None and (key, year) in self._title_year_to_id:
            return int(self._title_year_to_id[(key, year)])

        ids = self._title_to_ids.get(key)
        if not ids:
            return None
        if len(ids) == 1:
            return int(ids[0])

        # tie-break: most-rated
        self.ensure_rating_counts()
        best = None
        best_c = -1
        for mid in ids:
            c = self._rating_counts.get(int(mid), 0)
            if c > best_c:
                best_c = c
                best = int(mid)
        return best