from __future__ import annotations

import csv
import json
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

YEAR_RE = re.compile(r"\((\d{4})\)")
PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")


def _parse_title_year(s: str) -> Tuple[str, Optional[int]]:
    year = None
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
    URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

    def __init__(self, cache_root: Path | str = "./cache"):
        self.cache_root = Path(cache_root)
        self.root = self.cache_root / "datasets" / "movielens" / "ml-25m"
        self.root.mkdir(parents=True, exist_ok=True)

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

        self.ensure_downloaded()

    def movies_csv(self) -> Path:
        return self.root / "ml-25m" / "movies.csv"

    def ratings_csv(self) -> Path:
        return self.root / "ml-25m" / "ratings.csv"

    def ensure_downloaded(self) -> None:
        if self.movies_csv().exists() and self.ratings_csv().exists():
            return
        zip_path = self.root / "ml-25m.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(self.URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.root)
        if not (self.movies_csv().exists() and self.ratings_csv().exists()):
            raise FileNotFoundError("Expected ml-25m/movies.csv and ml-25m/ratings.csv after extraction")

    def _load_movies(self) -> None:
        if self._movies_loaded:
            return
        self._title_to_ids.clear()
        self._title_year_to_id.clear()
        self._id_to_genres.clear()

        with open(self.movies_csv(), "r", encoding="utf-8", newline="") as f:
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
        return self._id_to_genres.get(int(movie_id), [])

    def ensure_rating_counts(self) -> None:
        if self._counts_loaded:
            return
        counts_path = self.root / "rating_counts.json"
        if counts_path.exists():
            with open(counts_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._rating_counts = {int(k): int(v) for k, v in raw.items()}
        else:
            counts = defaultdict(int)
            with open(self.ratings_csv(), "r", encoding="utf-8", newline="") as f:
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
        import bisect
        self.ensure_rating_counts()
        c = self.rating_count(movie_id)
        if not self._sorted_counts:
            return 0.0
        i = bisect.bisect_left(self._sorted_counts, c)
        return float(i) / float(len(self._sorted_counts))

    def ensure_genre_baseline(self) -> None:
        if self._baseline_loaded:
            return
        path = self.root / "genre_baseline.json"
        if path.exists():
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

        self.ensure_rating_counts()
        best = None
        best_c = -1
        for mid in ids:
            c = self._rating_counts.get(int(mid), 0)
            if c > best_c:
                best_c = c
                best = int(mid)
        return best
