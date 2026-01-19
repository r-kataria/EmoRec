from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import MovieLens25M

CACHE = ROOT / "cache"

ml = MovieLens25M(cache_root=CACHE)
ml.ensure_rating_counts()
ml.ensure_genre_baseline()
