from __future__ import annotations

import bisect
import gzip
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

from utils.ensure import require_dir, require_file
from dataset.cosrec import CoSRecDataset

ASIN2CATEGORY_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/asin2category.json"  # informational
META_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"  # informational
REVIEW_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"  # informational

_WS_RE = re.compile(r"\s+")


def _clean_text(s: str) -> str:
    return str(re.sub(_WS_RE, " ", (s or "").replace("<br />", "")).strip())


def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ord(ch) <= 255) / len(s)


def _cat_to_stem(cat: str) -> str:
    s = (cat or "").strip().replace("&", "and")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass
class AmazonReviews2023Subset:
    """
    Pure loader + processor for a CoSRec-driven subset of Amazon Reviews 2023.

    This module DOES NOT download anything.
    It only:
      1) derives which categories are needed from CoSRec ASIN qrels
      2) requires the corresponding raw .jsonl.gz files exist
      3) writes a processed_catalogue.jsonl.gz cache

    Expected layout (created by scripts/download_data.py):
      <cache_root>/datasets/amazon_2023/asin2category.json
      <cache_root>/datasets/amazon_2023/raw/meta_categories/meta_<Category>.jsonl.gz
      <cache_root>/datasets/amazon_2023/raw/review_categories/<Category>.jsonl.gz   (optional)
    """

    cache_root: Path
    cosrec: CoSRecDataset
    verbose: bool = True

    threshold_too_short_description: int = 10
    threshold_english_title_ascii: float = 0.5
    threshold_english_description_ascii: float = 0.8

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root)
        self.root = require_dir(
            self.cache_root / "datasets" / "amazon_2023",
            hint="Run: python3 scripts/download_data.py",
        )

        self.asin2cat_path = self.root / "asin2category.json"
        self.asin_to_cat_small_path = self.root / "asin_to_category_qrels.json"

        self.categories_path = self.root / "categories_needed.txt"

        self.raw_meta_dir = self.root / "raw" / "meta_categories"
        self.raw_review_dir = self.root / "raw" / "review_categories"
        self.processed_dir = self.root / "processed"

        require_file(self.asin2cat_path, hint="Run: python3 scripts/download_data.py")
        require_dir(self.raw_meta_dir, hint="Run: python3 scripts/download_data.py")
        # raw_review_dir is optional unless include_reviews=True

        self.qrels_asins: Set[str] = set(self.cosrec.asin_doc_ids())

        self._asin_to_cat: Optional[Dict[str, str]] = None
        self._categories: Optional[Set[str]] = None

    # ---------- mapping ----------
    def _ensure_asin_mapping_loaded(self) -> None:
        if self._asin_to_cat is not None and self._categories is not None:
            return

        if self.asin_to_cat_small_path.exists() and self.asin_to_cat_small_path.stat().st_size > 0:
            with open(self.asin_to_cat_small_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            asin_to_cat = {str(k): _cat_to_stem(str(v)) for k, v in raw.items()}
        else:
            # streaming parse (required; asin2category is huge)
            try:
                import ijson  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Missing dependency 'ijson' for streaming asin2category.json. "
                    "Install: pip install ijson"
                ) from e

            asin2cat_p = require_file(self.asin2cat_path, hint="Run: python3 scripts/download_data.py")
            remaining = set(self.qrels_asins)
            asin_to_cat = {}

            if self.verbose:
                print(f"[amazon] mapping {len(remaining)} qrels ASINs -> categories (streaming)")

            with open(asin2cat_p, "rb") as f:
                for asin, cat in ijson.kvitems(f, ""):
                    if asin in remaining:
                        asin_to_cat[str(asin)] = _cat_to_stem(str(cat))
                        remaining.remove(asin)
                        if not remaining:
                            break

            with open(self.asin_to_cat_small_path, "w", encoding="utf-8") as f:
                json.dump(asin_to_cat, f, ensure_ascii=False, indent=2)

        cats = set(asin_to_cat.values())
        cats.discard("Unknown")

        self._asin_to_cat = asin_to_cat
        self._categories = cats

    def build_needed_categories(self) -> Set[str]:
        """
        Writes categories_needed.txt and returns the category set.
        """
        self._ensure_asin_mapping_loaded()
        cats = sorted(self._categories or set())
        self.categories_path.write_text("\n".join(cats) + ("\n" if cats else ""), encoding="utf-8")
        if self.verbose:
            print(f"[amazon] categories_needed.txt ({len(cats)} categories)")
        return set(cats)

    def ensure_raw_files(self, include_reviews: bool = False) -> None:
        """
        Ensures the raw files needed for the CoSRec qrels subset exist; otherwise raises.
        """
        self._ensure_asin_mapping_loaded()
        cats = sorted(self._categories or set())

        missing_meta = []
        for c in cats:
            p = self.raw_meta_dir / f"meta_{c}.jsonl.gz"
            if (not p.exists()) or p.stat().st_size == 0:
                missing_meta.append(str(p))

        missing_reviews = []
        if include_reviews:
            require_dir(self.raw_review_dir, hint="Run: python3 scripts/download_data.py")
            for c in cats:
                p = self.raw_review_dir / f"{c}.jsonl.gz"
                if (not p.exists()) or p.stat().st_size == 0:
                    missing_reviews.append(str(p))

        if missing_meta or missing_reviews:
            parts = []
            if missing_meta:
                parts.append("Missing meta files:\n  - " + "\n  - ".join(missing_meta[:50]) + ("\n  - ..." if len(missing_meta) > 50 else ""))
            if missing_reviews:
                parts.append("Missing review files:\n  - " + "\n  - ".join(missing_reviews[:50]) + ("\n  - ..." if len(missing_reviews) > 50 else ""))
            raise FileNotFoundError(
                "AmazonReviews2023Subset raw inputs missing.\n"
                + "\n".join(parts)
                + "\nHint: run scripts/download_data.py to populate cache/datasets/amazon_2023/"
            )

    # ---------- processing ----------
    def process_dataset(self, include_reviews: bool = False) -> Path:
        """
        Writes:
          <cache_root>/datasets/amazon_2023/processed/processed_catalogue.jsonl.gz
        """
        self._ensure_asin_mapping_loaded()
        self.ensure_raw_files(include_reviews=include_reviews)

        out = self.processed_dir / "processed_catalogue.jsonl.gz"
        out.parent.mkdir(parents=True, exist_ok=True)

        # overwrite
        with gzip.open(out, "wt", encoding="utf-8") as _:
            pass

        asin_to_cat = self._asin_to_cat or {}
        cats = sorted(self._categories or set())

        if self.verbose:
            print(f"[amazon] processing meta files for {len(cats)} categories -> {out.name}")

        for cat in cats:
            meta_path = self.raw_meta_dir / f"meta_{cat}.jsonl.gz"
            require_file(meta_path, hint="Run: python3 scripts/download_data.py")

            catalogue: Dict[str, Dict[str, Any]] = {}

            with gzip.open(meta_path, "rt", encoding="utf-8") as fi:
                for line in fi:
                    data = json.loads(line)

                    parent_asin = data.get("parent_asin")
                    if not parent_asin or parent_asin not in self.qrels_asins:
                        continue

                    mapped_cat = asin_to_cat.get(parent_asin)
                    if mapped_cat and mapped_cat != cat:
                        continue

                    title = data.get("title")
                    categories = data.get("categories")
                    description = data.get("description")
                    features = data.get("features")
                    details = data.get("details")
                    store = data.get("store")
                    price = data.get("price")
                    avg_rating = data.get("average_rating")
                    num_rating = data.get("rating_number")

                    if not isinstance(title, str) or not title:
                        continue
                    if not isinstance(description, list) or not all(isinstance(x, str) for x in description):
                        continue
                    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
                        continue
                    if len(description) == 0 and len(features) == 0:
                        continue
                    if not isinstance(details, dict) or len(details) == 0:
                        continue
                    if not isinstance(categories, list) or len(categories) == 0 or not all(isinstance(x, str) for x in categories):
                        continue
                    if not isinstance(num_rating, int) or num_rating <= 0:
                        continue
                    if not isinstance(avg_rating, float) or (not math.isfinite(avg_rating)) or avg_rating < 0.0:
                        continue
                    if not isinstance(store, str) or not store:
                        continue

                    if price is None or not isinstance(price, (float, str)):
                        continue
                    if isinstance(price, str):
                        try:
                            price = float(price)
                        except Exception:
                            continue
                    if (not math.isfinite(float(price))) or float(price) <= 0.0:
                        continue

                    full_description = " ".join(description) + " " + " ".join(features)
                    if len(full_description) < self.threshold_too_short_description:
                        continue
                    if _ascii_ratio(title) < self.threshold_english_title_ascii:
                        continue
                    if _ascii_ratio(full_description) < self.threshold_english_description_ascii:
                        continue

                    catalogue[parent_asin] = {
                        "parent_asin": parent_asin,
                        "title": _clean_text(title),
                        "description": [_clean_text(x) for x in description] + [_clean_text(x) for x in features],
                        "details": details,
                        "categories": categories,
                        "store": store,
                        "price": float(price),
                        "num_ratings": int(num_rating),
                        "avg_ratings": float(avg_rating),
                        "valid_reviews": [],
                        "valid_ratings": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                        "num_valid_ratings": 0,
                        "avg_valid_ratings": 0.0,
                    }

            if not catalogue:
                continue

            if include_reviews:
                review_path = self.raw_review_dir / f"{cat}.jsonl.gz"
                require_file(review_path, hint="Run: python3 scripts/download_data.py")

                with gzip.open(review_path, "rt", encoding="utf-8") as fi:
                    for line in fi:
                        data = json.loads(line)
                        parent_asin = data.get("parent_asin")
                        if not parent_asin or parent_asin not in catalogue:
                            continue
                        rating = data.get("rating")
                        if isinstance(rating, float) and 1.0 <= rating <= 5.0:
                            r = int(round(rating + 1e-3))
                            catalogue[parent_asin]["valid_ratings"][r] += 1

                for asin, v in catalogue.items():
                    total = sum(v["valid_ratings"].values())
                    if total > 0:
                        avg = sum(star * cnt for star, cnt in v["valid_ratings"].items()) / total
                        v["num_valid_ratings"] = int(total)
                        v["avg_valid_ratings"] = float(avg)

            with gzip.open(out, "at", encoding="utf-8") as fo:
                for asin in sorted(catalogue.keys()):
                    fo.write(json.dumps(catalogue[asin], ensure_ascii=False) + "\n")

        if self.verbose:
            print(f"[amazon] processed catalogue written: {out}")
        return out


class AmazonReviews2023Index:
    """
    Index for the CoSRec-specific Amazon Reviews 2023 subset.
    Uses ONLY qrels ASINs, not the full Amazon dataset.
    """

    def __init__(
        self,
        cache_root: Path | str = "./cache",
        cosrec: Optional[CoSRecDataset] = None,
        include_reviews: bool = False,
        verbose: bool = True,
    ):
        self.cache_root = Path(cache_root)
        self.cosrec = cosrec or CoSRecDataset(cache_root=self.cache_root)
        self.subset = AmazonReviews2023Subset(
            cache_root=self.cache_root,
            cosrec=self.cosrec,
            verbose=verbose,
        )
        self.include_reviews = bool(include_reviews)

        self.index_path = self.subset.processed_dir / "catalogue_index.json"

        self._index: Optional[Dict[str, Dict[str, Any]]] = None
        self._sorted_counts: Optional[list[int]] = None
        self._sorted_ratings: Optional[list[float]] = None
        self._top10_threshold: Optional[int] = None
        self._category_baseline: Optional[Dict[str, float]] = None

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(v):
            return float(default)
        return float(v)

    def _processed_path(self) -> Path:
        return self.subset.processed_dir / "processed_catalogue.jsonl.gz"

    def ensure_index(self) -> None:
        if self._index is not None:
            return

        if self.index_path.exists() and self.index_path.stat().st_size > 0:
            with open(self.index_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._index = {str(k): v for k, v in raw.items()}
            return

        processed = self._processed_path()
        if not processed.exists() or processed.stat().st_size == 0:
            processed = self.subset.process_dataset(include_reviews=self.include_reviews)
        processed = require_file(processed, hint="Run: python3 scripts/download_data.py")

        idx: Dict[str, Dict[str, Any]] = {}
        with gzip.open(processed, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                asin = str(data.get("parent_asin") or "").strip()
                if not asin:
                    continue
                categories = data.get("categories") or []
                if not isinstance(categories, list):
                    categories = []

                idx[asin] = {
                    "categories": [str(c) for c in categories if c],
                    "num_ratings": self._safe_int(data.get("num_ratings"), 0),
                    "avg_ratings": self._safe_float(data.get("avg_ratings"), 0.0),
                    "num_valid_ratings": self._safe_int(data.get("num_valid_ratings"), 0),
                    "avg_valid_ratings": self._safe_float(data.get("avg_valid_ratings"), 0.0),
                    "price": self._safe_float(data.get("price"), 0.0),
                    "store": str(data.get("store") or ""),
                }

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False)

        self._index = idx

    def index(self) -> Dict[str, Dict[str, Any]]:
        self.ensure_index()
        return dict(self._index or {})

    def get(self, asin: str) -> Optional[Dict[str, Any]]:
        self.ensure_index()
        if self._index is None:
            return None
        return self._index.get(str(asin))

    def rating_value(self, meta: Optional[Dict[str, Any]]) -> Optional[float]:
        if not meta:
            return None
        num_valid = self._safe_int(meta.get("num_valid_ratings"), 0)
        if num_valid > 0:
            val = meta.get("avg_valid_ratings")
        else:
            val = meta.get("avg_ratings")
        try:
            v = float(val)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        return float(v)

    def _ensure_sorted_counts(self) -> None:
        if self._sorted_counts is not None:
            return
        self.ensure_index()
        counts = [self._safe_int(v.get("num_ratings"), 0) for v in (self._index or {}).values()]
        counts.sort()
        self._sorted_counts = counts
        n = len(counts)
        self._top10_threshold = counts[int(0.9 * (n - 1))] if n > 0 else 0

    def _ensure_sorted_ratings(self) -> None:
        if self._sorted_ratings is not None:
            return
        self.ensure_index()
        ratings = []
        for v in (self._index or {}).values():
            r = self.rating_value(v)
            if r is None:
                continue
            ratings.append(float(r))
        ratings.sort()
        self._sorted_ratings = ratings

    def popularity_percentile(self, num_ratings: int) -> float:
        self._ensure_sorted_counts()
        if not self._sorted_counts:
            return 0.0
        i = bisect.bisect_left(self._sorted_counts, int(num_ratings))
        return float(i) / float(len(self._sorted_counts))

    def rating_percentile(self, rating: float) -> float:
        self._ensure_sorted_ratings()
        if not self._sorted_ratings:
            return 0.0
        i = bisect.bisect_left(self._sorted_ratings, float(rating))
        return float(i) / float(len(self._sorted_ratings))

    def top10_threshold(self) -> int:
        self._ensure_sorted_counts()
        return int(self._top10_threshold or 0)

    def category_baseline(self) -> Dict[str, float]:
        if self._category_baseline is not None:
            return dict(self._category_baseline)

        self.ensure_index()
        counts = defaultdict(int)
        total = 0
        for v in (self._index or {}).values():
            cats = v.get("categories") or []
            if not isinstance(cats, list):
                continue
            for c in cats:
                counts[str(c)] += 1
                total += 1
        self._category_baseline = {c: counts[c] / total for c in counts} if total else {}
        return dict(self._category_baseline)
