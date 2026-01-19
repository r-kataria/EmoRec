from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dataset.cosrec import CoSRecDataset

META_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"
REVIEW_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"

ASIN2CATEGORY_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/asin2category.json"

_WS_RE = re.compile(r"\s+")


try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):
        return x


def _clean_text(s: str) -> str:
    return str(re.sub(_WS_RE, " ", (s or "").replace("<br />", "")).strip())


def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ord(ch) <= 255) / len(s)


def _cat_to_stem(cat: str) -> str:
    # Matches UCSD filename convention: spaces/punct -> underscores, "&" -> "and"
    s = (cat or "").strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass
class AmazonReviews2023Subset:
    cache_root: Path
    cosrec: CoSRecDataset

    threshold_too_short_description: int = 10
    threshold_english_title_ascii: float = 0.5
    threshold_english_description_ascii: float = 0.8

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root)

        self.root = self.cache_root / "datasets" / "amazon_2023"
        self.asin2cat_path = self.root / "asin2category.json"

        self.raw_meta_dir = self.root / "raw" / "meta_categories"
        self.raw_review_dir = self.root / "raw" / "review_categories"
        self.processed_dir = self.root / "processed"

        self.asin_to_cat_small_path = self.root / "asin_to_category_qrels.json"
        self.categories_path = self.root / "categories_needed.txt"
        self.meta_urls_path = self.root / "meta_urls.txt"
        self.review_urls_path = self.root / "review_urls.txt"

        self.qrels_asins: Set[str] = set(self.cosrec.asin_doc_ids())

        self._asin_to_cat: Optional[Dict[str, str]] = None
        self._categories: Optional[Set[str]] = None

    def categories_needed(self) -> Set[str]:
        self._ensure_asin_mapping_loaded()
        return set(self._categories or set())

    def write_download_lists(self) -> None:
        cats = sorted(self.categories_needed())
        self.root.mkdir(parents=True, exist_ok=True)
        self.categories_path.write_text("\n".join(cats) + ("\n" if cats else ""), encoding="utf-8")

        meta_urls = [META_BASE_URL + f"meta_{c}.jsonl.gz" for c in cats]
        review_urls = [REVIEW_BASE_URL + f"{c}.jsonl.gz" for c in cats]

        self.meta_urls_path.write_text("\n".join(meta_urls) + ("\n" if meta_urls else ""), encoding="utf-8")
        self.review_urls_path.write_text("\n".join(review_urls) + ("\n" if review_urls else ""), encoding="utf-8")

    def _ensure_asin_mapping_loaded(self) -> None:
        if self._asin_to_cat is not None and self._categories is not None:
            return

        if self.asin_to_cat_small_path.exists():
            with open(self.asin_to_cat_small_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            asin_to_cat = {str(k): _cat_to_stem(str(v)) for k, v in raw.items()}


        else:
            if not self.asin2cat_path.exists():
                raise FileNotFoundError(f"Missing {self.asin2cat_path}. Run scripts/setup_datasets.sh first.")

            # stream parse huge dict with ijson
            try:
                import ijson  # pip install ijson
            except Exception:
                raise RuntimeError("Install ijson for streaming: pip install ijson")

            remaining = set(self.qrels_asins)
            asin_to_cat: Dict[str, str] = {}

            with open(self.asin2cat_path, "rb") as f:
                for asin, cat in ijson.kvitems(f, ""):
                    if asin in remaining:
                        asin_to_cat[str(asin)] = _cat_to_stem(str(cat))
                        remaining.remove(asin)
                        if not remaining:
                            break

            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.asin_to_cat_small_path, "w", encoding="utf-8") as f:
                json.dump(asin_to_cat, f, ensure_ascii=False, indent=2)

        cats = set(asin_to_cat.values())
        cats.discard("Unknown")  # no meta_Unknown file

        self._asin_to_cat = asin_to_cat
        self._categories = cats

    def process_dataset(self, include_reviews: bool = False) -> Path:
        """
        Meta-only processing (same filtering ideas as CoSRec catalogue_preprocessing.py, but qrels-restricted).
        Output:
          cache/datasets/amazon_2023/processed/processed_catalogue.jsonl.gz
        """
        self._ensure_asin_mapping_loaded()
        out = self.processed_dir / "processed_catalogue.jsonl.gz"
        out.parent.mkdir(parents=True, exist_ok=True)

        # overwrite
        with gzip.open(out, "wt", encoding="utf-8") as _:
            pass

        asin_to_cat = self._asin_to_cat or {}
        cats = sorted(self._categories or set())

        for cat in cats:
            meta_path = self.raw_meta_dir / f"meta_{cat}.jsonl.gz"
            if not meta_path.exists():
                continue

            # collect only qrels ASINs that pass filters
            catalogue: Dict[str, Dict[str, Any]] = {}

            with gzip.open(meta_path, "rt", encoding="utf-8") as fi:
                for line in tqdm(fi, desc=f"meta_{cat}", unit="lines", dynamic_ncols=True, leave=False):
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

                    title_c = _clean_text(title)
                    desc_c = [_clean_text(x) for x in description]
                    feat_c = [_clean_text(x) for x in features]

                    catalogue[parent_asin] = {
                        "parent_asin": parent_asin,
                        "title": title_c,
                        "description": desc_c + feat_c,
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

            # reviews optional (not needed for your current biases)
            if include_reviews:
                review_path = self.raw_review_dir / f"{cat}.jsonl.gz"
                if review_path.exists():
                    with gzip.open(review_path, "rt", encoding="utf-8") as fi:
                        for line in tqdm(fi, desc=f"reviews_{cat}", unit="lines", dynamic_ncols=True, leave=False):
                            data = json.loads(line)
                            parent_asin = data.get("parent_asin")
                            if not parent_asin or parent_asin not in catalogue:
                                continue
                            # keep it minimal; you can extend later
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

        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, default="./cache")
    ap.add_argument("--mode", type=str, choices=["lists", "process"], required=True)
    ap.add_argument("--include_reviews", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    cosrec = CoSRecDataset(cache_root=cache_root, quiet_download=True)
    ar = AmazonReviews2023Subset(cache_root=cache_root, cosrec=cosrec)

    if args.mode == "lists":
        ar.write_download_lists()
        print(f"Wrote:\n  {ar.categories_path}\n  {ar.meta_urls_path}\n  {ar.review_urls_path}")
    else:
        out = ar.process_dataset(include_reviews=bool(args.include_reviews))
        print(f"Processed catalogue:\n  {out}")


if __name__ == "__main__":
    main()
