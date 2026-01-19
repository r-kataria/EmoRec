from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

from dataset.cosrec import CoSRecDataset

ASIN2CATEGORY_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/asin2category.json"

META_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"
REVIEW_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"

_WS_RE = re.compile(r"\s+")


try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _clean_text(s: str) -> str:
    return str(re.sub(_WS_RE, " ", (s or "").replace("<br />", "")).strip())


def _ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ord(ch) <= 255) / len(s)


def _cat_to_stem(cat: str) -> str:
    # Fixes "Video Games" -> "Video_Games", "Industrial & Scientific" -> "Industrial_and_Scientific"
    s = (cat or "").strip().replace("&", "and")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass
class AmazonReviews2023Subset:
    """
    Uses CoSRec ASIN qrels to:
      - build the list of categories needed (from asin2category.json)
      - download only meta_<Category>.jsonl.gz (and optionally review <Category>.jsonl.gz) via wget
      - process a qrels-only processed_catalogue.jsonl.gz into cache

    Filtering logic is adapted from CoSRec's catalogue_preprocessing.py, but restricted to qrels ASINs.
    """

    cache_root: Path
    cosrec: CoSRecDataset
    verbose: bool = True

    threshold_too_short_description: int = 10
    threshold_english_title_ascii: float = 0.5
    threshold_english_description_ascii: float = 0.8

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root)
        self.root = self.cache_root / "datasets" / "amazon_2023"
        self.root.mkdir(parents=True, exist_ok=True)

        self.asin2cat_path = self.root / "asin2category.json"
        self.asin_to_cat_small_path = self.root / "asin_to_category_qrels.json"

        self.categories_path = self.root / "categories_needed.txt"
        self.meta_urls_path = self.root / "meta_urls.txt"
        self.review_urls_path = self.root / "review_urls.txt"

        self.raw_meta_dir = self.root / "raw" / "meta_categories"
        self.raw_review_dir = self.root / "raw" / "review_categories"
        self.processed_dir = self.root / "processed"

        self.qrels_asins: Set[str] = set(self.cosrec.asin_doc_ids())

        self._asin_to_cat: Optional[Dict[str, str]] = None
        self._categories: Optional[Set[str]] = None

    # ---------- step 1 ----------
    def ensure_asin2category(self) -> None:
        if self.asin2cat_path.exists() and self.asin2cat_path.stat().st_size > 0:
            if self.verbose:
                print("[amazon] asin2category.json already present")
            return
        if self.verbose:
            print("[amazon] asin2category.json (wget)")
        _run(["wget", "-c", "-O", str(self.asin2cat_path), ASIN2CATEGORY_URL])


    # ---------- step 2 ----------
    def build_needed_categories(self) -> None:
        """
        Produces:
          - cache/datasets/amazon_2023/asin_to_category_qrels.json  (small mapping)
          - cache/datasets/amazon_2023/categories_needed.txt
          - cache/datasets/amazon_2023/meta_urls.txt
          - cache/datasets/amazon_2023/review_urls.txt
        """
        self._ensure_asin_mapping_loaded()
        cats = sorted(self._categories or set())

        self.categories_path.write_text("\n".join(cats) + ("\n" if cats else ""), encoding="utf-8")
        meta_urls = [META_BASE_URL + f"meta_{c}.jsonl.gz" for c in cats]
        review_urls = [REVIEW_BASE_URL + f"{c}.jsonl.gz" for c in cats]
        self.meta_urls_path.write_text("\n".join(meta_urls) + ("\n" if meta_urls else ""), encoding="utf-8")
        self.review_urls_path.write_text("\n".join(review_urls) + ("\n" if review_urls else ""), encoding="utf-8")

        if self.verbose:
            print(f"[amazon] categories_needed.txt ({len(cats)} categories)")
            print(f"[amazon] meta_urls.txt / review_urls.txt written")

    def _ensure_asin_mapping_loaded(self) -> None:
        if self._asin_to_cat is not None and self._categories is not None:
            return

        if self.asin_to_cat_small_path.exists():
            with open(self.asin_to_cat_small_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            asin_to_cat = {str(k): _cat_to_stem(str(v)) for k, v in raw.items()}
        else:
            self.ensure_asin2category()

            try:
                import ijson  # pip install ijson
            except Exception:
                raise RuntimeError("Install ijson for streaming asin2category.json: pip install ijson")

            remaining = set(self.qrels_asins)
            asin_to_cat: Dict[str, str] = {}

            if self.verbose:
                print(f"[amazon] mapping {len(remaining)} qrels ASINs -> categories (streaming)")

            with open(self.asin2cat_path, "rb") as f:
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


    def download_needed_files(self, download_reviews: bool = False) -> None:
        self.build_needed_categories()

        self.raw_meta_dir.mkdir(parents=True, exist_ok=True)
        self.raw_review_dir.mkdir(parents=True, exist_ok=True)

        cats = sorted(self._categories or set())

        # Build "missing only" url lists
        missing_meta_urls = []
        missing_review_urls = []

        for c in cats:
            meta_dest = self.raw_meta_dir / f"meta_{c}.jsonl.gz"
            if not meta_dest.exists() or meta_dest.stat().st_size == 0:
                missing_meta_urls.append(META_BASE_URL + f"meta_{c}.jsonl.gz")

            if download_reviews:
                review_dest = self.raw_review_dir / f"{c}.jsonl.gz"
                if not review_dest.exists() or review_dest.stat().st_size == 0:
                    missing_review_urls.append(REVIEW_BASE_URL + f"{c}.jsonl.gz")

        # Write temp files under cache (not scripts/)
        missing_meta_list = self.root / "meta_urls_missing.txt"
        missing_review_list = self.root / "review_urls_missing.txt"

        if missing_meta_urls:
            missing_meta_list.write_text("\n".join(missing_meta_urls) + "\n", encoding="utf-8")
            if self.verbose:
                print(f"[amazon] downloading {len(missing_meta_urls)} missing meta files (wget)")
            _run(["wget", "-c", "-i", str(missing_meta_list), "-P", str(self.raw_meta_dir)])
        else:
            if self.verbose:
                print("[amazon] meta files already present (skip)")

        if download_reviews:
            if missing_review_urls:
                missing_review_list.write_text("\n".join(missing_review_urls) + "\n", encoding="utf-8")
                if self.verbose:
                    print(f"[amazon] downloading {len(missing_review_urls)} missing review files (wget)")
                _run(["wget", "-c", "-i", str(missing_review_list), "-P", str(self.raw_review_dir)])
            else:
                if self.verbose:
                    print("[amazon] review files already present (skip)")

    # ---------- step 4 ----------
    def process_dataset(self, include_reviews: bool = False) -> Path:
        """
        Writes:
        cache/datasets/amazon_2023/processed/processed_catalogue.jsonl.gz
        Meta-only is enough for price/store/categories/num_ratings biases.
        """
        self._ensure_asin_mapping_loaded()

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
            if not meta_path.exists():
                continue

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
                if review_path.exists():
                    with gzip.open(review_path, "rt", encoding="utf-8") as fi:
                        for line in tqdm(fi, desc=f"reviews_{cat}", unit="lines", dynamic_ncols=True, leave=False):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, default="./cache")
    ap.add_argument("--mode", choices=["lists", "download", "process"], required=True)
    ap.add_argument("--download_reviews", action="store_true")
    ap.add_argument("--include_reviews", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    cosrec = CoSRecDataset(cache_root=cache_root, quiet_download=True)
    ar = AmazonReviews2023Subset(cache_root=cache_root, cosrec=cosrec, verbose=(not args.quiet))

    if args.mode == "lists":
        ar.build_needed_categories()
    elif args.mode == "download":
        ar.download_needed_files(download_reviews=bool(args.download_reviews))
    else:
        ar.process_dataset(include_reviews=bool(args.include_reviews))


if __name__ == "__main__":
    main()
