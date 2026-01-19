#!/usr/bin/env python3
"""
scripts/setup_datasets.py

One setup script that:
1) Downloads ReDial (git + unzip) into ./cache/datasets/redial/
2) Downloads CoSRec (git) into ./cache/datasets/cosrec/
3) Downloads asin2category.json (wget) into ./cache/datasets/amazon_2023/
4) Uses CoSRec qrels to generate the required Amazon category file URL lists
5) Downloads only the required Amazon meta_category gz files (wget)
6) Optionally processes a qrels-only processed catalogue (python)

Requires: git, unzip, wget
Optional (recommended): ijson (pip install ijson), tqdm (pip install tqdm)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ASIN2CATEGORY_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/asin2category.json"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def ensure_git_repo(url: str, dest: Path, pull: bool = True) -> None:
    if (dest / ".git").exists():
        if pull:
            run(["git", "pull", "-q"], cwd=dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", url, str(dest)])


def wget(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["wget", "-c", "-O", str(out), url])


def wget_list(url_list: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["wget", "-c", "-i", str(url_list), "-P", str(out_dir)])


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["unzip", "-o", str(zip_path), "-d", str(out_dir)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="./cache")
    ap.add_argument("--download_reviews", action="store_true", help="Also download review_categories gz files")
    ap.add_argument(
        "--process_amazon",
        action="store_true",
        help="After downloads, build processed_catalogue.jsonl.gz (qrels-only) into cache",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache = (root / args.cache_dir).resolve()

    # Make src importable (so we can call your dataset.amazon_reviews module)
    sys.path.insert(0, str(root / "src"))

    # Paths
    redial_repo = cache / "datasets" / "redial" / "repo"
    redial_data = cache / "datasets" / "redial" / "data"

    cosrec_repo = cache / "datasets" / "cosrec" / "repo"

    amazon_root = cache / "datasets" / "amazon_2023"
    asin2cat_path = amazon_root / "asin2category.json"
    amazon_meta_dir = amazon_root / "raw" / "meta_categories"
    amazon_review_dir = amazon_root / "raw" / "review_categories"

    print(f"[1/6] Using cache dir: {cache}")
    (cache / "datasets").mkdir(parents=True, exist_ok=True)

    print("[2/6] ReDial: git clone + checkout data + unzip")
    ensure_git_repo("https://github.com/ReDialData/website", redial_repo, pull=True)
    run(["git", "fetch", "--all", "-q"], cwd=redial_repo)
    run(["git", "checkout", "-q", "data"], cwd=redial_repo)
    unzip(redial_repo / "redial_dataset.zip", redial_data)
    print(f"      ReDial ready: {redial_data}")

    print("[3/6] CoSRec: git clone/pull")
    ensure_git_repo("https://github.com/CAMEO-22/CoSRec", cosrec_repo, pull=True)
    print(f"      CoSRec ready: {cosrec_repo}")

    print("[4/6] Amazon: download asin2category.json (wget)")
    wget(ASIN2CATEGORY_URL, asin2cat_path)
    print(f"      asin2category ready: {asin2cat_path}")

    print("[5/6] Generate required Amazon category URL lists from CoSRec qrels (python)")
    # This uses your existing module: src/dataset/amazon_reviews.py
    from dataset.cosrec import CoSRecDataset
    from dataset.amazon_reviews import AmazonReviews2023Subset

    cosrec = CoSRecDataset(cache_root=cache, quiet_download=True)  # repo already present; this just parses qrels
    ar = AmazonReviews2023Subset(cache_root=cache, cosrec=cosrec)
    ar.write_download_lists()
    print(f"      categories: {ar.categories_path}")
    print(f"      meta urls : {ar.meta_urls_path}")
    print(f"      review urls: {ar.review_urls_path}")

    print("[6/6] Download required Amazon category files (wget)")
    wget_list(ar.meta_urls_path, amazon_meta_dir)
    print(f"      meta_categories downloaded to: {amazon_meta_dir}")

    if args.download_reviews:
        wget_list(ar.review_urls_path, amazon_review_dir)
        print(f"      review_categories downloaded to: {amazon_review_dir}")

    if args.process_amazon:
        print("[extra] Processing qrels-only catalogue (meta-only by default)")
        out = ar.process_dataset(include_reviews=bool(args.download_reviews))
        print(f"      processed catalogue: {out}")

    print("Done.")


if __name__ == "__main__":
    main()
