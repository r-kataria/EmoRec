from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.redial import ReDialDataset
from dataset.cosrec import CoSRecDataset
from dataset.amazon_reviews import AmazonReviews2023Subset
from dataset.movielens import MovieLens25M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--download_reviews", action="store_true")
    ap.add_argument("--process_amazon", action="store_true")
    args = ap.parse_args()

    cache = (ROOT / args.cache_dir).resolve()

    print(f"[1/5] ReDial -> {cache}/datasets/redial")
    _ = ReDialDataset(cache_root=cache, quiet_download=False)

    print(f"[2/5] CoSRec -> {cache}/datasets/cosrec")
    cosrec = CoSRecDataset(cache_root=cache, quiet_download=False)

    print(f"[3/5] Amazon lists + download -> {cache}/datasets/amazon_2023")
    ar = AmazonReviews2023Subset(cache_root=cache, cosrec=cosrec, verbose=True)
    ar.build_needed_categories()
    ar.download_needed_files(download_reviews=bool(args.download_reviews))



    print(f"[4/5] MovieLens 25M -> {cache}/datasets/movielens/ml-25m")
    ml = MovieLens25M(cache_root=cache)
    ml.ensure_rating_counts()
    ml.ensure_genre_baseline()
    
    
    if args.process_amazon:
        print("[5/5] Process qrels-only catalogue")
        ar.process_dataset(include_reviews=bool(args.download_reviews))
    else:
        print("[5/5] Skip processing (use --process_amazon to build processed catalogue)")



    print("Done.")


if __name__ == "__main__":
    main()
