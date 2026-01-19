from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.redial import ReDialDataset
from dataset.cosrec import CoSRecDataset
from dataset.amazon_reviews import AmazonReviews2023Subset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--download_reviews", action="store_true")
    ap.add_argument("--process_amazon", action="store_true")
    args = ap.parse_args()

    cache = (ROOT / args.cache_dir).resolve()

    print(f"[1/4] ReDial -> {cache}/datasets/redial")
    _ = ReDialDataset(cache_root=cache, quiet_download=False)

    print(f"[2/4] CoSRec -> {cache}/datasets/cosrec")
    cosrec = CoSRecDataset(cache_root=cache, quiet_download=False)

    print(f"[3/4] Amazon lists + download -> {cache}/datasets/amazon_2023")
    ar = AmazonReviews2023Subset(cache_root=cache, cosrec=cosrec, verbose=True)
    ar.build_needed_categories()
    ar.download_needed_files(download_reviews=bool(args.download_reviews))

    if args.process_amazon:
        print("[4/4] Process qrels-only catalogue")
        ar.process_dataset(include_reviews=bool(args.download_reviews))
    else:
        print("[4/4] Skip processing (use --process_amazon to build processed catalogue)")

    print("Done.")


if __name__ == "__main__":
    main()
