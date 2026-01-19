from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.cosrec import CoSRecDataset
from dataset.amazon_reviews import AmazonReviews2023Subset

CACHE = ROOT / "cache"

cosrec = CoSRecDataset(cache_root=CACHE, quiet_download=True)
ar = AmazonReviews2023Subset(cache_root=CACHE, cosrec=cosrec)

# 1) Download asin2category.json
ar.ensure_asin2category()

# 2) Download only the category meta files needed by qrels ASINs
#    (set download_reviews=True if you also want review category gz files)
ar.ensure_category_files(download_reviews=False, workers=4)

# 3) Build processed catalogue (meta-only by default; enough for price/category/store/popularity bias)
out = ar.process_dataset(include_reviews=False)
print(out)
