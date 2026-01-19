# ReDial signals (minimal)

## Install deps
- python>=3.10
- pip install transformers torch

## Build caches (incremental)
- python scripts/build_emotions.py
- python scripts/build_bias.py

Progress is written to:
- cache/emotion_progress_train.json
- cache/emotion_progress_test.json
- cache/bias_progress_train.json
- cache/bias_progress_test.json

## Use lazily
```python
import sys
from pathlib import Path
sys.path.insert(0, "src")

from datasets import ReDialDataset
from emotion import GoEmotions

ds = ReDialDataset(cache_root="./cache")
emo = GoEmotions(cache_root="./cache", device="mps", top_k=5)

for conv in ds.iter("train", emotion=emo, max_convos=2):
    rec = conv.emotion  # computes once, caches per conversation
    print(rec["turns"][0]["emotion"])  # list of top-k emotions
```
