from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import ReDialDataset
from emotion.go_emotions_redial import GoEmotionsReDial

CACHE = ROOT / "cache"

ds = ReDialDataset(cache_root=CACHE, quiet_download=True)

# device examples:
#   device=-1 (CPU)
#   device=0  (CUDA GPU id 0)
#   device="mps" (Apple Silicon, if supported)
emo = GoEmotionsReDial(cache_root=CACHE, device="mps", top_k=5, resolve_movie_titles=True)

emo.build(ds, split="train", progress_path=CACHE / "emotion_progress_train.json", every=200)
emo.build(ds, split="test",  progress_path=CACHE / "emotion_progress_test.json",  every=200)
