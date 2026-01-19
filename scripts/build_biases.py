from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import ReDialDataset, MovieLens25M
from bias import PopularityBias, RedundancyBias, GenreBias, ExposureConcentration

CACHE = ROOT / "cache"

ds = ReDialDataset(cache_root=CACHE, quiet_download=True)
ml = MovieLens25M(cache_root=CACHE)

pop = PopularityBias(cache_root=CACHE, movielens=ml)
red = RedundancyBias(cache_root=CACHE, movielens=ml)
gen = GenreBias(cache_root=CACHE, movielens=ml)
exp = ExposureConcentration(cache_root=CACHE, movielens=ml)

pop.build(ds, split="train", progress_path=CACHE / "bias_popularity_train.json", every=200)
pop.build(ds, split="test",  progress_path=CACHE / "bias_popularity_test.json",  every=200)

red.build(ds, split="train", progress_path=CACHE / "bias_redundancy_train.json", every=200)
red.build(ds, split="test",  progress_path=CACHE / "bias_redundancy_test.json",  every=200)

gen.build(ds, split="train", progress_path=CACHE / "bias_genre_train.json", every=200)
gen.build(ds, split="test",  progress_path=CACHE / "bias_genre_test.json",  every=200)

exp.build(ds, split="train", progress_path=CACHE / "bias_exposure_train.json", every=200)
exp.build(ds, split="test",  progress_path=CACHE / "bias_exposure_test.json",  every=200)
