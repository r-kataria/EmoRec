from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from dataset.movielens import MovieLens25M
from dataset.redial import extract_movie_ids, get_speaker
from bias.metrics import gini, hhi


class ExposureConcentration:
    def __init__(self, cache_root: Path | str = "./cache", movielens: Optional[MovieLens25M] = None):
        self.cache_root = Path(cache_root)
        self.ml = movielens or MovieLens25M(cache_root=self.cache_root)
        self.ml.ensure_rating_counts()

    def _out_path(self, split: str) -> Path:
        return self.cache_root / "bias" / "exposure_concentration" / "redial" / "ml-25m" / f"{split}.json"

    def has(self, split: str) -> bool:
        return self._out_path(split).exists()

    def build(
        self,
        ds,
        split: str = "train",
        start: int = 0,
        max_convos: Optional[int] = None,
        force: bool = False,
        progress_path: Optional[Path | str] = None,
        every: int = 200,
    ) -> Dict[str, Any]:
        out_p = self._out_path(split)
        if out_p.exists() and not force:
            with open(out_p, "r", encoding="utf-8") as f:
                return json.load(f)

        progress_p = Path(progress_path) if progress_path is not None else None

        counts = Counter()
        processed = 0
        t0 = time.time()

        for conv in ds.iter(split=split, start=start, max_convos=max_convos):
            processed += 1
            for i, msg in enumerate(conv.messages):
                if get_speaker(msg, i) != "Recommender":
                    continue
                raw_text = msg.get("text", "") if isinstance(msg, dict) else ""
                redial_ids = extract_movie_ids(raw_text)
                for rid in redial_ids:
                    title = (conv.movie_mentions or {}).get(str(rid))
                    if not title:
                        continue
                    mid = self.ml.map_title(title)
                    if mid is None:
                        continue
                    counts[int(mid)] += 1

            if progress_p is not None and every and (processed % every == 0):
                progress_p.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_p, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "bias": "exposure_concentration",
                            "split": split,
                            "processed_conversations": processed,
                            "unique_items_so_far": len(counts),
                            "total_mentions_so_far": sum(counts.values()),
                            "elapsed_s": time.time() - t0,
                        },
                        f,
                        ensure_ascii=False,
                    )

        freqs = list(counts.values())
        total_mentions = int(sum(freqs))
        unique_items = int(len(freqs))

        top_sorted = sorted(freqs, reverse=True)

        def top_share(k: int) -> float:
            if total_mentions == 0:
                return 0.0
            return float(sum(top_sorted[:k]) / total_mentions)

        rec = {
            "dataset": "redial",
            "split": split,
            "movielens_version": "ml-25m",
            "created_at_unix": time.time(),
            "unique_items": unique_items,
            "total_mentions": total_mentions,
            "top10_share": top_share(10),
            "top50_share": top_share(50),
            "top100_share": top_share(100),
            "gini": gini(freqs),
            "hhi": hhi(freqs),
        }

        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return rec
