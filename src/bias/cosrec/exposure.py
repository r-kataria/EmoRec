from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from bias.metrics import gini, hhi

class ExposureConcentrationCoSRec:
    def __init__(self, cache_root: Path | str = "./cache"):
        self.cache_root = Path(cache_root)

    def _out_path(self) -> Path:
        return (
            self.cache_root
            / "bias"
            / "exposure_concentration"
            / "cosrec"
            / "amazon_2023"
            / "curated.json"
        )

    def has(self) -> bool:
        return self._out_path().exists()

    def build(
        self,
        ds,
        intent_type: str = "recommendation",
        min_relevance: int = 1,
        max_new: Optional[int] = None,
        force: bool = False,
        progress_path: Optional[Path | str] = None,
        every: int = 25,
    ) -> Dict[str, Any]:
        out_p = self._out_path()
        if out_p.exists() and not force:
            with open(out_p, "r", encoding="utf-8") as f:
                return json.load(f)

        progress_p = Path(progress_path) if progress_path is not None else None
        counts = Counter()
        processed = 0
        t0 = time.time()

        for ep in ds.iter_rec_episodes(
            min_relevance=min_relevance,
        ):
            if intent_type and ep.intent_type != intent_type:
                continue
            processed += 1
            for asin, _ in ep.qrels:
                counts[str(asin)] += 1

            if max_new is not None and processed >= max_new:
                break

            if progress_p is not None and every and (processed % every == 0):
                progress_p.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_p, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "bias": "exposure_concentration",
                            "dataset": "cosrec",
                            "partition": "curated",
                            "processed_episodes": processed,
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
            "dataset": "cosrec",
            "partition": "curated",
            "catalogue": "amazon_2023",
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
