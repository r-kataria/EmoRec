from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset.amazon_reviews import AmazonReviews2023Index
from dataset.cosrec import CoSRecRecEpisode, safe_id


class RatingBiasCoSRec:
    def __init__(
        self,
        cache_root: Path | str = "./cache",
        amazon_index: Optional[AmazonReviews2023Index] = None,
    ):
        self.cache_root = Path(cache_root)
        self.amazon = amazon_index or AmazonReviews2023Index(cache_root=self.cache_root)
        self.amazon.ensure_index()

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "rating" / "cosrec" / "amazon_2023" / "curated"

    def _ep_path(self, ep: CoSRecRecEpisode) -> Path:
        return self._base_dir() / f"{safe_id(ep.topic_id)}.json"

    def has(self, ep: CoSRecRecEpisode) -> bool:
        return self._ep_path(ep).exists()

    def get(self, ep: CoSRecRecEpisode) -> Dict[str, Any]:
        p = self._ep_path(ep)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        rec = self._compute(ep)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return rec

    def _compute(self, ep: CoSRecRecEpisode) -> Dict[str, Any]:
        asins = [str(a) for a, _ in ep.qrels]
        rels = [int(r) for _, r in ep.qrels]

        ratings: List[float] = []
        pcts: List[float] = []
        missing: List[str] = []

        for asin in asins:
            meta = self.amazon.get(asin)
            if not meta:
                missing.append(asin)
                continue
            val = self.amazon.rating_value(meta)
            if val is None:
                missing.append(asin)
                continue
            ratings.append(float(val))
            pcts.append(float(self.amazon.rating_percentile(float(val))))

        rating_obj = None
        if ratings:
            mean_r = float(sum(ratings) / len(ratings))
            med_r = float(statistics.median(ratings))
            mean_pct = float(sum(pcts) / len(pcts)) if pcts else None
            rating_obj = {
                "ratings": ratings,
                "percentiles": pcts,
                "mean_rating": mean_r,
                "median_rating": med_r,
                "mean_percentile": mean_pct,
            }

        summary = {
            "num_items_total": int(len(asins)),
            "num_items_mapped": int(len(ratings)),
            "mean_rating_all": float(sum(ratings) / len(ratings)) if ratings else None,
            "mean_percentile_all": float(sum(pcts) / len(pcts)) if pcts else None,
        }

        return {
            "dataset": "cosrec",
            "partition": "curated",
            "topic_id": ep.topic_id,
            "base_intent_id": ep.base_intent_id,
            "user_index": ep.user_index,
            "conversation_id": ep.conversation_id,
            "utterance_idx": ep.utterance_idx,
            "intent_type": ep.intent_type,
            "created_at_unix": time.time(),
            "catalogue": "amazon_2023",
            "qrels": [{"asin": a, "relevance": r} for a, r in zip(asins, rels)],
            "missing_asins": missing,
            "rating": rating_obj,
            "summary": summary,
        }

    def build(
        self,
        ds,
        intent_type: str = "recommendation",
        min_relevance: int = 1,
        max_new: Optional[int] = None,
        progress_path: Optional[Path | str] = None,
        every: int = 25,
    ) -> None:
        progress_p = Path(progress_path) if progress_path is not None else None
        processed = 0
        computed = 0
        t0 = time.time()

        for ep in ds.iter_rec_episodes(
            min_relevance=min_relevance,
            bias=self,
        ):
            if intent_type and ep.intent_type != intent_type:
                continue
            processed += 1
            if not self.has(ep):
                _ = self.get(ep)
                computed += 1
                if max_new is not None and computed >= max_new:
                    break

            if every and (processed % every == 0):
                if progress_p is not None:
                    progress_p.parent.mkdir(parents=True, exist_ok=True)
                    with open(progress_p, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "bias": "rating",
                                "dataset": "cosrec",
                                "partition": "curated",
                                "processed_episodes": processed,
                                "computed_new": computed,
                                "elapsed_s": time.time() - t0,
                            },
                            f,
                            ensure_ascii=False,
                        )
                print(
                    f"[bias:rating:cosrec] processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
                    flush=True,
                )

        print(
            f"[bias:rating:cosrec] done processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )
