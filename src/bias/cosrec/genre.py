from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from dataset.amazon_reviews import AmazonReviews2023Index
from dataset.cosrec import CoSRecRecEpisode, safe_id
from bias.metrics import entropy, js_divergence_dict
from utils.progress import write_progress


class GenreBiasCoSRec:
    def __init__(
        self,
        cache_root: Path | str = "./cache",
        amazon_index: Optional[AmazonReviews2023Index] = None,
    ):
        self.cache_root = Path(cache_root)
        self.amazon = amazon_index or AmazonReviews2023Index(cache_root=self.cache_root)
        self.amazon.ensure_index()
        self.baseline = self.amazon.category_baseline()

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "genre" / "cosrec" / "amazon_2023" / "curated"

    def _ep_path(self, ep: CoSRecRecEpisode) -> Path:
        return self._base_dir() / f"{safe_id(ep.topic_id)}.json"

    def has(self, ep: CoSRecRecEpisode) -> bool:
        return self._ep_path(ep).exists()

    def get(self, ep: CoSRecRecEpisode) -> Dict[str, Any]:
        p = self._ep_path(ep)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        asins = [str(a) for a, _ in ep.qrels]
        rels = [int(r) for _, r in ep.qrels]

        counts = defaultdict(int)
        total = 0
        for asin in asins:
            meta = self.amazon.get(asin)
            if not meta:
                continue
            cats = meta.get("categories") or []
            if not isinstance(cats, list):
                continue
            for c in cats:
                counts[str(c)] += 1
                total += 1
        dist = {c: counts[c] / total for c in counts} if total else {}

        jsd = js_divergence_dict(dist, self.baseline) if dist and self.baseline else None
        ent = entropy(dist) if dist else 0.0

        summary = {
            "total_items": int(len(asins)),
            "unique_items": int(len(set(asins))),
            "categories_dist": dist,
            "categories_entropy": ent,
            "js_divergence_vs_catalog": jsd,
            "category_coverage": int(len(dist)),
        }

        rec = {
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
            "summary": summary,
        }

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return rec

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
                write_progress(progress_p, {
                    "bias": "genre",
                    "dataset": "cosrec",
                    "partition": "curated",
                    "processed_episodes": processed,
                    "computed_new": computed,
                    "elapsed_s": time.time() - t0,
                })
                print(
                    f"[bias:genre:cosrec] processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
                    flush=True,
                )

        print(
            f"[bias:genre:cosrec] done processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )
