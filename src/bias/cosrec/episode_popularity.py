from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset.amazon_reviews import AmazonReviews2023Index
from dataset.cosrec import CoSRecRecEpisode, safe_id


def _popularity_bins(values: List[float], bins: int) -> List[float]:
    if not values or bins <= 0:
        return []
    counts = [0] * bins
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if x < 0:
            idx = 0
        elif x >= 1:
            idx = bins - 1
        else:
            idx = int(x * bins)
            if idx >= bins:
                idx = bins - 1
        counts[idx] += 1
    total = sum(counts)
    if total == 0:
        return []
    return [c / total for c in counts]


def _js_divergence(p: List[float], q: List[float]) -> Optional[float]:
    if not p or not q or len(p) != len(q):
        return None
    m = [(p[i] + q[i]) * 0.5 for i in range(len(p))]

    def kl(a: List[float], b: List[float]) -> float:
        out = 0.0
        for i, av in enumerate(a):
            if av <= 0:
                continue
            bv = b[i]
            if bv <= 0:
                continue
            out += av * math.log(av / bv)
        return float(out)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _js_similarity(p: List[float], q: List[float]) -> Optional[float]:
    jsd = _js_divergence(p, q)
    if jsd is None:
        return None
    if jsd <= 0:
        return 1.0
    denom = math.log(2.0)
    if denom <= 0:
        return None
    sim = 1.0 - (jsd / denom)
    if sim < 0:
        return 0.0
    if sim > 1:
        return 1.0
    return float(sim)


def _rank_utility(pcts: List[float]) -> Optional[float]:
    if not pcts:
        return None
    weights = [1.0 / math.log2(i + 2) for i in range(len(pcts))]
    denom = sum(weights)
    if denom <= 0:
        return None
    return float(sum(w * p for w, p in zip(weights, pcts)) / denom)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


class EpisodePopularityBiasCoSRec:
    def __init__(
        self,
        cache_root: Path | str = "./cache",
        amazon_index: Optional[AmazonReviews2023Index] = None,
        bins: int = 10,
    ):
        self.cache_root = Path(cache_root)
        self.amazon = amazon_index or AmazonReviews2023Index(cache_root=self.cache_root)
        self.amazon.ensure_index()
        self.bins = int(bins)

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "episode_popularity" / "cosrec" / "amazon_2023" / "curated"

    def _ep_path(self, ep: CoSRecRecEpisode) -> Path:
        return self._base_dir() / f"{safe_id(ep.topic_id)}.json"

    def has(self, ep: CoSRecRecEpisode) -> bool:
        return self._ep_path(ep).exists()

    def _dominant_category(self, asins: List[str]) -> Optional[str]:
        counts: Counter[str] = Counter()
        for asin in asins:
            meta = self.amazon.get(str(asin))
            if not meta:
                continue
            cats = meta.get("categories") or []
            if not isinstance(cats, list):
                continue
            for c in cats:
                counts[str(c)] += 1
        if not counts:
            return None
        best = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
        return str(best)

    def _episode_metrics(self, ep: CoSRecRecEpisode) -> Tuple[Dict[str, Any], List[float]]:
        ordered = sorted(ep.qrels, key=lambda x: (-int(x[1]), str(x[0])))
        asins = [str(a) for a, _ in ordered]
        rels = [int(r) for _, r in ordered]

        pcts: List[float] = []
        mapped_asins: List[str] = []
        for asin in asins:
            meta = self.amazon.get(asin)
            if not meta:
                continue
            num = int(meta.get("num_ratings", 0))
            pcts.append(float(self.amazon.popularity_percentile(num)))
            mapped_asins.append(asin)

        bins = _popularity_bins(pcts, self.bins)
        p_coverage = float(sum(1 for p in pcts if p >= 0.9) / len(pcts)) if pcts else None
        pi_rank = _rank_utility(pcts)
        mean_pct = _mean(pcts)

        intent_category = self._dominant_category(asins)
        uiop = None
        if intent_category:
            intent_bins = self.amazon.category_popularity_distribution(intent_category, bins=self.bins)
            if intent_bins and bins:
                uiop = _js_similarity(bins, intent_bins)

        metrics = {
            "num_items_total": int(len(asins)),
            "num_items_mapped": int(len(mapped_asins)),
            "p_coverage": p_coverage,
            "pi_rank_utility": pi_rank,
            "mean_percentile": mean_pct,
            "uiop_similarity": uiop,
            "uiop_category": intent_category,
            "qrels": [{"asin": a, "relevance": r} for a, r in zip(asins, rels)],
        }
        return metrics, bins

    def _write(self, ep: CoSRecRecEpisode, record: Dict[str, Any]) -> None:
        p = self._ep_path(ep)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)

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

        by_conv: Dict[str, List[CoSRecRecEpisode]] = defaultdict(list)
        for ep in ds.iter_rec_episodes(min_relevance=min_relevance, bias=self):
            if intent_type and ep.intent_type != intent_type:
                continue
            by_conv[str(ep.conversation_id)].append(ep)

        stop = False
        for _, eps in by_conv.items():
            if stop:
                break
            eps.sort(key=lambda e: (int(e.utterance_idx), int(e.user_index), str(e.topic_id)))
            prev_bins: Optional[List[float]] = None
            for ep in eps:
                metrics, bins = self._episode_metrics(ep)
                cep = _js_similarity(prev_bins, bins) if prev_bins and bins else None
                if bins:
                    prev_bins = bins

                processed += 1
                if not self.has(ep):
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
                        "bins": self.bins,
                        "episode_popularity": {
                            "p_coverage": metrics["p_coverage"],
                            "pi_rank_utility": metrics["pi_rank_utility"],
                            "mean_percentile": metrics["mean_percentile"],
                            "cep_similarity": cep,
                            "uiop_similarity": metrics["uiop_similarity"],
                            "uiop_category": metrics["uiop_category"],
                            "num_items_total": metrics["num_items_total"],
                            "num_items_mapped": metrics["num_items_mapped"],
                        },
                        "qrels": metrics["qrels"],
                    }
                    self._write(ep, rec)
                    computed += 1
                    if max_new is not None and computed >= max_new:
                        stop = True
                        break

                if every and (processed % every == 0):
                    if progress_p is not None:
                        progress_p.parent.mkdir(parents=True, exist_ok=True)
                        with open(progress_p, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "bias": "episode_popularity",
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
                        f"[bias:episode_popularity:cosrec] processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
                        flush=True,
                    )

        print(
            f"[bias:episode_popularity:cosrec] done processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )
