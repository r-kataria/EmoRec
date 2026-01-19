from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset.movielens import MovieLens25M
from dataset.redial import ReDialConversation, extract_movie_ids, get_speaker, safe_id


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


class EpisodePopularityBias:
    def __init__(
        self,
        cache_root: Path | str = "./cache",
        movielens: Optional[MovieLens25M] = None,
        bins: int = 10,
    ):
        self.cache_root = Path(cache_root)
        self.ml = movielens or MovieLens25M(cache_root=self.cache_root)
        self.ml.ensure_rating_counts()
        self.bins = int(bins)

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "episode_popularity" / "redial" / "ml-25m"

    def _conv_path(self, conv: ReDialConversation) -> Path:
        return self._base_dir() / conv.split / f"{safe_id(conv.conversation_id)}.json"

    def has(self, conv: ReDialConversation) -> bool:
        return self._conv_path(conv).exists()

    def get(self, conv: ReDialConversation) -> Dict[str, Any]:
        p = self._conv_path(conv)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        rec = self._compute(conv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return rec

    def _map_redial_ids(self, conv: ReDialConversation, redial_ids: List[str]) -> List[int]:
        out: List[int] = []
        for rid in redial_ids:
            title = (conv.movie_mentions or {}).get(str(rid))
            if not title:
                continue
            mid = self.ml.map_title(title)
            if mid is None:
                continue
            out.append(int(mid))
        return out

    def _compute(self, conv: ReDialConversation) -> Dict[str, Any]:
        turns: List[Dict[str, Any]] = []
        prev_bins: Optional[List[float]] = None

        p_vals: List[float] = []
        pi_vals: List[float] = []
        cep_vals: List[float] = []
        uiop_vals: List[float] = []

        for i, msg in enumerate(conv.messages):
            raw_text = msg.get("text", "") if isinstance(msg, dict) else ""
            speaker = get_speaker(msg, i)
            redial_ids = extract_movie_ids(raw_text)
            ml_ids = self._map_redial_ids(conv, redial_ids)

            episode_popularity = None
            if speaker == "Recommender" and ml_ids:
                pcts = [float(self.ml.popularity_percentile(mid)) for mid in ml_ids]
                bins = _popularity_bins(pcts, self.bins)

                p_coverage = float(sum(1 for p in pcts if p >= 0.9) / len(pcts)) if pcts else None
                pi_rank = _rank_utility(pcts)
                mean_pct = _mean(pcts)

                cep = _js_similarity(prev_bins, bins) if prev_bins and bins else None
                if bins:
                    prev_bins = bins

                uiop = None
                user_items = 0
                for j in range(i - 1, -1, -1):
                    if get_speaker(conv.messages[j], j) != "Seeker":
                        continue
                    user_ids = extract_movie_ids(
                        conv.messages[j].get("text", "") if isinstance(conv.messages[j], dict) else ""
                    )
                    user_ml = self._map_redial_ids(conv, user_ids)
                    user_items = len(user_ml)
                    if user_ml:
                        user_pcts = [float(self.ml.popularity_percentile(mid)) for mid in user_ml]
                        user_bins = _popularity_bins(user_pcts, self.bins)
                        uiop = _js_similarity(bins, user_bins) if bins and user_bins else None
                    break

                episode_popularity = {
                    "num_items": int(len(pcts)),
                    "p_coverage": p_coverage,
                    "pi_rank_utility": pi_rank,
                    "mean_percentile": mean_pct,
                    "cep_similarity": cep,
                    "uiop_similarity": uiop,
                    "uiop_user_items": int(user_items),
                }

                if p_coverage is not None:
                    p_vals.append(float(p_coverage))
                if pi_rank is not None:
                    pi_vals.append(float(pi_rank))
                if cep is not None:
                    cep_vals.append(float(cep))
                if uiop is not None:
                    uiop_vals.append(float(uiop))

            turns.append(
                {
                    "msg_idx": i,
                    "message_id": f"{conv.split}:{conv.conversation_id}:{i}",
                    "speaker": speaker,
                    "movielens_movie_ids": ml_ids,
                    "episode_popularity": episode_popularity,
                }
            )

        summary = {
            "num_recommender_turns_with_items": int(
                sum(1 for t in turns if t["speaker"] == "Recommender" and t["movielens_movie_ids"])
            ),
            "p_coverage_mean": _mean(p_vals),
            "pi_rank_utility_mean": _mean(pi_vals),
            "cep_similarity_mean": _mean(cep_vals),
            "uiop_similarity_mean": _mean(uiop_vals),
        }

        return {
            "dataset": "redial",
            "split": conv.split,
            "conversationId": conv.conversation_id,
            "movielens_version": "ml-25m",
            "bins": self.bins,
            "created_at_unix": time.time(),
            "turns": turns,
            "summary": summary,
        }

    def build(
        self,
        ds,
        split: str = "train",
        start: int = 0,
        max_convos: Optional[int] = None,
        max_new: Optional[int] = None,
        progress_path: Optional[Path | str] = None,
        every: int = 200,
    ) -> None:
        progress_p = Path(progress_path) if progress_path is not None else None
        processed = 0
        computed = 0
        t0 = time.time()

        for conv in ds.iter(split=split, start=start, max_convos=max_convos, bias=self):
            processed += 1
            if not self.has(conv):
                _ = self.get(conv)
                computed += 1
                if max_new is not None and computed >= max_new:
                    break

            if progress_p is not None and every and (processed % every == 0):
                progress_p.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_p, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "bias": "episode_popularity",
                            "split": split,
                            "processed_conversations": processed,
                            "computed_new": computed,
                            "elapsed_s": time.time() - t0,
                        },
                        f,
                        ensure_ascii=False,
                    )

        print(
            f"[bias:episode_popularity:redial] split={split} done processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )
