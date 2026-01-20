from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset.movielens import MovieLens25M
from dataset.redial import ReDialConversation, extract_movie_ids, get_speaker, safe_id
from utils.progress import write_progress


class PopularityBias:
    def __init__(self, cache_root: Path | str = "./cache", movielens: Optional[MovieLens25M] = None):
        self.cache_root = Path(cache_root)
        self.ml = movielens or MovieLens25M(cache_root=self.cache_root)
        self.ml.ensure_rating_counts()

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "popularity" / "redial" / "ml-25m"

    def _conv_path(self, conv: ReDialConversation) -> Path:
        return self._base_dir() / conv.split / f"{safe_id(conv.conversation_id)}.json"

    def has(self, conv: ReDialConversation) -> bool:
        return self._conv_path(conv).exists()

    def get(self, conv: ReDialConversation) -> Dict[str, Any]:
        p = self._conv_path(conv)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        turns: List[Dict[str, Any]] = []
        all_counts: List[int] = []
        all_pcts: List[float] = []
        top10_thr = self.ml.top10_threshold()

        for i, msg in enumerate(conv.messages):
            raw_text = msg.get("text", "") if isinstance(msg, dict) else ""
            speaker = get_speaker(msg, i)
            redial_ids = extract_movie_ids(raw_text)

            ml_ids: List[int] = []
            unmapped: List[str] = []
            counts: List[int] = []
            pcts: List[float] = []

            for rid in redial_ids:
                title = (conv.movie_mentions or {}).get(str(rid))
                if not title:
                    unmapped.append(str(rid))
                    continue
                mid = self.ml.map_title(title)
                if mid is None:
                    unmapped.append(str(rid))
                    continue
                ml_ids.append(int(mid))
                c = self.ml.rating_count(int(mid))
                pct = self.ml.popularity_percentile(int(mid))
                counts.append(int(c))
                pcts.append(float(pct))

            popularity_obj = None
            if speaker == "Recommender" and ml_ids:
                all_counts.extend(counts)
                all_pcts.extend(pcts)

                mean_c = float(sum(counts) / len(counts))
                med_c = float(statistics.median(counts))
                mean_pct = float(sum(pcts) / len(pcts))

                head_share_top10 = (
                    float(sum(1 for c in counts if c >= top10_thr) / len(counts))
                    if top10_thr > 0
                    else 0.0
                )
                novelty = [float(-math.log1p(c)) for c in counts]
                novelty_mean = float(sum(novelty) / len(novelty))

                popularity_obj = {
                    "counts": counts,
                    "percentiles": pcts,
                    "mean_count": mean_c,
                    "median_count": med_c,
                    "mean_percentile": mean_pct,
                    "head_share_top10pct": head_share_top10,
                    "novelty_mean": novelty_mean,
                }

            turns.append(
                {
                    "msg_idx": i,
                    "message_id": f"{conv.split}:{conv.conversation_id}:{i}",
                    "speaker": speaker,
                    "redial_movie_ids": redial_ids,
                    "movielens_movie_ids": ml_ids,
                    "unmapped_redial_movie_ids": unmapped,
                    "popularity": popularity_obj,
                }
            )

        summary = {
            "num_recommender_turns_with_movies": sum(
                1 for t in turns if t["speaker"] == "Recommender" and t["movielens_movie_ids"]
            ),
            "num_movies_mapped_total": int(len(all_counts)),
            "mean_count_all": float(sum(all_counts) / len(all_counts)) if all_counts else None,
            "mean_percentile_all": float(sum(all_pcts) / len(all_pcts)) if all_pcts else None,
        }

        rec = {
            "dataset": "redial",
            "split": conv.split,
            "conversationId": conv.conversation_id,
            "movielens_version": "ml-25m",
            "created_at_unix": time.time(),
            "turns": turns,
            "summary": summary,
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return rec

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

            if every and (processed % every == 0):
                write_progress(progress_p, {
                    "bias": "popularity",
                    "split": split,
                    "processed_conversations": processed,
                    "computed_new_conversations": computed,
                    "elapsed_s": time.time() - t0,
                })
