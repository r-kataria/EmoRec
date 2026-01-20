from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset.movielens import MovieLens25M
from dataset.redial import ReDialConversation, extract_movie_ids, get_speaker, safe_id
from bias.metrics import entropy, js_divergence_dict
from utils.progress import write_progress


class GenreBias:
    def __init__(self, cache_root: Path | str = "./cache", movielens: Optional[MovieLens25M] = None):
        self.cache_root = Path(cache_root)
        self.ml = movielens or MovieLens25M(cache_root=self.cache_root)
        self.ml.ensure_genre_baseline()
        self.ml.ensure_rating_counts()
        self.baseline = self.ml.genre_baseline()

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "genre" / "redial" / "ml-25m"

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
        conv_items: List[int] = []

        for i, msg in enumerate(conv.messages):
            raw_text = msg.get("text", "") if isinstance(msg, dict) else ""
            speaker = get_speaker(msg, i)
            redial_ids = extract_movie_ids(raw_text)

            ml_ids: List[int] = []
            for rid in redial_ids:
                title = (conv.movie_mentions or {}).get(str(rid))
                if not title:
                    continue
                mid = self.ml.map_title(title)
                if mid is None:
                    continue
                ml_ids.append(int(mid))

            bias_obj = None
            if speaker == "Recommender" and ml_ids:
                conv_items.extend(ml_ids)
                dist = self._dist_from_items(ml_ids)
                jsd = js_divergence_dict(dist, self.baseline) if dist and self.baseline else None
                ent = entropy(dist) if dist else 0.0
                bias_obj = {
                    "genres_dist": dist,
                    "genres_entropy": ent,
                    "js_divergence_vs_catalog": jsd,
                    "genre_coverage": int(len(dist)),
                }

            turns.append(
                {
                    "msg_idx": i,
                    "message_id": f"{conv.split}:{conv.conversation_id}:{i}",
                    "speaker": speaker,
                    "movielens_movie_ids": ml_ids,
                    "genre_bias": bias_obj,
                }
            )

        conv_dist = self._dist_from_items(conv_items)
        conv_jsd = js_divergence_dict(conv_dist, self.baseline) if conv_dist and self.baseline else None
        conv_ent = entropy(conv_dist) if conv_dist else 0.0

        summary = {
            "total_items": int(len(conv_items)),
            "unique_items": int(len(set(conv_items))),
            "genres_dist": conv_dist,
            "genres_entropy": conv_ent,
            "js_divergence_vs_catalog": conv_jsd,
            "genre_coverage": int(len(conv_dist)),
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

    def _dist_from_items(self, movie_ids: List[int]) -> Dict[str, float]:
        counts = defaultdict(int)
        total = 0
        for mid in movie_ids:
            for g in self.ml.genres(int(mid)):
                counts[g] += 1
                total += 1
        if total == 0:
            return {}
        return {g: counts[g] / total for g in counts}


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
                    "bias": "genre",
                    "split": split,
                    "processed_conversations": processed,
                    "computed_new_conversations": computed,
                    "elapsed_s": time.time() - t0,
                })
