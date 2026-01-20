from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset.movielens import MovieLens25M
from dataset.redial import ReDialConversation, extract_movie_ids, get_speaker, safe_id
from utils.progress import write_progress


class RedundancyBias:
    def __init__(self, cache_root: Path | str = "./cache", movielens: Optional[MovieLens25M] = None):
        self.cache_root = Path(cache_root)
        self.ml = movielens or MovieLens25M(cache_root=self.cache_root)
        self.ml.ensure_rating_counts()

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "redundancy" / "redial" / "ml-25m"

    def _conv_path(self, conv: ReDialConversation) -> Path:
        return self._base_dir() / conv.split / f"{safe_id(conv.conversation_id)}.json"

    def has(self, conv: ReDialConversation) -> bool:
        return self._conv_path(conv).exists()

    def get(self, conv: ReDialConversation) -> Dict[str, Any]:
        p = self._conv_path(conv)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        seen = set()
        all_items: List[int] = []
        turns: List[Dict[str, Any]] = []

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

            new_items = [m for m in ml_ids if m not in seen]
            repeated_items = [m for m in ml_ids if m in seen]

            if speaker == "Recommender":
                for m in ml_ids:
                    all_items.append(m)
                    seen.add(m)

            turns.append(
                {
                    "msg_idx": i,
                    "message_id": f"{conv.split}:{conv.conversation_id}:{i}",
                    "speaker": speaker,
                    "movielens_movie_ids": ml_ids,
                    "new_items": new_items if speaker == "Recommender" else [],
                    "repeated_items": repeated_items if speaker == "Recommender" else [],
                }
            )

        c = Counter(all_items)
        total = int(sum(c.values()))
        unique = int(len(c))
        redundancy_rate = float(1.0 - (unique / total)) if total > 0 else 0.0

        summary = {
            "total_items": total,
            "unique_items": unique,
            "redundancy_rate": redundancy_rate,
            "repeated_item_counts": {str(mid): int(cnt) for mid, cnt in c.items() if cnt > 1},
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
                    "bias": "redundancy",
                    "split": split,
                    "processed_conversations": processed,
                    "computed_new_conversations": computed,
                    "elapsed_s": time.time() - t0,
                })
