from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import pipeline

from dataset.redial import (
    ReDialConversation,
    ReDialDataset,
    extract_movie_ids,
    get_speaker,
    replace_movie_tags_with_titles,
    safe_id,
)


class GoEmotions:
    MODEL = "SamLowe/roberta-base-go_emotions"

    def __init__(
        self,
        cache_root: Path | str = "./cache",
        device: Any = -1,
        top_k: int = 5,
        truncation: bool = True,
        resolve_movie_titles: bool = True,
    ):
        self.cache_root = Path(cache_root)
        self.resolve_movie_titles = resolve_movie_titles
        self.top_k = int(top_k)

        self.emo_clf = pipeline(
            task="text-classification",
            model=self.MODEL,
            top_k=self.top_k,
            device=device,
            truncation=truncation,
        )

    def _base_dir(self) -> Path:
        mode = "titles" if self.resolve_movie_titles else "raw"
        return self.cache_root / "emotion" / "go_emotions" / f"top{self.top_k}" / mode

    def _conv_path(self, conv: ReDialConversation) -> Path:
        return self._base_dir() / conv.split / f"{safe_id(conv.conversation_id)}.json"

    def has(self, conv: ReDialConversation) -> bool:
        return self._conv_path(conv).exists()

    def get(self, conv: ReDialConversation) -> Dict[str, Any]:
        p = self._conv_path(conv)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        record = self._compute(conv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        return record

    def _compute(self, conv: ReDialConversation) -> Dict[str, Any]:
        ds: ReDialDataset = conv.dataset
        mentions = dict(ds.movie_mentions_map())
        mentions.update(conv.movie_mentions or {})

        texts: List[str] = []
        speakers: List[str] = []
        movie_ids: List[List[str]] = []

        for i, msg in enumerate(conv.messages):
            raw_text = msg.get("text", "") if isinstance(msg, dict) else ""
            speakers.append(get_speaker(msg, i))
            movie_ids.append(extract_movie_ids(raw_text))
            texts.append(
                replace_movie_tags_with_titles(raw_text, mentions)
                if self.resolve_movie_titles
                else (raw_text or "")
            )

        outs = self.emo_clf(texts)

        preds_topk: List[List[Dict[str, float]]] = []
        for o in outs:
            if isinstance(o, list):
                preds_topk.append([{"label": str(d["label"]), "score": float(d["score"])} for d in o])
            else:
                preds_topk.append([{"label": str(o["label"]), "score": float(o["score"])}])

        return {
            "dataset": "redial",
            "split": conv.split,
            "conversationId": conv.conversation_id,
            "model": self.MODEL,
            "top_k": self.top_k,
            "resolve_movie_titles": self.resolve_movie_titles,
            "created_at_unix": time.time(),
            "turns": [
                {
                    "msg_idx": i,
                    "message_id": f"{conv.split}:{conv.conversation_id}:{i}",
                    "speaker": speakers[i],
                    "movie_ids": movie_ids[i],
                    "emotion": preds_topk[i],
                }
                for i in range(len(preds_topk))
            ],
        }

    def build(
        self,
        ds: ReDialDataset,
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

        for conv in ds.iter(split=split, start=start, max_convos=max_convos, emotion=self):
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
                            "signal": "emotion",
                            "model": self.MODEL,
                            "top_k": self.top_k,
                            "split": split,
                            "processed_conversations": processed,
                            "computed_new_conversations": computed,
                            "elapsed_s": time.time() - t0,
                        },
                        f,
                        ensure_ascii=False,
                    )
