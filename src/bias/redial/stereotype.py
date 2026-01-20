from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset.redial import ReDialConversation, ReDialDataset, get_speaker, safe_id
from bias.stereotype_base import StereotypeBiasBase
from utils.progress import write_progress

# Labels: LABEL_0 = Non-biased, LABEL_1 = Biased.

class StereotypeBiasReDial(StereotypeBiasBase):
    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "stereotype" / "redial" / self._top_dir()

    def _conv_path(self, conv: ReDialConversation) -> Path:
        return self._base_dir() / conv.split / f"{safe_id(conv.conversation_id)}.json"

    def has(self, conv: ReDialConversation) -> bool:
        return self._conv_path(conv).exists()

    def get(self, conv: ReDialConversation) -> Dict[str, Any]:
        p = self._conv_path(conv)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        speakers: List[str] = []
        rec_texts: List[str] = []
        rec_indices: List[int] = []

        for i, msg in enumerate(conv.messages):
            speaker = get_speaker(msg, i)
            speakers.append(speaker)
            if speaker != "Recommender":
                continue
            raw_text = msg.get("text", "") if isinstance(msg, dict) else ""
            text = self._coerce_text(raw_text)
            rec_texts.append(text)
            rec_indices.append(i)

        preds = self._predict_texts(rec_texts)
        by_idx = {rec_indices[i]: preds[i] for i in range(len(rec_indices))}

        record = {
            "dataset": "redial",
            "split": conv.split,
            "conversationId": conv.conversation_id,
            "model": self.MODEL,
            "tokenizer": self.TOKENIZER,
            "top_k": self.top_k,
            "target_speaker": "Recommender",
            "created_at_unix": time.time(),
            "turns": [
                {
                    "msg_idx": i,
                    "message_id": f"{conv.split}:{conv.conversation_id}:{i}",
                    "speaker": speakers[i],
                    "bias": by_idx.get(i),
                }
                for i in range(len(conv.messages))
            ],
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        return record

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

        for conv in ds.iter(split=split, start=start, max_convos=max_convos, bias=self):
            processed += 1

            if not self.has(conv):
                _ = self.get(conv)
                computed += 1
                if max_new is not None and computed >= max_new:
                    break

            if every and (processed % every == 0):
                write_progress(progress_p, {
                    "bias": "stereotype",
                    "model": self.MODEL,
                    "tokenizer": self.TOKENIZER,
                    "top_k": self.top_k,
                    "split": split,
                    "processed_conversations": processed,
                    "computed_new_conversations": computed,
                    "elapsed_s": time.time() - t0,
                })
                print(
                    f"[bias:stereotype:redial] split={split} processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
                    flush=True,
                )

        print(
            f"[bias:stereotype:redial] split={split} done processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )
