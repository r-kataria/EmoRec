from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dataset.cosrec import CoSRecRecEpisode, safe_id
from bias.stereotype_base import StereotypeBiasBase


class StereotypeBiasCoSRec(StereotypeBiasBase):
    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "stereotype" / "cosrec" / self._top_dir() / "curated"

    def _ep_path(self, ep: CoSRecRecEpisode) -> Path:
        return self._base_dir() / f"{safe_id(ep.topic_id)}.json"

    def has(self, ep: CoSRecRecEpisode) -> bool:
        return self._ep_path(ep).exists()

    def get(self, ep: CoSRecRecEpisode) -> Dict[str, Any]:
        p = self._ep_path(ep)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        text = self._coerce_text(ep.system_text)
        preds = self._predict_texts([text])[0]

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
            "model": self.MODEL,
            "tokenizer": self.TOKENIZER,
            "top_k": self.top_k,
            "target_speaker": "System",
            "bias": preds,
            "system_text": text,
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

            if progress_p is not None and every and (processed % every == 0):
                progress_p.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_p, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "bias": "stereotype",
                            "dataset": "cosrec",
                            "partition": "curated",
                            "processed_episodes": processed,
                            "computed_new": computed,
                            "elapsed_s": time.time() - t0,
                        },
                        f,
                        ensure_ascii=False,
                    )
