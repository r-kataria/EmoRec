from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import pipeline

from dataset.cosrec import CoSRecCuratedEpisode, safe_id


class GoEmotionsCoSRec:
    MODEL = "SamLowe/roberta-base-go_emotions"

    def __init__(
        self,
        cache_root: Path | str = "./cache",
        device: Any = -1,
        top_k: int = 5,
        truncation: bool = True,
    ):
        self.cache_root = Path(cache_root)
        self.top_k = int(top_k)
        self.emo_clf = pipeline(
            task="text-classification",
            model=self.MODEL,
            top_k=self.top_k,
            device=device,
            truncation=truncation,
        )

    def _base_dir(self) -> Path:
        return self.cache_root / "emotion" / "go_emotions" / "cosrec" / f"top{self.top_k}" / "curated"

    def _ep_path(self, ep: CoSRecCuratedEpisode) -> Path:
        return self._base_dir() / f"{safe_id(ep.topic_id)}.json"

    def has(self, ep: CoSRecCuratedEpisode) -> bool:
        return self._ep_path(ep).exists()

    def get(self, ep: CoSRecCuratedEpisode) -> Dict[str, Any]:
        p = self._ep_path(ep)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        text = ep.next_user_text or ""
        out = self.emo_clf(text)

        # HF pipeline returns list[dict] for single string when top_k>1
        preds = [{"label": str(d["label"]), "score": float(d["score"])} for d in (out or [])]

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
            "top_k": self.top_k,
            "emotion": preds,
            "next_user_text": text,
        }

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return rec

    def build(
        self,
        ds,
        max_new: Optional[int] = None,
        progress_path: Optional[Path | str] = None,
        every: int = 25,
    ) -> None:
        progress_p = Path(progress_path) if progress_path is not None else None
        processed = 0
        computed = 0
        t0 = time.time()

        for ep in ds.iter_curated_item_episodes(emotion=self):
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
                            "signal": "emotion",
                            "dataset": "cosrec",
                            "partition": "curated",
                            "processed_episodes": processed,
                            "computed_new": computed,
                            "elapsed_s": time.time() - t0,
                        },
                        f,
                        ensure_ascii=False,
                    )
