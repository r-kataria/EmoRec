from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dataset.cosrec import CoSRecConversation, CoSRecRecEpisode, safe_id
from emotion.go_emotions_base import GoEmotionsBase
from utils.progress import write_progress


class GoEmotionsCoSRec(GoEmotionsBase):
    def __init__(
        self,
        cache_root: Path | str = "./cache",
        device: Any = -1,
        top_k: int = 5,
        truncation: bool = True,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__(
            cache_root=cache_root,
            device=device,
            top_k=top_k,
            truncation=truncation,
            max_length=max_length,
            batch_size=batch_size,
        )

    def _base_dir(self) -> Path:
        return self.cache_root / "emotion" / "go_emotions" / "cosrec" / self._top_dir() / "curated"

    def _turn_base_dir(self) -> Path:
        return self.cache_root / "emotion" / "go_emotions" / "cosrec" / self._top_dir() / "curated_turns"

    def _ep_path(self, ep: CoSRecRecEpisode) -> Path:
        return self._base_dir() / f"{safe_id(ep.topic_id)}.json"

    def _conv_path(self, conv: CoSRecConversation) -> Path:
        return self._turn_base_dir() / f"{safe_id(conv.conversation_id)}.json"

    def has(self, ep: CoSRecRecEpisode) -> bool:
        return self._ep_path(ep).exists()

    def has_turns(self, conv: CoSRecConversation) -> bool:
        return self._conv_path(conv).exists()

    def get(self, ep: CoSRecRecEpisode) -> Dict[str, Any]:
        p = self._ep_path(ep)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        text = self._coerce_text(ep.next_user_text)
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
            "top_k": self.top_k,
            "emotion": preds,
            "next_user_text": text,
        }

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return rec

    def get_turns(self, conv: CoSRecConversation) -> Dict[str, Any]:
        p = self._conv_path(conv)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        turns = conv.turns
        user_texts = []
        user_idxs = []
        for i, t in enumerate(turns):
            if t.get("speaker") != "U":
                continue
            user_idxs.append(i)
            user_texts.append(self._coerce_text(t.get("text", "")))

        preds = self._predict_texts(user_texts)
        by_idx = {user_idxs[i]: preds[i] for i in range(len(user_idxs))}

        rec = {
            "dataset": "cosrec",
            "partition": "curated",
            "conversation_id": conv.conversation_id,
            "created_at_unix": time.time(),
            "model": self.MODEL,
            "top_k": self.top_k,
            "turns": [
                {
                    "turn_idx": i,
                    "speaker": t.get("speaker"),
                    "text": t.get("text", ""),
                    "emotion": by_idx.get(i),
                }
                for i, t in enumerate(turns)
            ],
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
            emotion=self,
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
                    "signal": "emotion",
                    "dataset": "cosrec",
                    "partition": "curated",
                    "processed_episodes": processed,
                    "computed_new": computed,
                    "elapsed_s": time.time() - t0,
                })
                print(
                    f"[emotion:cosrec] processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
                    flush=True,
                )

        print(
            f"[emotion:cosrec] done processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )

    def build_turns(
        self,
        ds,
        partition: str = "curated",
        max_new: Optional[int] = None,
        progress_path: Optional[Path | str] = None,
        every: int = 25,
    ) -> None:
        progress_p = Path(progress_path) if progress_path is not None else None
        processed = 0
        computed = 0
        t0 = time.time()

        for conv in ds.iter_conversations(partition):
            processed += 1
            if not self.has_turns(conv):
                _ = self.get_turns(conv)
                computed += 1
                if max_new is not None and computed >= max_new:
                    break

            if every and (processed % every == 0):
                write_progress(progress_p, {
                    "signal": "emotion",
                    "dataset": "cosrec",
                    "partition": partition,
                    "scope": "all_user_turns",
                    "processed_conversations": processed,
                    "computed_new": computed,
                    "elapsed_s": time.time() - t0,
                })
                print(
                    f"[emotion:cosrec:turns] processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
                    flush=True,
                )

        print(
            f"[emotion:cosrec:turns] done processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )
