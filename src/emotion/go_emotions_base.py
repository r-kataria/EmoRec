from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from utils.hf_pipeline import HFTextClassifier


class GoEmotionsBase(HFTextClassifier):
    MODEL = "SamLowe/roberta-base-go_emotions"

    def __init__(
        self,
        cache_root: Path | str = "./cache",
        device: Any = "auto",
        top_k: Optional[int] = 5,
        truncation: bool = True,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__(
            cache_root=cache_root,
            model=self.MODEL,
            tokenizer=None,
            device=device,
            top_k=top_k,
            truncation=truncation,
            max_length=max_length,
            batch_size=batch_size,
        )
