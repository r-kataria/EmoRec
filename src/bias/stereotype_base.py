from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from utils.hf_pipeline import HFTextClassifier


# Labels: LABEL_0 = Non-biased, LABEL_1 = Biased.
class StereotypeBiasBase(HFTextClassifier):
    MODEL = "himel7/bias-detector"
    TOKENIZER = "roberta-base"

    def __init__(
        self,
        cache_root: Path | str = "./cache",
        device: Any = "auto",
        top_k: Optional[int] = 1,
        truncation: bool = True,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        limited_k = None if top_k is None else min(int(top_k), 2)
        super().__init__(
            cache_root=cache_root,
            model=self.MODEL,
            tokenizer=self.TOKENIZER,
            device=device,
            top_k=limited_k,
            truncation=truncation,
            max_length=max_length,
            batch_size=batch_size,
        )
