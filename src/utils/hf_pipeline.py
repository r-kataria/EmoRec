from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import pipeline


class HFTextClassifier:
    def __init__(
        self,
        cache_root: Path | str = "./cache",
        model: str = "",
        tokenizer: Optional[str] = None,
        device: Any = "auto",
        top_k: Optional[int] = None,
        truncation: bool = True,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.cache_root = Path(cache_root)
        self.top_k = None if top_k is None else int(top_k)
        self.batch_size = int(batch_size) if batch_size else None

        device = self._resolve_device(device)
        pipe_kwargs = {
            "task": "text-classification",
            "model": model,
            "top_k": self.top_k,
            "device": device,
            "truncation": truncation,
        }
        if tokenizer is not None:
            pipe_kwargs["tokenizer"] = tokenizer
        if max_length is not None:
            pipe_kwargs["max_length"] = int(max_length)
        self._pipeline = pipeline(**pipe_kwargs)

    def _top_dir(self) -> str:
        return f"top{self.top_k}" if self.top_k is not None else "top_all"

    @staticmethod
    def _resolve_device(device: Any) -> Any:
        if device is None:
            mode = "auto"
        elif isinstance(device, str):
            mode = device.strip().lower()
        else:
            mode = ""

        if mode in {"auto", "mps"}:
            try:
                import torch  # type: ignore
            except Exception:
                return -1 if mode == "auto" else device
            if hasattr(torch, "device") and torch.backends.mps.is_available():
                return torch.device("mps")
            return -1 if mode == "auto" else device
        return device

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception:
            return ""

    @staticmethod
    def _score_value(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _normalize_preds(self, preds: Any) -> List[Dict[str, float]]:
        if preds is None:
            return []
        if isinstance(preds, dict):
            return [
                {
                    "label": str(preds.get("label", "")),
                    "score": self._score_value(preds.get("score", 0.0)),
                }
            ]
        if isinstance(preds, list):
            out: List[Dict[str, float]] = []
            for d in preds:
                if not isinstance(d, dict):
                    continue
                out.append(
                    {
                        "label": str(d.get("label", "")),
                        "score": self._score_value(d.get("score", 0.0)),
                    }
                )
            out.sort(key=lambda x: x["score"], reverse=True)
            return out
        return []

    def _normalize_batch_output(self, outs: Any, n_inputs: int) -> List[List[Dict[str, float]]]:
        if n_inputs <= 0:
            return []
        if n_inputs == 1:
            if isinstance(outs, list) and len(outs) == 1:
                return [self._normalize_preds(outs[0])]
            return [self._normalize_preds(outs)]
        if isinstance(outs, list) and len(outs) == n_inputs:
            return [self._normalize_preds(o) for o in outs]
        first = self._normalize_preds(outs)
        return [first] + [[] for _ in range(n_inputs - 1)]

    def _run_pipeline(self, texts: List[str]) -> Any:
        if self.batch_size:
            return self._pipeline(texts, batch_size=self.batch_size)
        return self._pipeline(texts)

    def _predict_texts(self, texts: List[str]) -> List[List[Dict[str, float]]]:
        if not texts:
            return []
        normalized = [self._coerce_text(t) for t in texts]
        results: List[List[Dict[str, float]]] = [[] for _ in normalized]

        indexed = [(i, t) for i, t in enumerate(normalized) if t and t.strip()]
        if not indexed:
            return results

        idxs = [i for i, _ in indexed]
        non_empty = [t for _, t in indexed]

        chunk_size = self.batch_size or len(non_empty)
        for start in range(0, len(non_empty), chunk_size):
            chunk_texts = non_empty[start : start + chunk_size]
            outs = self._run_pipeline(chunk_texts)
            preds = self._normalize_batch_output(outs, len(chunk_texts))
            for offset, pred in enumerate(preds):
                results[idxs[start + offset]] = pred
        return results
