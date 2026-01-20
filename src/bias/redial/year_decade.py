from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset.movielens import MovieLens25M
from dataset.redial import ReDialConversation, extract_movie_ids, get_speaker, safe_id
from bias.metrics import entropy, js_divergence_dict


class YearDecadeBias:
    def __init__(self, cache_root: Path | str = "./cache", movielens: Optional[MovieLens25M] = None):
        self.cache_root = Path(cache_root)
        self.ml = movielens or MovieLens25M(cache_root=self.cache_root)
        self.ml.ensure_year_baseline()
        self.year_baseline = self.ml.year_baseline()
        self.decade_baseline = self.ml.decade_baseline()

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "year_decade" / "redial" / "ml-25m"

    def _conv_path(self, conv: ReDialConversation) -> Path:
        return self._base_dir() / conv.split / f"{safe_id(conv.conversation_id)}.json"

    def has(self, conv: ReDialConversation) -> bool:
        return self._conv_path(conv).exists()

    def get(self, conv: ReDialConversation) -> Dict[str, Any]:
        p = self._conv_path(conv)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        rec = self._compute(conv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        return rec

    def _dist_from_labels(self, labels: List[str]) -> Dict[str, float]:
        counts = defaultdict(int)
        total = 0
        for lab in labels:
            if not lab:
                continue
            counts[lab] += 1
            total += 1
        if total == 0:
            return {}
        return {k: counts[k] / total for k in counts}

    def _compute(self, conv: ReDialConversation) -> Dict[str, Any]:
        turns: List[Dict[str, Any]] = []
        conv_years: List[int] = []
        conv_decades: List[str] = []

        for i, msg in enumerate(conv.messages):
            raw_text = msg.get("text", "") if isinstance(msg, dict) else ""
            speaker = get_speaker(msg, i)
            redial_ids = extract_movie_ids(raw_text)

            ml_ids: List[int] = []
            years: List[int] = []
            decades: List[str] = []

            for rid in redial_ids:
                title = (conv.movie_mentions or {}).get(str(rid))
                if not title:
                    continue
                mid = self.ml.map_title(title)
                if mid is None:
                    continue
                ml_ids.append(int(mid))
                year = self.ml.year_for_movie_id(int(mid))
                if year is None:
                    continue
                y = int(year)
                years.append(y)
                decades.append(f"{(y // 10) * 10}s")

            bias_obj = None
            if speaker == "Recommender" and years:
                conv_years.extend(years)
                conv_decades.extend(decades)

                year_dist = self._dist_from_labels([str(y) for y in years])
                decade_dist = self._dist_from_labels(decades)

                bias_obj = {
                    "years_dist": year_dist,
                    "years_entropy": entropy(year_dist) if year_dist else 0.0,
                    "years_js_divergence_vs_catalog": js_divergence_dict(year_dist, self.year_baseline)
                    if year_dist and self.year_baseline
                    else None,
                    "year_coverage": int(len(year_dist)),
                    "mean_year": float(sum(years) / len(years)),
                    "median_year": float(statistics.median(years)),
                    "decades_dist": decade_dist,
                    "decades_entropy": entropy(decade_dist) if decade_dist else 0.0,
                    "decades_js_divergence_vs_catalog": js_divergence_dict(decade_dist, self.decade_baseline)
                    if decade_dist and self.decade_baseline
                    else None,
                    "decade_coverage": int(len(decade_dist)),
                }

            turns.append(
                {
                    "msg_idx": i,
                    "message_id": f"{conv.split}:{conv.conversation_id}:{i}",
                    "speaker": speaker,
                    "movielens_movie_ids": ml_ids,
                    "movie_years": years,
                    "movie_decades": decades,
                    "year_decade_bias": bias_obj,
                }
            )

        conv_year_dist = self._dist_from_labels([str(y) for y in conv_years])
        conv_decade_dist = self._dist_from_labels(conv_decades)

        summary = {
            "total_items": int(len(conv_years)),
            "unique_items": int(len(set(conv_years))),
            "years_dist": conv_year_dist,
            "years_entropy": entropy(conv_year_dist) if conv_year_dist else 0.0,
            "years_js_divergence_vs_catalog": js_divergence_dict(conv_year_dist, self.year_baseline)
            if conv_year_dist and self.year_baseline
            else None,
            "year_coverage": int(len(conv_year_dist)),
            "mean_year": float(sum(conv_years) / len(conv_years)) if conv_years else None,
            "median_year": float(statistics.median(conv_years)) if conv_years else None,
            "decades_dist": conv_decade_dist,
            "decades_entropy": entropy(conv_decade_dist) if conv_decade_dist else 0.0,
            "decades_js_divergence_vs_catalog": js_divergence_dict(conv_decade_dist, self.decade_baseline)
            if conv_decade_dist and self.decade_baseline
            else None,
            "decade_coverage": int(len(conv_decade_dist)),
        }

        return {
            "dataset": "redial",
            "split": conv.split,
            "conversationId": conv.conversation_id,
            "movielens_version": "ml-25m",
            "created_at_unix": time.time(),
            "turns": turns,
            "summary": summary,
        }

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

            if progress_p is not None and every and (processed % every == 0):
                progress_p.parent.mkdir(parents=True, exist_ok=True)
                with open(progress_p, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "bias": "year_decade",
                            "split": split,
                            "processed_conversations": processed,
                            "computed_new_conversations": computed,
                            "elapsed_s": time.time() - t0,
                        },
                        f,
                        ensure_ascii=False,
                    )
