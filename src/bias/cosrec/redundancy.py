from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from dataset.cosrec import CoSRecRecEpisode, safe_id
from utils.progress import write_progress


class RedundancyBiasCoSRec:
    def __init__(self, cache_root: Path | str = "./cache"):
        self.cache_root = Path(cache_root)

    def _base_dir(self) -> Path:
        return self.cache_root / "bias" / "redundancy" / "cosrec" / "amazon_2023" / "curated"

    def _ep_path(self, ep: CoSRecRecEpisode) -> Path:
        return self._base_dir() / f"{safe_id(ep.topic_id)}.json"

    def has(self, ep: CoSRecRecEpisode) -> bool:
        return self._ep_path(ep).exists()

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

        by_conv: Dict[str, List[CoSRecRecEpisode]] = defaultdict(list)
        for ep in ds.iter_rec_episodes(min_relevance=min_relevance, bias=self):
            if intent_type and ep.intent_type != intent_type:
                continue
            by_conv[str(ep.conversation_id)].append(ep)

        stop = False
        for _, eps in by_conv.items():
            if stop:
                break
            eps.sort(key=lambda e: (int(e.utterance_idx), int(e.user_index), str(e.topic_id)))
            seen: Counter[str] = Counter()
            for ep in eps:
                asins = [str(a) for a, _ in ep.qrels]
                new_items = [a for a in asins if a not in seen]
                repeated_items = [a for a in asins if a in seen]
                for a in asins:
                    seen[a] += 1

                processed += 1
                if not self.has(ep):
                    counts = dict(seen)
                    total = int(sum(counts.values()))
                    unique = int(len(counts))
                    redundancy_rate = float(1.0 - (unique / total)) if total > 0 else 0.0

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
                        "catalogue": "amazon_2023",
                        "asins": asins,
                        "new_items": new_items,
                        "repeated_items": repeated_items,
                        "summary": {
                            "total_items_so_far": total,
                            "unique_items_so_far": unique,
                            "redundancy_rate_so_far": redundancy_rate,
                            "repeated_item_counts": {k: v for k, v in counts.items() if v > 1},
                        },
                    }

                    p = self._ep_path(ep)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(rec, f, ensure_ascii=False)
                    computed += 1
                    if max_new is not None and computed >= max_new:
                        stop = True
                        break

                if every and (processed % every == 0):
                    write_progress(progress_p, {
                        "bias": "redundancy",
                        "dataset": "cosrec",
                        "partition": "curated",
                        "processed_episodes": processed,
                        "computed_new": computed,
                        "elapsed_s": time.time() - t0,
                    })
                    print(
                        f"[bias:redundancy:cosrec] processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
                        flush=True,
                    )

        print(
            f"[bias:redundancy:cosrec] done processed={processed} new={computed} elapsed_s={time.time() - t0:.1f}",
            flush=True,
        )
