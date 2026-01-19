#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import CoSRecDataset, ReDialDataset
from emotion.go_emotions_cosrec import GoEmotionsCoSRec
from emotion.go_emotions_redial import GoEmotionsReDial


def _parse_device(value: str):
    v = str(value).strip()
    if v.lower() == "mps":
        return "mps"
    try:
        return int(v)
    except Exception:
        return v


def _parse_top_k(value: str) -> Optional[int]:
    v = str(value).strip().lower()
    if v in {"all", "none"}:
        return None
    return int(v)


def _has_json(path: Path) -> bool:
    if not path.exists():
        return False
    return any(path.rglob("*.json"))


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _build_redial(args, cache_root: Path) -> None:
    ds = ReDialDataset(cache_root=cache_root)
    emo = GoEmotionsReDial(
        cache_root=cache_root,
        device=_parse_device(args.device),
        top_k=_parse_top_k(args.top_k),
        truncation=not args.no_truncation,
        resolve_movie_titles=not args.raw_movie_tags,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        split_dir = emo._base_dir() / split
        if _has_json(split_dir) and not args.force:
            print(f"[redial] cache exists for split={split}, skipping")
            continue
        if args.force:
            _clear_dir(split_dir)
        emo.build(
            ds,
            split=split,
            max_new=args.max_new,
            progress_path=cache_root / f"emotion_progress_redial_{split}.json",
            every=args.every,
        )


def _build_cosrec(args, cache_root: Path) -> None:
    ds = CoSRecDataset(cache_root=cache_root)
    emo = GoEmotionsCoSRec(
        cache_root=cache_root,
        device=_parse_device(args.device),
        top_k=_parse_top_k(args.top_k),
        truncation=not args.no_truncation,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    base_dir = emo._base_dir()
    if _has_json(base_dir) and not args.force:
        print("[cosrec] cache exists, skipping")
        return
    if args.force:
        _clear_dir(base_dir)

    emo.build(
        ds,
        intent_type=args.intent_type,
        min_relevance=args.min_relevance,
        max_new=args.max_new,
        progress_path=cache_root / "emotion_progress_cosrec.json",
        every=args.every,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GoEmotions caches for ReDial and CoSRec.")
    parser.add_argument("--cache-root", default=str(ROOT / "cache"))
    parser.add_argument("--dataset", choices=["redial", "cosrec", "all"], default="all")
    parser.add_argument("--device", default="-1")
    parser.add_argument("--top-k", default="5")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--no-truncation", action="store_true")
    parser.add_argument("--raw-movie-tags", action="store_true")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--intent-type", default="recommendation")
    parser.add_argument("--min-relevance", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)

    if args.dataset in {"redial", "all"}:
        _build_redial(args, cache_root)
    if args.dataset in {"cosrec", "all"}:
        _build_cosrec(args, cache_root)


if __name__ == "__main__":
    main()
