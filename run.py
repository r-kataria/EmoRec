#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run emotion (seeker responses) then bias (recommendations) for ReDial/CoSRec."
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--emotions", action="store_true")
    parser.add_argument("--biases", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cache-root", default=str(ROOT / "cache"))
    parser.add_argument("--device", default="auto")
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
    parser.add_argument("--emotion-dataset", choices=["redial", "cosrec", "all"], default="all")
    parser.add_argument("--bias-dataset", choices=["redial", "cosrec", "all"], default="all")
    args = parser.parse_args()

    if args.all:
        args.download = True
        args.emotions = True
        args.biases = True

    if not (args.download or args.emotions or args.biases):
        args.emotions = True
        args.biases = True

    cache_root = Path(args.cache_root)
    scripts_dir = ROOT / "scripts"

    if args.download:
        dl_mod = _load_module(scripts_dir / "1_download_data.py", "download_data_module")
        dl_mod.download_all()

    if args.emotions:
        emo_mod = _load_module(scripts_dir / "2_build_emotions.py", "build_emotions_module")
        emo_mod.build_emotions(
            cache_root=cache_root,
            dataset=args.emotion_dataset,
            device=args.device,
            top_k=args.top_k,
            truncation=not args.no_truncation,
            resolve_movie_titles=not args.raw_movie_tags,
            max_length=args.max_length,
            batch_size=args.batch_size,
            splits=args.splits,
            intent_type=args.intent_type,
            min_relevance=args.min_relevance,
            max_new=args.max_new,
            every=args.every,
            force=args.force,
        )

    if args.biases:
        bias_mod = _load_module(scripts_dir / "3_build_biases.py", "build_biases_module")
        bias_mod.build_biases(
            cache_root=cache_root,
            dataset=args.bias_dataset,
            device=args.device,
            truncation=not args.no_truncation,
            max_length=args.max_length,
            batch_size=args.batch_size,
            splits=args.splits,
            intent_type=args.intent_type,
            min_relevance=args.min_relevance,
            max_new=args.max_new,
            every=args.every,
            force=args.force,
        )


if __name__ == "__main__":
    main()
