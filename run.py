#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _parse_device(value: str):
    v = str(value).strip()
    if v.lower() == "mps":
        return "mps"
    try:
        return int(v)
    except Exception:
        return v


def _parse_top_k(value: str):
    v = str(value).strip().lower()
    if v in {"all", "none"}:
        return None
    return int(v)


def _run_script(path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(path)] + args
    subprocess.run(cmd, check=True)


def _has_json(path: Path) -> bool:
    if not path.exists():
        return False
    return any(path.rglob("*.json"))


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _parse_splits(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _build_biases(args, cache_root: Path) -> None:
    from dataset import MovieLens25M, ReDialDataset
    from bias import (
        ExposureConcentration,
        GenreBias,
        PopularityBias,
        RedundancyBias,
        StereotypeBiasCoSRec,
        StereotypeBiasReDial,
    )

    device = _parse_device(args.device)
    top_k = _parse_top_k(args.top_k)

    if args.bias_dataset in {"redial", "all"}:
        ds = ReDialDataset(cache_root=cache_root)
        ml = MovieLens25M(cache_root=cache_root)

        pop = PopularityBias(cache_root=cache_root, movielens=ml)
        red = RedundancyBias(cache_root=cache_root, movielens=ml)
        gen = GenreBias(cache_root=cache_root, movielens=ml)
        exp = ExposureConcentration(cache_root=cache_root, movielens=ml)
        stereo = StereotypeBiasReDial(
            cache_root=cache_root,
            device=device,
            top_k=top_k,
            truncation=not args.no_truncation,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

        splits = _parse_splits(args.splits)
        for split in splits:
            for name, bias_obj in [("popularity", pop), ("redundancy", red), ("genre", gen)]:
                split_dir = bias_obj._base_dir() / split
                if _has_json(split_dir) and not args.force:
                    print(f"[bias:{name}] cache exists for split={split}, skipping")
                    continue
                if args.force:
                    _clear_dir(split_dir)
                bias_obj.build(
                    ds,
                    split=split,
                    max_new=args.max_new,
                    progress_path=cache_root / f"bias_{name}_{split}.json",
                    every=args.every,
                )

            out_path = exp._out_path(split)
            if out_path.exists() and not args.force:
                print(f"[bias:exposure] cache exists for split={split}, skipping")
                continue
            exp.build(
                ds,
                split=split,
                force=args.force,
                progress_path=cache_root / f"bias_exposure_{split}.json",
                every=args.every,
            )

            stereo_dir = stereo._base_dir() / split
            if _has_json(stereo_dir) and not args.force:
                print(f"[bias:stereotype] cache exists for split={split}, skipping")
                continue
            if args.force:
                _clear_dir(stereo_dir)
            stereo.build(
                ds,
                split=split,
                max_new=args.max_new,
                progress_path=cache_root / f"bias_stereotype_{split}.json",
                every=args.every,
            )

    if args.bias_dataset in {"cosrec", "all"}:
        from dataset import CoSRecDataset

        ds_cos = CoSRecDataset(cache_root=cache_root)
        stereo_cos = StereotypeBiasCoSRec(
            cache_root=cache_root,
            device=device,
            top_k=top_k,
            truncation=not args.no_truncation,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        base_dir = stereo_cos._base_dir()
        if _has_json(base_dir) and not args.force:
            print("[bias:stereotype] cache exists for cosrec, skipping")
            return
        if args.force:
            _clear_dir(base_dir)
        stereo_cos.build(
            ds_cos,
            intent_type=args.intent_type,
            min_relevance=args.min_relevance,
            max_new=args.max_new,
            progress_path=cache_root / "bias_stereotype_cosrec.json",
            every=args.every,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run emotion (seeker responses) then bias (recommendations) for ReDial/CoSRec."
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--emotions", action="store_true")
    parser.add_argument("--biases", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cache-root", default=str(ROOT / "cache"))
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
        _run_script(scripts_dir / "1_download_data.py", [])

    if args.emotions:
        emo_args = [
            "--cache-root",
            str(cache_root),
            "--dataset",
            args.emotion_dataset,
            "--device",
            str(args.device),
            "--top-k",
            str(args.top_k),
            "--splits",
            args.splits,
            "--intent-type",
            args.intent_type,
            "--min-relevance",
            str(args.min_relevance),
            "--every",
            str(args.every),
        ]
        if args.batch_size is not None:
            emo_args += ["--batch-size", str(args.batch_size)]
        if args.max_length is not None:
            emo_args += ["--max-length", str(args.max_length)]
        if args.no_truncation:
            emo_args.append("--no-truncation")
        if args.raw_movie_tags:
            emo_args.append("--raw-movie-tags")
        if args.max_new is not None:
            emo_args += ["--max-new", str(args.max_new)]
        if args.force:
            emo_args.append("--force")
        _run_script(scripts_dir / "2_build_emotions.py", emo_args)

    if args.biases:
        _build_biases(args, cache_root)


if __name__ == "__main__":
    main()
