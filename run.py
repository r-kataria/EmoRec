#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EmoRecc pipeline.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build_emotion", action="store_true")
    parser.add_argument("--build_bias", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if not (args.download or args.build_emotion or args.build_bias or args.stats):
        args.download = True
        args.build_emotion = True
        args.build_bias = True
        args.stats = True

    cache_root = ROOT / "cache"
    results_dir = ROOT / "results"
    scripts_dir = ROOT / "scripts"

    if args.download:
        dl_mod = _load_module(scripts_dir / "1_download_data.py", "download_data_module")
        dl_mod.download_all()

    if args.build_emotion:
        emo_mod = _load_module(scripts_dir / "2_build_emotions.py", "build_emotions_module")
        emo_mod.build_emotions(cache_root=cache_root)

    if args.build_bias:
        bias_mod = _load_module(scripts_dir / "3_build_biases.py", "build_biases_module")
        bias_mod.build_biases(cache_root=cache_root)

    if args.stats:
        stats_mod = _load_module(scripts_dir / "4_analyze_stats.py", "analyze_stats_module")
        stats_mod.analyze_stats(cache_root=cache_root, out_dir=results_dir)


if __name__ == "__main__":
    main()
