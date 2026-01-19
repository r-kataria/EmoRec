#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional


def _parse_device(value):
    if isinstance(value, int):
        return value
    if value is None:
        return "auto"
    v = str(value).strip()
    if v.lower() in {"auto", "mps"}:
        return v.lower()
    try:
        return int(v)
    except Exception:
        return v


def _parse_top_k(value: Optional[str | int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    v = str(value).strip().lower()
    if v in {"all", "none"}:
        return None
    return int(v)


def _normalize_splits(splits: Iterable[str] | str) -> list[str]:
    if isinstance(splits, str):
        return [s.strip() for s in splits.split(",") if s.strip()]
    return [str(s).strip() for s in splits if str(s).strip()]


def build_redial_emotions(
    cache_root: Path | str = "./cache",
    device: str | int = "auto",
    top_k: Optional[int | str] = 5,
    truncation: bool = True,
    resolve_movie_titles: bool = True,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    splits: Iterable[str] | str = ("train", "test"),
    max_new: Optional[int] = None,
    every: int = 200,
    force: bool = False,
) -> None:
    from dataset import ReDialDataset
    from emotion.go_emotions_redial import GoEmotionsReDial

    cache_root = Path(cache_root)
    ds = ReDialDataset(cache_root=cache_root)
    emo = GoEmotionsReDial(
        cache_root=cache_root,
        device=_parse_device(device),
        top_k=_parse_top_k(top_k),
        truncation=truncation,
        resolve_movie_titles=resolve_movie_titles,
        max_length=max_length,
        batch_size=batch_size,
    )

    for split in _normalize_splits(splits):
        if force:
            split_dir = emo._base_dir() / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
        emo.build(
            ds,
            split=split,
            max_new=max_new,
            progress_path=cache_root / f"emotion_progress_redial_{split}.json",
            every=every,
        )


def build_cosrec_emotions(
    cache_root: Path | str = "./cache",
    device: str | int = "auto",
    top_k: Optional[int | str] = 5,
    truncation: bool = True,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    intent_type: str = "recommendation",
    min_relevance: int = 1,
    max_new: Optional[int] = None,
    every: int = 25,
    force: bool = False,
) -> None:
    from dataset import CoSRecDataset
    from emotion.go_emotions_cosrec import GoEmotionsCoSRec

    cache_root = Path(cache_root)
    ds = CoSRecDataset(cache_root=cache_root)
    emo = GoEmotionsCoSRec(
        cache_root=cache_root,
        device=_parse_device(device),
        top_k=_parse_top_k(top_k),
        truncation=truncation,
        max_length=max_length,
        batch_size=batch_size,
    )

    if force and emo._base_dir().exists():
        shutil.rmtree(emo._base_dir())

    emo.build(
        ds,
        intent_type=intent_type,
        min_relevance=min_relevance,
        max_new=max_new,
        progress_path=cache_root / "emotion_progress_cosrec.json",
        every=every,
    )


def build_cosrec_turn_emotions(
    cache_root: Path | str = "./cache",
    device: str | int = "auto",
    top_k: Optional[int | str] = 5,
    truncation: bool = True,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_new: Optional[int] = None,
    every: int = 25,
    force: bool = False,
) -> None:
    from dataset import CoSRecDataset
    from emotion.go_emotions_cosrec import GoEmotionsCoSRec

    cache_root = Path(cache_root)
    ds = CoSRecDataset(cache_root=cache_root)
    emo = GoEmotionsCoSRec(
        cache_root=cache_root,
        device=_parse_device(device),
        top_k=_parse_top_k(top_k),
        truncation=truncation,
        max_length=max_length,
        batch_size=batch_size,
    )

    if force and emo._turn_base_dir().exists():
        shutil.rmtree(emo._turn_base_dir())

    emo.build_turns(
        ds,
        partition="curated",
        max_new=max_new,
        progress_path=cache_root / "emotion_progress_cosrec_turns.json",
        every=every,
    )


def build_emotions(
    cache_root: Path | str = "./cache",
    dataset: str = "all",
    device: str | int = "auto",
    top_k: Optional[int | str] = 5,
    truncation: bool = True,
    resolve_movie_titles: bool = True,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    splits: Iterable[str] | str = ("train", "test"),
    intent_type: str = "recommendation",
    min_relevance: int = 1,
    max_new: Optional[int] = None,
    every: int = 200,
    include_cosrec_turns: bool = True,
    force: bool = False,
) -> None:
    dataset = str(dataset or "all").strip().lower()
    if dataset in {"redial", "all"}:
        build_redial_emotions(
            cache_root=cache_root,
            device=device,
            top_k=top_k,
            truncation=truncation,
            resolve_movie_titles=resolve_movie_titles,
            max_length=max_length,
            batch_size=batch_size,
            splits=splits,
            max_new=max_new,
            every=every,
            force=force,
        )
    if dataset in {"cosrec", "all"}:
        build_cosrec_emotions(
            cache_root=cache_root,
            device=device,
            top_k=top_k,
            truncation=truncation,
            max_length=max_length,
            batch_size=batch_size,
            intent_type=intent_type,
            min_relevance=min_relevance,
            max_new=max_new,
            every=25 if every is None else every,
            force=force,
        )
        if include_cosrec_turns:
            build_cosrec_turn_emotions(
                cache_root=cache_root,
                device=device,
                top_k=top_k,
                truncation=truncation,
                max_length=max_length,
                batch_size=batch_size,
                max_new=max_new,
                every=25 if every is None else every,
                force=force,
            )
