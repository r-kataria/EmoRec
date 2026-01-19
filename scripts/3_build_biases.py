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


def build_redial_biases(
    cache_root: Path | str = "./cache",
    device: str | int = "auto",
    top_k: Optional[int | str] = 1,
    truncation: bool = True,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    splits: Iterable[str] | str = ("train", "test"),
    max_new: Optional[int] = None,
    every: int = 200,
    force: bool = False,
) -> None:
    from dataset import MovieLens25M, ReDialDataset
    from bias import (
        ExposureConcentration,
        EpisodePopularityBias,
        GenreBias,
        PopularityBias,
        RedundancyBias,
        StereotypeBiasReDial,
        YearDecadeBias,
    )

    cache_root = Path(cache_root)
    ds = ReDialDataset(cache_root=cache_root)
    ml = MovieLens25M(cache_root=cache_root)

    pop = PopularityBias(cache_root=cache_root, movielens=ml)
    epi = EpisodePopularityBias(cache_root=cache_root, movielens=ml)
    red = RedundancyBias(cache_root=cache_root, movielens=ml)
    gen = GenreBias(cache_root=cache_root, movielens=ml)
    exp = ExposureConcentration(cache_root=cache_root, movielens=ml)
    yd = YearDecadeBias(cache_root=cache_root, movielens=ml)
    stereo = StereotypeBiasReDial(
        cache_root=cache_root,
        device=_parse_device(device),
        top_k=_parse_top_k(top_k),
        truncation=truncation,
        max_length=max_length,
        batch_size=batch_size,
    )

    for split in _normalize_splits(splits):
        if force:
            for base_dir in [
                pop._base_dir(),
                epi._base_dir(),
                red._base_dir(),
                gen._base_dir(),
                yd._base_dir(),
                stereo._base_dir(),
            ]:
                split_dir = base_dir / split
                if split_dir.exists():
                    shutil.rmtree(split_dir)

        pop.build(
            ds,
            split=split,
            max_new=max_new,
            progress_path=cache_root / f"bias_popularity_{split}.json",
            every=every,
        )
        epi.build(
            ds,
            split=split,
            max_new=max_new,
            progress_path=cache_root / f"bias_episode_popularity_{split}.json",
            every=every,
        )
        red.build(
            ds,
            split=split,
            max_new=max_new,
            progress_path=cache_root / f"bias_redundancy_{split}.json",
            every=every,
        )
        gen.build(
            ds,
            split=split,
            max_new=max_new,
            progress_path=cache_root / f"bias_genre_{split}.json",
            every=every,
        )
        yd.build(
            ds,
            split=split,
            max_new=max_new,
            progress_path=cache_root / f"bias_year_decade_{split}.json",
            every=every,
        )
        stereo.build(
            ds,
            split=split,
            max_new=max_new,
            progress_path=cache_root / f"bias_stereotype_{split}.json",
            every=every,
        )
        exp.build(
            ds,
            split=split,
            force=force,
            progress_path=cache_root / f"bias_exposure_{split}.json",
            every=every,
        )


def build_cosrec_biases(
    cache_root: Path | str = "./cache",
    device: str | int = "auto",
    top_k: Optional[int | str] = 1,
    truncation: bool = True,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    intent_type: str = "recommendation",
    min_relevance: int = 1,
    max_new: Optional[int] = None,
    every: int = 25,
    force: bool = False,
) -> None:
    from dataset import AmazonReviews2023Index, CoSRecDataset
    from bias import (
        ExposureConcentrationCoSRec,
        EpisodePopularityBiasCoSRec,
        GenreBiasCoSRec,
        PopularityBiasCoSRec,
        RatingBiasCoSRec,
        RedundancyBiasCoSRec,
        StereotypeBiasCoSRec,
    )

    cache_root = Path(cache_root)
    ds = CoSRecDataset(cache_root=cache_root)
    amazon = AmazonReviews2023Index(cache_root=cache_root, cosrec=ds)
    amazon.ensure_index()

    pop = PopularityBiasCoSRec(cache_root=cache_root, amazon_index=amazon)
    epi = EpisodePopularityBiasCoSRec(cache_root=cache_root, amazon_index=amazon)
    rating = RatingBiasCoSRec(cache_root=cache_root, amazon_index=amazon)
    genre = GenreBiasCoSRec(cache_root=cache_root, amazon_index=amazon)
    red = RedundancyBiasCoSRec(cache_root=cache_root)
    exp = ExposureConcentrationCoSRec(cache_root=cache_root)
    stereo = StereotypeBiasCoSRec(
        cache_root=cache_root,
        device=_parse_device(device),
        truncation=truncation,
        max_length=max_length,
        batch_size=batch_size,
    )

    if force:
        for base_dir in [
            pop._base_dir(),
            epi._base_dir(),
            rating._base_dir(),
            genre._base_dir(),
            red._base_dir(),
            stereo._base_dir(),
        ]:
            if base_dir.exists():
                shutil.rmtree(base_dir)

    pop.build(
        ds,
        intent_type=intent_type,
        min_relevance=min_relevance,
        max_new=max_new,
        progress_path=cache_root / "bias_popularity_cosrec.json",
        every=every,
    )
    epi.build(
        ds,
        intent_type=intent_type,
        min_relevance=min_relevance,
        max_new=max_new,
        progress_path=cache_root / "bias_episode_popularity_cosrec.json",
        every=every,
    )
    rating.build(
        ds,
        intent_type=intent_type,
        min_relevance=min_relevance,
        max_new=max_new,
        progress_path=cache_root / "bias_rating_cosrec.json",
        every=every,
    )
    genre.build(
        ds,
        intent_type=intent_type,
        min_relevance=min_relevance,
        max_new=max_new,
        progress_path=cache_root / "bias_genre_cosrec.json",
        every=every,
    )
    red.build(
        ds,
        intent_type=intent_type,
        min_relevance=min_relevance,
        max_new=max_new,
        progress_path=cache_root / "bias_redundancy_cosrec.json",
        every=every,
    )
    stereo.build(
        ds,
        intent_type=intent_type,
        min_relevance=min_relevance,
        max_new=max_new,
        progress_path=cache_root / "bias_stereotype_cosrec.json",
        every=every,
    )
    exp.build(
        ds,
        intent_type=intent_type,
        min_relevance=min_relevance,
        max_new=max_new,
        force=force,
        progress_path=cache_root / "bias_exposure_cosrec.json",
        every=every,
    )


def build_biases(
    cache_root: Path | str = "./cache",
    dataset: str = "all",
    device: str | int = "auto",
    top_k: Optional[int | str] = 1,
    truncation: bool = True,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    splits: Iterable[str] | str = ("train", "test"),
    intent_type: str = "recommendation",
    min_relevance: int = 1,
    max_new: Optional[int] = None,
    every: int = 200,
    force: bool = False,
) -> None:
    dataset = str(dataset or "all").strip().lower()
    if dataset in {"redial", "all"}:
        build_redial_biases(
            cache_root=cache_root,
            device=device,
            truncation=truncation,
            max_length=max_length,
            batch_size=batch_size,
            splits=splits,
            max_new=max_new,
            every=every,
            force=force,
        )
    if dataset in {"cosrec", "all"}:
        build_cosrec_biases(
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
