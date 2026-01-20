#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coverage_at(percentiles: List[float], cutoff: float) -> Optional[float]:
    if not percentiles:
        return None
    return float(sum(1 for p in percentiles if p >= cutoff) / len(percentiles))


def _rank_utility(percentiles: List[float]) -> Optional[float]:
    if not percentiles:
        return None
    weights = [1.0 / math.log2(i + 2) for i in range(len(percentiles))]
    denom = sum(weights)
    if denom <= 0:
        return None
    return float(sum(w * p for w, p in zip(weights, percentiles)) / denom)


def _gini(counts: Iterable[int]) -> float:
    vals = sorted(int(v) for v in counts if v is not None)
    if not vals:
        return 0.0
    n = len(vals)
    s = sum(vals)
    if s == 0:
        return 0.0
    num = 0.0
    for i, v in enumerate(vals, start=1):
        num += i * v
    return float((2.0 * num) / (n * s) - (n + 1) / n)


def _hhi(counts: Iterable[int]) -> float:
    vals = [int(v) for v in counts if v is not None]
    s = sum(vals)
    if s == 0:
        return 0.0
    return float(sum((v / s) ** 2 for v in vals))


def _emotion_vector(emotions: Any) -> Tuple[Dict[str, float], Optional[str], Optional[float]]:
    if not isinstance(emotions, list):
        return {}, None, None
    scores: Dict[str, float] = {}
    top_label = None
    top_score = None
    for item in emotions:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        try:
            score = float(item.get("score", 0.0))
        except Exception:
            score = 0.0
        scores[label] = score
        if top_score is None or score > top_score:
            top_score = score
            top_label = label
    return scores, top_label, top_score


def _stereotype_scores(stereo_obj: Any) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    score_0 = None
    score_1 = None
    if isinstance(stereo_obj, list):
        for item in stereo_obj:
            if not isinstance(item, dict):
                continue
            lab = item.get("label")
            try:
                score = float(item.get("score", 0.0))
            except Exception:
                score = 0.0
            if lab == "LABEL_0":
                score_0 = score
            elif lab == "LABEL_1":
                score_1 = score
    label = None
    if score_0 is not None or score_1 is not None:
        if score_0 is None:
            label = 1
        elif score_1 is None:
            label = 0
        else:
            label = 1 if score_1 >= score_0 else 0
    return label, score_0, score_1


def _top_dir(base: Path, prefix: str = "top") -> Optional[Path]:
    if not base.exists():
        return None
    best = None
    best_k = -1
    for p in base.iterdir():
        if not p.is_dir() or not p.name.startswith(prefix):
            continue
        suffix = p.name[len(prefix) :]
        try:
            k = int(suffix)
        except Exception:
            continue
        if k > best_k:
            best_k = k
            best = p
    return best


def _load_emotion_labels(records: List[Dict[str, Any]]) -> List[str]:
    labels = set()
    for r in records:
        for k in r:
            if k.startswith("emo_"):
                labels.add(k[len("emo_") :])
    return sorted(labels)


def _emotion_counts(df: pd.DataFrame) -> Dict[str, int]:
    if "emotion_top1" not in df:
        return {}
    counts = df["emotion_top1"].dropna().value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    if len(x) < 3 or len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return float(pr.statistic), float(pr.pvalue), float(sr.correlation), float(sr.pvalue)


def _set_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#e6e6e6",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _pretty_metric_name(metric: str) -> str:
    mapping = {
        "p01": "P@1%",
        "p05": "P@5%",
        "p10": "P@10%",
        "pi": "PI (rank utility)",
        "pop_mean_pct": "Mean popularity percentile",
        "cep": "CEP similarity",
        "uiop": "UIOP similarity",
        "genre_js": "Genre JS divergence",
        "genre_entropy": "Genre entropy",
        "year_decade_js": "Year/decade JS divergence",
        "year_decade_entropy": "Year/decade entropy",
        "mean_year": "Mean release year",
        "rating_mean": "Mean rating",
        "rating_mean_pct": "Mean rating percentile",
        "redundancy_new": "Redundancy: new items",
        "redundancy_repeated": "Redundancy: repeated items",
    }
    return mapping.get(metric, metric)


def _plot_emotion_distribution(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    counts = df["emotion_top1"].value_counts().sort_values(ascending=False)
    if counts.empty:
        return
    total = float(counts.sum())
    if total <= 0:
        return
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [f"{lab} (n={int(cnt)})" for lab, cnt in zip(counts.index.tolist(), counts.values)]
    values = (counts.values / total) * 100.0
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4c78a8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Percent of episodes")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_emotion_distribution_panels(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    counts = df["emotion_top1"].value_counts().sort_values(ascending=False)
    if counts.empty:
        return
    total = float(counts.sum())
    if total <= 0:
        return

    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    if not emo_cols:
        return

    raw_labels = counts.index.tolist()
    labels = [f"{lab} (n={int(cnt)})" for lab, cnt in zip(raw_labels, counts.values)]
    values = (counts.values / total) * 100.0

    score_data = []
    for lab in raw_labels:
        col = f"emo_{lab}"
        if col not in df:
            score_data.append([])
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        score_data.append(vals)

    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ax1, ax2 = axes

    y = np.arange(len(labels))
    ax1.barh(y, values, color="#4c78a8", alpha=0.9)
    ax1.set_title("Top-1 emotion")
    ax1.set_xlabel("Percent of episodes")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.grid(axis="x")

    ax2.violinplot(score_data, showmeans=False, showmedians=True, vert=False)
    ax2.set_title("Score distributions (emo_*)")
    ax2.set_yticks(np.arange(1, len(labels) + 1))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Score")
    ax2.grid(axis="x")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_emotion_cooccurrence(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 3,
) -> None:
    import matplotlib.pyplot as plt

    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    if not emo_cols:
        return

    raw_labels = sorted([c[len("emo_") :] for c in emo_cols])
    idx = {lab: i for i, lab in enumerate(raw_labels)}
    mat = np.zeros((len(raw_labels), len(raw_labels)), dtype=float)

    for _, row in df[emo_cols].iterrows():
        scores = []
        for col in emo_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            try:
                score = float(val)
            except Exception:
                continue
            if score <= 0:
                continue
            scores.append((col[len("emo_") :], score))
        if not scores:
            continue
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [lab for lab, _ in scores[:top_n]]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a = idx[top[i]]
                b = idx[top[j]]
                mat[a, b] += 1
                mat[b, a] += 1

    if not mat.any():
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(mat, cmap="Reds")
    ax.set_title(title)
    counts = _emotion_counts(df)
    display_labels = [f"{lab} (n={counts.get(lab, 0)})" for lab in raw_labels]
    ax.set_xticks(np.arange(len(raw_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(raw_labels)))
    ax.set_yticklabels(display_labels)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Co-occurrence count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_metric_hist(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    series = df[metric].dropna()
    if series.empty:
        return
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(series, bins=20, color="#72b7b2", edgecolor="#2f2f2f", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("Episodes")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_threshold_box(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    cols = ["p01", "p05", "p10"]
    series = [df[c].dropna().to_numpy() if c in df else np.array([]) for c in cols]
    if all(len(s) == 0 for s in series):
        return
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    try:
        ax.boxplot(
            series,
            tick_labels=["P@1%", "P@5%", "P@10%"],
            showfliers=False,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": "#f58518", "alpha": 0.6},
            medianprops={"color": "#222222"},
        )
    except TypeError:
        ax.boxplot(
            series,
            labels=["P@1%", "P@5%", "P@10%"],
            showfliers=False,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": "#f58518", "alpha": 0.6},
            medianprops={"color": "#222222"},
        )
    ax.set_title(title)
    ax.set_ylabel("Coverage distribution")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stereotype_distribution(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if "stereotype_label" not in df:
        return
    counts = df["stereotype_label"].value_counts().sort_index()
    if counts.empty:
        return
    total = float(counts.sum())
    if total <= 0:
        return
    labels = ["Non-biased", "Biased"]
    values = [
        (counts.get(0, 0) / total) * 100.0,
        (counts.get(1, 0) / total) * 100.0,
    ]
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(labels, values, color=["#54a24b", "#e45756"])
    ax.set_title(title)
    ax.set_ylabel("Percent of episodes")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stereotype_by_emotion(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if "stereotype_label" not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", "stereotype_label"]].dropna()
    sub = sub[sub["stereotype_label"].isin([0, 1])]
    if sub.empty:
        return

    counts = (
        sub.groupby(["emotion_top1", "stereotype_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    counts = counts[counts["total"] > 0].sort_values("total", ascending=False)
    if counts.empty:
        return
    overall = sub["stereotype_label"].value_counts().reindex([0, 1], fill_value=0)
    overall_total = int(overall.sum())
    if overall_total > 0:
        overall_row = pd.DataFrame(
            {0: [int(overall.get(0, 0))], 1: [int(overall.get(1, 0))], "total": [overall_total]},
            index=["All"],
        )
        counts = pd.concat([overall_row, counts], axis=0)

    labels = [f"{idx} (n={int(row.total)})" for idx, row in counts.iterrows()]
    p0 = (counts[0] / counts["total"]) * 100.0
    p1 = (counts[1] / counts["total"]) * 100.0

    height = max(4.0, 0.35 * len(labels) + 1.5)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(labels))
    ax.barh(y, p0, color="#54a24b", label="Non-biased")
    ax.barh(y, p1, left=p0, color="#e45756", label="Biased")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Percent of episodes")
    ax.set_title(title)
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_bias_by_emotion_point(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if metric not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    stats_df = sub.groupby("emotion_top1")[metric].agg(["mean", "std", "count"])
    stats_df = stats_df[stats_df["count"] > 0].sort_values("mean", ascending=True)
    if stats_df.empty:
        return

    means = stats_df["mean"].to_numpy()
    counts = stats_df["count"].to_numpy()
    stds = stats_df["std"].fillna(0.0).to_numpy()
    ses = np.where(counts > 1, stds / np.sqrt(counts), 0.0)
    cis = 1.96 * ses
    labels = [f"{idx} (n={int(n)})" for idx, n in zip(stats_df.index.tolist(), counts)]

    height = max(4.0, 0.33 * len(labels) + 1.5)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=cis, fmt="o", color="#4c78a8", ecolor="#999999", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(_pretty_metric_name(metric))
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_violin_by_emotion(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if metric not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    stats_df = sub.groupby("emotion_top1")[metric].agg(["count", "median"])
    stats_df = stats_df[stats_df["count"] > 0].sort_values("median", ascending=True)
    if stats_df.empty:
        return

    labels = [f"{idx} (n={int(row['count'])})" for idx, row in stats_df.iterrows()]
    data = [sub[sub["emotion_top1"] == idx][metric].to_numpy() for idx in stats_df.index.tolist()]

    height = max(4.0, 0.35 * len(labels) + 1.5)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, vert=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c78a8")
        pc.set_alpha(0.7)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel(_pretty_metric_name(metric))
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 500, seed: int = 7) -> Tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    low = float(np.percentile(means, 2.5))
    high = float(np.percentile(means, 97.5))
    return low, high


def _plot_threshold_profile_by_emotion(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    cols = ["p01", "p05", "p10"]
    if any(c not in df for c in cols):
        return
    counts = df["emotion_top1"].dropna().value_counts()
    labels = counts.index.tolist()
    if not labels:
        return

    x = np.array([1, 5, 10], dtype=float)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(labels):
        count = int(counts.get(lab, 0))
        sub = df[df["emotion_top1"] == lab]
        means = []
        lows = []
        highs = []
        for c in cols:
            vals = pd.to_numeric(sub[c], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            means.append(float(np.mean(vals)))
            lo, hi = _bootstrap_mean_ci(vals)
            lows.append(lo)
            highs.append(hi)
        color = cmap(i % 20)
        ax.plot(
            x,
            means,
            marker="o",
            linestyle="-",
            linewidth=1.0,
            color=color,
            alpha=0.7,
            label=f"{lab} (n={count})",
        )
        ax.fill_between(x, lows, highs, color=color, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(["1%", "5%", "10%"])
    ax.set_xlabel("Popularity threshold")
    ax.set_ylabel("Mean coverage")
    ax.set_title(title)
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(labels) / 8)))) if labels else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(labels)))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_score_vs_metric(
    df: pd.DataFrame,
    emotion_label: str,
    metric: str,
    out_path: Path,
    title: str,
    bins: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    col = f"emo_{emotion_label}"
    if col not in df or metric not in df:
        return
    x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    y = pd.to_numeric(df[metric], errors="coerce").to_numpy()
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(x, y, s=8, alpha=0.2, color="#4c78a8")

    edges = np.linspace(0.0, 1.0, bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    means = []
    cis = []
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        vals = y[mask]
        if len(vals) == 0:
            means.append(np.nan)
            cis.append((np.nan, np.nan))
        else:
            means.append(float(np.mean(vals)))
            cis.append(_bootstrap_mean_ci(vals))
    means = np.array(means, dtype=float)
    lows = np.array([c[0] for c in cis], dtype=float)
    highs = np.array([c[1] for c in cis], dtype=float)

    ax.plot(mids, means, color="#e45756", linewidth=2)
    ax.fill_between(mids, lows, highs, color="#e45756", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(f"{emotion_label} score")
    ax.set_ylabel(_pretty_metric_name(metric))
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_popularity_rating_joint(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    use_percentile: bool = True,
    show_quadrants: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    xcol = "pop_mean_pct"
    ycol = "rating_mean_pct" if use_percentile else "rating_mean"
    if xcol not in df or ycol not in df:
        return
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = ~(x.isna() | y.isna())
    if mask.sum() == 0:
        return
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()
    labels = df.loc[mask, "emotion_top1"].fillna("unknown").to_numpy()

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    uniq = sorted(set(labels))
    counts = {lab: int(np.sum(labels == lab)) for lab in uniq}
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(uniq):
        sel = labels == lab
        ax.scatter(
            x[sel],
            y[sel],
            s=14,
            alpha=0.35,
            color=cmap(i % 20),
            label=f"{lab} (n={int(sel.sum())})",
        )
    if show_quadrants:
        x_med = float(np.median(x))
        y_med = float(np.median(y))
        ax.axvline(x_med, color="#999999", linestyle="--", linewidth=1)
        ax.axhline(y_med, color="#999999", linestyle="--", linewidth=1)

    ax.set_xlabel(_pretty_metric_name(xcol))
    ax.set_ylabel(_pretty_metric_name(ycol))
    ax.set_title(title)
    ax.grid(axis="both")
    ncol = min(6, max(2, int(math.ceil(len(uniq) / 8)))) if uniq else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(uniq)))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_quadrant_heatmap(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if xcol not in df or ycol not in df or "emotion_top1" not in df:
        return
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = ~(x.isna() | y.isna())
    if mask.sum() == 0:
        return
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()
    emos = df.loc[mask, "emotion_top1"].fillna("unknown").to_numpy()

    x_med = float(np.median(x))
    y_med = float(np.median(y))
    quadrants = ["low-low", "low-high", "high-low", "high-high"]

    rows = []
    emotions = sorted(set(emos))
    counts = {emo: int(np.sum(emos == emo)) for emo in emotions}
    for emo in emotions:
        sel = emos == emo
        xs = x[sel]
        ys = y[sel]
        if len(xs) == 0:
            continue
        q = [
            np.mean((xs < x_med) & (ys < y_med)),
            np.mean((xs < x_med) & (ys >= y_med)),
            np.mean((xs >= x_med) & (ys < y_med)),
            np.mean((xs >= x_med) & (ys >= y_med)),
        ]
        rows.append(q)

    if not rows:
        return
    mat = np.array(rows, dtype=float)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(5.0, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("Blues").copy()
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(quadrants)))
    ax.set_xticklabels(quadrants, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels([f"{emo} (n={counts.get(emo, 0)})" for emo in emotions])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Proportion")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _metric_stats_by_emotion(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df or "emotion_top1" not in df:
        return pd.DataFrame([])
    sub = df[["emotion_top1", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.DataFrame([])
    stats_df = sub.groupby("emotion_top1")[metric].agg(["mean", "std", "count"])
    stats_df = stats_df[stats_df["count"] > 0]
    return stats_df


def _plot_metric_pair_by_emotion(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    stats_a = _metric_stats_by_emotion(df, metric_a)
    stats_b = _metric_stats_by_emotion(df, metric_b)
    if stats_a.empty and stats_b.empty:
        return

    emotions = set(stats_a.index).union(stats_b.index)
    if not emotions:
        return

    def _count(emotion: str) -> int:
        ca = int(stats_a.loc[emotion, "count"]) if emotion in stats_a.index else 0
        cb = int(stats_b.loc[emotion, "count"]) if emotion in stats_b.index else 0
        return max(ca, cb)

    ordered = sorted(emotions, key=lambda e: (_count(e), e), reverse=True)
    labels = [f"{emo} (n={_count(emo)})" for emo in ordered]
    y = np.arange(len(labels))

    def _series(stats_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        cis = []
        for emo in ordered:
            if emo not in stats_df.index:
                means.append(np.nan)
                cis.append(0.0)
                continue
            row = stats_df.loc[emo]
            mean = float(row["mean"])
            std = float(row["std"]) if not math.isnan(float(row["std"])) else 0.0
            count = int(row["count"])
            se = std / math.sqrt(count) if count > 1 else 0.0
            ci = 1.96 * se
            means.append(mean)
            cis.append(ci)
        return np.array(means), np.array(cis)

    means_a, cis_a = _series(stats_a)
    means_b, cis_b = _series(stats_b)

    height = max(4.0, 0.33 * len(labels) + 1.5)
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, height), sharey=True)
    axes[0].errorbar(means_a, y, xerr=cis_a, fmt="o", color="#4c78a8", ecolor="#999999", capsize=2)
    axes[1].errorbar(means_b, y, xerr=cis_b, fmt="o", color="#f58518", ecolor="#999999", capsize=2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[0].set_xlabel(_pretty_metric_name(metric_a))
    axes[1].set_xlabel(_pretty_metric_name(metric_b))
    axes[0].grid(axis="x")
    axes[1].grid(axis="x")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_coverage_by_emotion_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    metrics = ["p01", "p05", "p10"]
    stats = {m: _metric_stats_by_emotion(df, m) for m in metrics}
    emotions = set()
    for s in stats.values():
        emotions.update(s.index.tolist())
    if not emotions:
        return
    ordered = sorted(emotions, key=lambda e: max(int(stats[m].loc[e, "count"]) if e in stats[m].index else 0 for m in metrics), reverse=True)
    data = []
    for emo in ordered:
        row = []
        for m in metrics:
            s = stats[m]
            if emo not in s.index:
                row.append(np.nan)
            else:
                row.append(float(s.loc[emo, "mean"]))
        data.append(row)

    mat = np.array(data, dtype=float)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(5.0, 0.35 * len(ordered) + 2.2))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([_pretty_metric_name(m) for m in metrics], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Mean coverage")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_exposure(exposure: Dict[str, Any], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    gini = exposure.get("gini")
    hhi = exposure.get("hhi")
    vals = [gini, hhi]
    if any(v is None for v in vals):
        return
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    labels = ["gini", "hhi"]
    ax.bar(labels, vals, color=["#4c78a8", "#f58518"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _exposure_by_emotion_table(
    counts_by_emotion: Dict[str, Counter],
    episodes_by_emotion: Dict[str, int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for emotion, counts in counts_by_emotion.items():
        freqs = list(counts.values())
        if not freqs:
            continue
        total = int(sum(freqs))
        rows.append(
            {
                "emotion": emotion,
                "episodes": int(episodes_by_emotion.get(emotion, 0)),
                "unique_items": int(len(freqs)),
                "total_mentions": total,
                "gini": _gini(freqs),
                "hhi": _hhi(freqs),
                "mean_mentions_per_item": float(total / len(freqs)) if freqs else None,
            }
        )
    if not rows:
        return pd.DataFrame([])
    return pd.DataFrame(rows)


def _exposure_curve(
    exposure_counts: Counter,
    popularity_percentiles: Dict[Any, float],
) -> Optional[Tuple[List[float], List[float], float, float]]:
    pairs: List[Tuple[float, int]] = []
    for item, count in exposure_counts.items():
        pct = popularity_percentiles.get(item)
        if pct is None:
            continue
        try:
            pct_f = float(pct)
        except Exception:
            continue
        pairs.append((pct_f, int(count)))

    if not pairs:
        return None

    pairs.sort(key=lambda x: x[0])
    total = sum(c for _, c in pairs)
    if total <= 0:
        return None

    xs = [0.0]
    ys = [0.0]
    cum = 0
    for pct, count in pairs:
        cum += count
        xs.append(min(max(pct, 0.0), 1.0))
        ys.append(cum / total)
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(1.0)

    gini = _gini([c for _, c in pairs])
    hhi = _hhi([c for _, c in pairs])
    return xs, ys, gini, hhi


def _plot_exposure_emotion_curves(
    dataset_counts: Counter,
    counts_by_emotion: Dict[str, Counter],
    popularity_percentiles: Dict[Any, float],
    episodes_by_emotion: Dict[str, int],
    out_path: Path,
    title: str,
    metric: str,
) -> None:
    import matplotlib.pyplot as plt

    base = _exposure_curve(dataset_counts, popularity_percentiles)
    if not base:
        return
    xs_all, ys_all, gini_all, hhi_all = base
    metric_key = "gini" if metric.lower() == "gini" else "hhi"
    metric_tag = "G" if metric_key == "gini" else "HHI"
    metric_all = gini_all if metric_key == "gini" else hhi_all

    ordered = sorted(
        counts_by_emotion.keys(),
        key=lambda k: (episodes_by_emotion.get(k, 0), k),
        reverse=True,
    )

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(xs_all, ys_all, color="#4c78a8", linewidth=2, label=f"all ({metric_tag}={metric_all:.2f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#999999", linewidth=1)

    cmap = plt.get_cmap("tab10")
    for i, emotion in enumerate(ordered):
        curve = _exposure_curve(counts_by_emotion[emotion], popularity_percentiles)
        if not curve:
            continue
        xs, ys, gini_val, hhi_val = curve
        metric_val = gini_val if metric_key == "gini" else hhi_val
        color = cmap(i % 10)
        n = int(episodes_by_emotion.get(emotion, 0))
        ax.plot(
            xs,
            ys,
            linestyle=":",
            linewidth=1.5,
            color=color,
            label=f"{emotion} (n={n}, {metric_tag}={metric_val:.2f})",
        )

    ax.set_title(title)
    ax.set_xlabel("Popularity percentile (items sorted by popularity)")
    ax.set_ylabel("Cumulative exposure share")
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(ordered) / 8)))) if ordered else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(ordered)))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_bias_emotion_corr_heatmap(
    corr_df: pd.DataFrame,
    emotions: List[str],
    metrics: List[str],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if corr_df.empty:
        return
    data = []
    for emo in emotions:
        row = []
        for metric in metrics:
            subset = corr_df[(corr_df["emotion"] == emo) & (corr_df["metric"] == metric)]
            if subset.empty:
                row.append(np.nan)
            else:
                row.append(float(subset.iloc[0]["spearman_r"]))
        data.append(row)
    if not data:
        return

    mat = np.array(data, dtype=float)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(0.6 * len(metrics) + 3.5, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels(emotions)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman r")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_bias_emotion_mean_heatmap(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if df.empty or "emotion_top1" not in df:
        return
    counts = df["emotion_top1"].dropna().value_counts()
    emotions = counts.index.tolist()
    rows = []
    for emo in emotions:
        row = []
        sub = df[df["emotion_top1"] == emo]
        for metric in metrics:
            if metric not in sub:
                row.append(np.nan)
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce")
            row.append(float(vals.mean()) if not vals.isna().all() else np.nan)
        rows.append(row)
    if not rows:
        return

    mat = np.array(rows, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if np.all(np.isnan(col)):
            continue
        cmin = np.nanmin(col)
        cmax = np.nanmax(col)
        if cmax > cmin:
            mat[:, j] = (col - cmin) / (cmax - cmin)
        else:
            mat[:, j] = np.nan

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(0.6 * len(metrics) + 3.5, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels([f"{emo} (n={int(counts.get(emo, 0))})" for emo in emotions])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normalized mean (per metric)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _add_turn_order(df: pd.DataFrame, group_col: str, sort_col: str, out_col: str) -> None:
    if group_col not in df or sort_col not in df:
        return
    ordered = df.sort_values([group_col, sort_col])
    order = ordered.groupby(group_col).cumcount()
    df.loc[order.index, out_col] = order.astype(int)


def _collect_redial_turn_dists(
    cache_root: Path,
    df: pd.DataFrame,
    bias_name: str,
    bias_key: str,
    dist_key: str,
) -> List[Tuple[str, Dict[str, float]]]:
    from dataset.redial import safe_id

    needed: Dict[Tuple[str, str, int], str] = {}
    for row in df.itertuples():
        if not getattr(row, "emotion_top1", None):
            continue
        needed[(row.split, str(row.conversation_id), int(row.turn_idx))] = str(row.emotion_top1)

    if not needed:
        return []

    base = cache_root / "bias" / bias_name / "redial" / "ml-25m"
    by_split: Dict[str, set[str]] = defaultdict(set)
    for split, cid, _ in needed.keys():
        by_split[split].add(cid)

    out: List[Tuple[str, Dict[str, float]]] = []
    for split, conv_ids in by_split.items():
        for cid in conv_ids:
            path = base / split / f"{safe_id(cid)}.json"
            obj = _load_json(path)
            if not obj:
                continue
            for t in obj.get("turns", []):
                key = (split, cid, int(t.get("msg_idx", -1)))
                emotion = needed.get(key)
                if not emotion:
                    continue
                bias_obj = (t.get(bias_key) or {}) if isinstance(t, dict) else {}
                dist = bias_obj.get(dist_key) if isinstance(bias_obj, dict) else None
                if not isinstance(dist, dict) or not dist:
                    continue
                out.append((emotion, {str(k): float(v) for k, v in dist.items()}))
    return out


def _collect_cosrec_episode_dists(
    cache_root: Path,
    df: pd.DataFrame,
    bias_name: str,
    dist_key: str,
) -> List[Tuple[str, Dict[str, float]]]:
    from dataset.cosrec import safe_id

    by_topic: Dict[str, List[str]] = defaultdict(list)
    for row in df.itertuples():
        if not getattr(row, "emotion_top1", None):
            continue
        by_topic[str(row.topic_id)].append(str(row.emotion_top1))

    if not by_topic:
        return []

    base = cache_root / "bias" / bias_name / "cosrec" / "amazon_2023" / "curated"
    out: List[Tuple[str, Dict[str, float]]] = []
    for topic_id, emotions in by_topic.items():
        path = base / f"{safe_id(topic_id)}.json"
        obj = _load_json(path)
        if not obj:
            continue
        summary = obj.get("summary") if isinstance(obj, dict) else None
        dist = summary.get(dist_key) if isinstance(summary, dict) else None
        if not isinstance(dist, dict) or not dist:
            continue
        dist = {str(k): float(v) for k, v in dist.items()}
        for emotion in emotions:
            out.append((emotion, dist))
    return out


def _aggregate_dist_stats(
    pairs: List[Tuple[str, Dict[str, float]]],
) -> Dict[str, Any]:
    overall_sum: Dict[str, float] = defaultdict(float)
    emotion_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    emotion_counts: Dict[str, int] = defaultdict(int)
    dominant_counts: Dict[str, Counter] = defaultdict(Counter)
    top_shares: List[Tuple[str, float]] = []
    effective_nums: List[Tuple[str, float]] = []

    for emotion, dist in pairs:
        if not dist:
            continue
        emotion_counts[emotion] += 1
        top_share = max(dist.values())
        top_shares.append((emotion, float(top_share)))
        ent = 0.0
        for v in dist.values():
            if v > 0:
                ent -= v * math.log(v)
        effective_nums.append((emotion, float(math.exp(ent)) if ent > 0 else 0.0))

        dom = max(dist, key=dist.get)
        dominant_counts[emotion][dom] += 1

        for k, v in dist.items():
            overall_sum[k] += float(v)
            emotion_sum[emotion][k] += float(v)

    total_eps = float(sum(emotion_counts.values()))
    overall_mean = {k: (v / total_eps) for k, v in overall_sum.items()} if total_eps > 0 else {}

    emotion_mean: Dict[str, Dict[str, float]] = {}
    for emo, dist_sum in emotion_sum.items():
        cnt = float(emotion_counts.get(emo, 0))
        if cnt <= 0:
            continue
        emotion_mean[emo] = {k: (v / cnt) for k, v in dist_sum.items()}

    return {
        "overall_mean": overall_mean,
        "emotion_mean": emotion_mean,
        "emotion_counts": emotion_counts,
        "dominant_counts": dominant_counts,
        "top_shares": top_shares,
        "effective_nums": effective_nums,
    }


def _plot_metric_scatter_by_emotion(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if xcol not in df or ycol not in df:
        return
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = ~(x.isna() | y.isna())
    if mask.sum() == 0:
        return
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()
    labels = df.loc[mask, "emotion_top1"].fillna("unknown").to_numpy()
    uniq = sorted(set(labels))
    counts = {lab: int(np.sum(labels == lab)) for lab in uniq}

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(uniq):
        sel = labels == lab
        ax.scatter(
            x[sel],
            y[sel],
            s=12,
            alpha=0.35,
            color=cmap(i % 20),
            label=f"{lab} (n={counts.get(lab, 0)})",
        )
    ax.set_xlabel(_pretty_metric_name(xcol))
    ax.set_ylabel(_pretty_metric_name(ycol))
    ax.set_title(title)
    ax.grid(axis="both")
    ncol = min(6, max(2, int(math.ceil(len(uniq) / 8)))) if uniq else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(uniq)))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_genre_lift_heatmap(
    stats: Dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    overall = stats.get("overall_mean") or {}
    emo_mean = stats.get("emotion_mean") or {}
    if not overall or not emo_mean:
        return

    genres = sorted(overall.keys())
    counts = stats.get("emotion_counts") or {}
    emotions = sorted(emo_mean.keys(), key=lambda e: counts.get(e, 0), reverse=True)
    if not genres or not emotions:
        return

    mat = []
    for emo in emotions:
        row = []
        for g in genres:
            row.append(float(emo_mean.get(emo, {}).get(g, 0.0) - overall.get(g, 0.0)))
        mat.append(row)
    mat = np.array(mat, dtype=float)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(0.6 * len(genres) + 3.5, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(genres)))
    ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels([f"{emo} (n={counts.get(emo, 0)})" for emo in emotions])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Lift vs overall")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_topk_uplift_dots(
    stats: Dict[str, Any],
    out_dir: Path,
    title_prefix: str,
    top_k: int = 5,
) -> None:
    import matplotlib.pyplot as plt

    overall = stats.get("overall_mean") or {}
    emo_mean = stats.get("emotion_mean") or {}
    if not overall or not emo_mean:
        return

    _ensure_dir(out_dir)
    counts = stats.get("emotion_counts") or {}
    for emo, dist in emo_mean.items():
        deltas = {g: float(dist.get(g, 0.0) - overall.get(g, 0.0)) for g in overall}
        top = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:top_k]
        if not top:
            continue
        labels = [g for g, _ in top][::-1]
        values = [v for _, v in top][::-1]
        _set_plot_style()
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        y = np.arange(len(labels))
        ax.hlines(y=y, xmin=0, xmax=values, color="#c7c7c7", linewidth=1.0)
        ax.plot(values, y, "o", color="#4c78a8", markersize=5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Lift vs overall")
        ax.set_title(f"{title_prefix}: {emo} (n={counts.get(emo, 0)})")
        ax.grid(axis="x")
        fig.tight_layout()
        fig.savefig(out_dir / f"top_uplift_{emo}.png")
        plt.close(fig)


def _plot_alluvial_emotion_to_category(
    stats: Dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    dominant_counts = stats.get("dominant_counts") or {}
    emotion_counts = stats.get("emotion_counts") or {}
    if not dominant_counts:
        return

    emotions = sorted(emotion_counts.keys(), key=lambda e: emotion_counts.get(e, 0), reverse=True)
    genre_totals: Counter = Counter()
    for emo, cnts in dominant_counts.items():
        genre_totals.update(cnts)
    genres = [g for g, _ in genre_totals.most_common()]
    if not emotions or not genres:
        return

    emo_total = float(sum(emotion_counts.values()))
    genre_total = float(sum(genre_totals.values()))
    if emo_total <= 0 or genre_total <= 0:
        return

    emo_pos = {}
    y = 0.0
    for emo in emotions:
        h = emotion_counts.get(emo, 0) / emo_total
        emo_pos[emo] = (y, y + h)
        y += h

    genre_pos = {}
    y = 0.0
    for g in genres:
        h = genre_totals.get(g, 0) / genre_total
        genre_pos[g] = (y, y + h)
        y += h

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0.05, 0), 0.1, 1, facecolor="#f0f0f0", edgecolor="#cccccc"))
    ax.add_patch(Rectangle((0.85, 0), 0.1, 1, facecolor="#f0f0f0", edgecolor="#cccccc"))

    cmap = plt.get_cmap("tab20")
    for i, emo in enumerate(emotions):
        y0, y1 = emo_pos[emo]
        ax.text(
            0.02,
            0.5 * (y0 + y1),
            f"{emo} (n={emotion_counts.get(emo, 0)})",
            ha="right",
            va="center",
            fontsize=8,
        )
        for g, count in dominant_counts.get(emo, {}).items():
            if count <= 0:
                continue
            g0, g1 = genre_pos.get(g, (0.0, 0.0))
            h = count / emo_total
            left_top = y1
            left_bot = y1 - h
            y1 = left_bot
            width = 0.7
            poly = Polygon(
                [
                    (0.15, left_bot),
                    (0.15, left_top),
                    (0.85, g1),
                    (0.85, g0),
                ],
                closed=True,
                facecolor=cmap(i % 20),
                alpha=0.25,
                edgecolor=None,
            )
            ax.add_patch(poly)

    for g in genres:
        y0, y1 = genre_pos[g]
        ax.text(0.98, 0.5 * (y0 + y1), g, ha="left", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_year_ridgeline(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    bins: int = 25,
) -> None:
    import matplotlib.pyplot as plt

    if "mean_year" not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", "mean_year"]].copy()
    sub["mean_year"] = pd.to_numeric(sub["mean_year"], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    counts = sub["emotion_top1"].value_counts()
    emotions = counts.index.tolist()
    years = sub["mean_year"].to_numpy()
    ymin, ymax = np.nanmin(years), np.nanmax(years)
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 0.35 * len(emotions) + 2.5))
    for i, emo in enumerate(emotions):
        vals = sub[sub["emotion_top1"] == emo]["mean_year"].to_numpy()
        if len(vals) == 0:
            continue
        hist, edges = np.histogram(vals, bins=bins, range=(ymin, ymax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        y = hist / (hist.max() if hist.max() > 0 else 1.0)
        ax.fill_between(centers, i, i + y, color="#4c78a8", alpha=0.6)
        ax.plot(centers, i + y, color="#4c78a8", linewidth=1.0)
    ax.set_yticks(np.arange(len(emotions)) + 0.2)
    ax.set_yticklabels([f"{emo} (n={int(counts.get(emo, 0))})" for emo in emotions])
    ax.set_xlabel("Mean release year")
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_decade_stacked_bars(
    stats: Dict[str, Any],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    overall = stats.get("overall_mean") or {}
    emo_mean = stats.get("emotion_mean") or {}
    if not overall or not emo_mean:
        return

    decades = sorted(overall.keys())
    counts = stats.get("emotion_counts") or {}
    emotions = sorted(emo_mean.keys(), key=lambda e: counts.get(e, 0), reverse=True)
    if not decades or not emotions:
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(emotions) + 2.5))
    bottoms = np.zeros(len(emotions), dtype=float)
    cmap = plt.get_cmap("tab20")

    for i, dec in enumerate(decades):
        vals = np.array([emo_mean.get(e, {}).get(dec, 0.0) for e in emotions], dtype=float)
        ax.barh(np.arange(len(emotions)), vals, left=bottoms, color=cmap(i % 20), label=dec)
        bottoms += vals

    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels([f"{emo} (n={counts.get(emo, 0)})" for emo in emotions])
    ax.set_xlabel("Proportion")
    ax.set_title(title)
    ax.grid(axis="x")
    ncol = min(6, max(2, int(math.ceil(len(decades) / 8)))) if decades else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(decades)))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_repeat_rate_bar(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if "redundancy_repeated" not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", "redundancy_repeated"]].copy()
    sub["redundancy_repeated"] = pd.to_numeric(sub["redundancy_repeated"], errors="coerce").fillna(0.0)
    if sub.empty:
        return

    rows = []
    for emo, group in sub.groupby("emotion_top1"):
        vals = (group["redundancy_repeated"] > 0).astype(float).to_numpy()
        if len(vals) == 0:
            continue
        mean = float(vals.mean())
        lo, hi = _bootstrap_mean_ci(vals)
        rows.append((emo, len(vals), mean, lo, hi))
    if not rows:
        return
    rows.sort(key=lambda x: x[2])
    labels = [f"{r[0]} (n={r[1]})" for r in rows]
    means = np.array([r[2] for r in rows])
    lows = np.array([r[3] for r in rows])
    highs = np.array([r[4] for r in rows])
    errs = np.vstack([means - lows, highs - means])

    height = max(4.0, 0.35 * len(labels) + 1.5)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=errs, fmt="o", color="#4c78a8", ecolor="#999999", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("P(repeated > 0)")
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_exposure_topk_share_by_emotion(
    counts_by_emotion: Dict[str, Counter],
    episodes_by_emotion: Optional[Dict[str, int]],
    out_path: Path,
    title: str,
    ks: Tuple[int, int, int] = (10, 50, 100),
) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for emo, counts in counts_by_emotion.items():
        freqs = sorted(counts.values(), reverse=True)
        total = sum(freqs)
        if total <= 0:
            continue
        shares = []
        for k in ks:
            shares.append(float(sum(freqs[:k]) / total))
        n = int(episodes_by_emotion.get(emo, 0)) if episodes_by_emotion else int(sum(counts.values()))
        rows.append((emo, n, shares))
    if not rows:
        return
    rows.sort(key=lambda x: x[1], reverse=True)
    labels = [f"{r[0]} (n={r[1]})" for r in rows]
    data = np.array([r[2] for r in rows], dtype=float)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 0.35 * len(labels) + 2.5))
    y = np.arange(len(labels))
    height = 0.25
    for i, k in enumerate(ks):
        ax.barh(y + (i - 1) * height, data[:, i], height=height, label=f"top{k}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share of exposure")
    ax.set_title(title)
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(ks), fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_gini_hhi_scatter(
    exp_df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if exp_df.empty:
        return
    if "gini" not in exp_df or "hhi" not in exp_df:
        return
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.scatter(exp_df["gini"], exp_df["hhi"], color="#4c78a8", alpha=0.8)
    for _, row in exp_df.iterrows():
        label = row["emotion"]
        if "episodes" in row and not pd.isna(row["episodes"]):
            label = f"{label} (n={int(row['episodes'])})"
        ax.text(row["gini"], row["hhi"], label, fontsize=7, alpha=0.8)
    ax.set_xlabel("Gini")
    ax.set_ylabel("HHI")
    ax.set_title(title)
    ax.grid(axis="both")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_bias_rate_by_turn(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    turn_col: str,
) -> None:
    import matplotlib.pyplot as plt

    if "stereotype_label" not in df or "emotion_top1" not in df:
        return
    if turn_col not in df:
        return
    sub = df[["emotion_top1", "stereotype_label", turn_col]].copy()
    sub = sub.dropna()
    if sub.empty:
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    counts = sub["emotion_top1"].value_counts()
    emotions = counts.index.tolist()
    cmap = plt.get_cmap("tab20")
    for i, emo in enumerate(emotions):
        grp = sub[sub["emotion_top1"] == emo]
        if grp.empty:
            continue
        means = grp.groupby(turn_col)["stereotype_label"].mean().sort_index()
        ax.plot(
            means.index.to_numpy(),
            means.values,
            color=cmap(i % 20),
            linewidth=1.2,
            label=f"{emo} (n={int(counts.get(emo, 0))})",
        )
    ax.set_xlabel("Recommendation index")
    ax.set_ylabel("P(biased)")
    ax.set_title(title)
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(emotions) / 8)))) if emotions else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(emotions)))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_bias_vs_stereotype(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if "stereotype_label" not in df:
        return
    rows = []
    for metric in metrics:
        if metric not in df:
            continue
        sub = df[[metric, "stereotype_label"]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna()
        if sub.empty:
            continue
        vals0 = sub[sub["stereotype_label"] == 0][metric].to_numpy()
        vals1 = sub[sub["stereotype_label"] == 1][metric].to_numpy()
        if len(vals0) == 0 or len(vals1) == 0:
            continue
        rows.append((metric, vals0, vals1))
    if not rows:
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(rows) + 2.5))
    positions = np.arange(len(rows)) * 2.5
    data = []
    labels = []
    for i, (metric, vals0, vals1) in enumerate(rows):
        data.append(vals0)
        data.append(vals1)
        labels.extend([f"{_pretty_metric_name(metric)} (Non-biased)", f"{_pretty_metric_name(metric)} (Biased)"])
    ax.violinplot(data, showmedians=True, vert=False, positions=np.arange(len(data)))
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Value")
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stereotype_score_by_emotion(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    if "stereotype_score_1" not in df:
        return
    _plot_violin_by_emotion(df, "stereotype_score_1", out_path, title)


def _plot_effect_size_heatmap(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if df.empty or "emotion_top1" not in df:
        return
    emotions = df["emotion_top1"].dropna().value_counts().index.tolist()
    if not emotions:
        return

    mat = []
    metric_names = []
    for metric in metrics:
        if metric not in df:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce")
        overall_mean = float(vals.mean())
        overall_std = float(vals.std()) if float(vals.std()) > 0 else 1.0
        metric_names.append(metric)
        col = []
        for emo in emotions:
            mvals = pd.to_numeric(df[df["emotion_top1"] == emo][metric], errors="coerce")
            mean = float(mvals.mean()) if not mvals.isna().all() else np.nan
            col.append((mean - overall_mean) / overall_std)
        mat.append(col)
    if not mat:
        return
    mat = np.array(mat, dtype=float).T

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(0.6 * len(metric_names) + 3.5, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels([_pretty_metric_name(m) for m in metric_names], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels(emotions)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Effect size (z)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_parallel_coordinates(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    title: str,
    max_samples: int = 1000,
    seed: int = 7,
) -> None:
    import matplotlib.pyplot as plt

    if df.empty or "emotion_top1" not in df:
        return
    use_metrics = [m for m in metrics if m in df]
    if not use_metrics:
        return

    sub = df[["emotion_top1"] + use_metrics].copy()
    for m in use_metrics:
        sub[m] = pd.to_numeric(sub[m], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return
    if len(sub) > max_samples:
        sub = sub.sample(n=max_samples, random_state=seed)

    # Normalize metrics to 0-1 for plotting.
    for m in use_metrics:
        vals = sub[m]
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax > vmin:
            sub[m] = (vals - vmin) / (vmax - vmin)
        else:
            sub[m] = 0.0

    counts = sub["emotion_top1"].value_counts()
    counts = sub["emotion_top1"].value_counts()
    emotions = sub["emotion_top1"].unique().tolist()
    cmap = plt.get_cmap("tab20")
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(use_metrics))
    for i, emo in enumerate(emotions):
        rows = sub[sub["emotion_top1"] == emo]
        for _, row in rows.iterrows():
            ax.plot(x, row[use_metrics].to_numpy(), color=cmap(i % 20), alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_metric_name(m) for m in use_metrics], rotation=45, ha="right")
    ax.set_ylabel("Normalized value")
    ax.set_title(title)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_radar_per_emotion(
    df: pd.DataFrame,
    metrics: List[str],
    out_dir: Path,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    _ensure_dir(out_dir)
    use_metrics = [m for m in metrics if m in df]
    if not use_metrics:
        return
    sub = df[["emotion_top1"] + use_metrics].copy()
    for m in use_metrics:
        sub[m] = pd.to_numeric(sub[m], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    # Normalize by overall min-max per metric.
    for m in use_metrics:
        vmin, vmax = float(sub[m].min()), float(sub[m].max())
        if vmax > vmin:
            sub[m] = (sub[m] - vmin) / (vmax - vmin)
        else:
            sub[m] = 0.0

    emotions = sub["emotion_top1"].unique().tolist()
    angles = np.linspace(0, 2 * np.pi, len(use_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for emo in emotions:
        rows = sub[sub["emotion_top1"] == emo]
        vals = [float(rows[m].mean()) for m in use_metrics]
        vals += vals[:1]
        fig, ax = plt.subplots(figsize=(5.2, 4.8), subplot_kw={"polar": True})
        ax.plot(angles, vals, color="#4c78a8", linewidth=2)
        ax.fill(angles, vals, color="#4c78a8", alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([_pretty_metric_name(m) for m in use_metrics], fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(f"{title_prefix}: {emo} (n={int(counts.get(emo, 0))})", y=1.08)
        fig.tight_layout()
        fig.savefig(out_dir / f"radar_{emo}.png")
        plt.close(fig)


def _plot_simple_violin_from_pairs(
    pairs: List[Tuple[str, float]],
    out_path: Path,
    title: str,
    xlabel: str,
) -> None:
    import matplotlib.pyplot as plt

    if not pairs:
        return
    df = pd.DataFrame(pairs, columns=["emotion_top1", "value"])
    stats = df.groupby("emotion_top1")["value"].agg(["count", "median"])
    stats = stats.sort_values("median", ascending=True)
    labels = [f"{idx} (n={int(row['count'])})" for idx, row in stats.iterrows()]
    data = [df[df["emotion_top1"] == idx]["value"].to_numpy() for idx in stats.index.tolist()]

    height = max(4.0, 0.35 * len(labels) + 1.5)
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, vert=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c78a8")
        pc.set_alpha(0.7)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_density_by_emotion(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_dir: Path,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    if xcol not in df or ycol not in df:
        return
    _ensure_dir(out_dir)
    emotions = df["emotion_top1"].dropna().value_counts().index.tolist()
    for emo in emotions:
        sub = df[df["emotion_top1"] == emo]
        x = pd.to_numeric(sub[xcol], errors="coerce")
        y = pd.to_numeric(sub[ycol], errors="coerce")
        mask = ~(x.isna() | y.isna())
        if mask.sum() == 0:
            continue
        x = x[mask].to_numpy()
        y = y[mask].to_numpy()
        _set_plot_style()
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.hexbin(x, y, gridsize=20, cmap="Blues", mincnt=1)
        ax.set_xlabel(_pretty_metric_name(xcol))
        ax.set_ylabel(_pretty_metric_name(ycol))
        ax.set_title(f"{title_prefix}: {emo}")
        ax.grid(axis="both")
        fig.tight_layout()
        fig.savefig(out_dir / f"{xcol}_{ycol}_{emo}.png")
        plt.close(fig)


def _plot_turn_trajectory_by_emotion(
    df: pd.DataFrame,
    value_col: str,
    turn_col: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if value_col not in df or turn_col not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", value_col, turn_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    counts = sub["emotion_top1"].value_counts()
    emotions = counts.index.tolist()
    cmap = plt.get_cmap("tab20")
    for i, emo in enumerate(emotions):
        grp = sub[sub["emotion_top1"] == emo]
        means = grp.groupby(turn_col)[value_col].mean().sort_index()
        ax.plot(
            means.index.to_numpy(),
            means.values,
            color=cmap(i % 20),
            linewidth=1.2,
            label=f"{emo} (n={int(counts.get(emo, 0))})",
        )
    ax.set_xlabel("Recommendation index")
    ax.set_ylabel(_pretty_metric_name(value_col))
    ax.set_title(title)
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(emotions) / 8)))) if emotions else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(emotions)))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pareto_repeated_items(
    counts_by_emotion: Dict[str, Counter],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if not counts_by_emotion:
        return
    emotions = sorted(counts_by_emotion.keys(), key=lambda e: sum(counts_by_emotion[e].values()), reverse=True)
    if not emotions:
        return

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.get_cmap("tab20")
    for i, emo in enumerate(emotions):
        freqs = sorted(counts_by_emotion[emo].values(), reverse=True)
        total = sum(freqs)
        if total <= 0:
            continue
        cum = np.cumsum(freqs) / total
        x = np.linspace(0, 1, len(cum), endpoint=True)
        ax.plot(
            x,
            cum,
            color=cmap(i % 20),
            linewidth=1.2,
            label=f"{emo} (n={total})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlabel("Share of items (sorted)")
    ax.set_ylabel("Cumulative share of repeats")
    ax.set_title(title)
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(emotions) / 8)))) if emotions else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(emotions)))
    fig.savefig(out_path)
    plt.close(fig)


def _collect_redial_redundancy_stats(
    cache_root: Path,
    df: pd.DataFrame,
) -> Tuple[Dict[str, Counter], pd.DataFrame]:
    from dataset.redial import safe_id

    needed: Dict[Tuple[str, str, int], str] = {}
    for row in df.itertuples():
        if not getattr(row, "emotion_top1", None):
            continue
        needed[(row.split, str(row.conversation_id), int(row.turn_idx))] = str(row.emotion_top1)

    counts_by_emotion: Dict[str, Counter] = defaultdict(Counter)
    rows: List[Dict[str, Any]] = []
    base = cache_root / "bias" / "redundancy" / "redial" / "ml-25m"
    by_split: Dict[str, set[str]] = defaultdict(set)
    for split, cid, _ in needed.keys():
        by_split[split].add(cid)

    for split, conv_ids in by_split.items():
        for cid in conv_ids:
            path = base / split / f"{safe_id(cid)}.json"
            obj = _load_json(path)
            if not obj:
                continue
            seen: set[int] = set()
            rec_order = 0
            for t in obj.get("turns", []):
                msg_idx = int(t.get("msg_idx", -1))
                key = (split, cid, msg_idx)
                emotion = needed.get(key)
                speaker = t.get("speaker")
                if speaker == "Recommender":
                    rec_order += 1
                if not emotion:
                    continue
                repeated = t.get("repeated_items") or []
                for item in repeated:
                    try:
                        counts_by_emotion[emotion][str(item)] += 1
                    except Exception:
                        counts_by_emotion[emotion][str(item)] += 1
                new_items = t.get("new_items") or []
                for item in new_items:
                    try:
                        seen.add(int(item))
                    except Exception:
                        continue
                rows.append(
                    {
                        "emotion_top1": emotion,
                        "rec_turn_order": rec_order,
                        "unique_items_so_far": len(seen),
                    }
                )
    return counts_by_emotion, pd.DataFrame(rows)


def _collect_cosrec_redundancy_stats(
    cache_root: Path,
    df: pd.DataFrame,
) -> Tuple[Dict[str, Counter], pd.DataFrame]:
    from dataset.cosrec import safe_id

    topic_emotion: Dict[str, List[str]] = defaultdict(list)
    for row in df.itertuples():
        if not getattr(row, "emotion_top1", None):
            continue
        topic_emotion[str(row.topic_id)].append(str(row.emotion_top1))

    counts_by_emotion: Dict[str, Counter] = defaultdict(Counter)
    rows: List[Dict[str, Any]] = []
    base = cache_root / "bias" / "redundancy" / "cosrec" / "amazon_2023" / "curated"

    for topic_id, emotions in topic_emotion.items():
        path = base / f"{safe_id(topic_id)}.json"
        obj = _load_json(path)
        if not obj:
            continue
        repeated = obj.get("repeated_items") or []
        summary = obj.get("summary") or {}
        unique_so_far = summary.get("unique_items_so_far")
        rec_order = obj.get("utterance_idx")
        for emo in emotions:
            for item in repeated:
                counts_by_emotion[emo][str(item)] += 1
            if unique_so_far is not None and rec_order is not None:
                rows.append(
                    {
                        "emotion_top1": emo,
                        "rec_turn_order": int(rec_order),
                        "unique_items_so_far": int(unique_so_far),
                    }
                )
    return counts_by_emotion, pd.DataFrame(rows)


def _corr_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    labels = sorted(df["emotion_top1"].dropna().unique().tolist())
    for metric in metrics:
        if metric not in df:
            continue
        series = df[metric].astype(float)
        if series.isna().all():
            continue
        for lab in labels:
            mask = ~series.isna()
            x = series[mask].to_numpy()
            y = (df.loc[mask, "emotion_top1"] == lab).astype(int).to_numpy()
            pr, pp, sr, sp = _pearson_spearman(x, y)
            rows.append(
                {
                    "metric": metric,
                    "emotion": lab,
                    "n": int(len(x)),
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": sp,
                }
            )
    return pd.DataFrame(rows)


def _summarize_correlations(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if df.empty:
        return df
    d = df[df["spearman_p"] < 0.05].copy()
    if d.empty:
        return d
    d["abs_spearman"] = d["spearman_r"].abs()
    d = d.sort_values("abs_spearman", ascending=False)
    return d.head(top_n)


def _write_summary(
    out_path: Path,
    dataset: str,
    df: pd.DataFrame,
    corr: pd.DataFrame,
    models: pd.DataFrame,
    exposure: Dict[str, Any],
) -> None:
    lines = []
    lines.append(f"Dataset: {dataset}")
    lines.append(f"Episodes: {len(df)}")
    if "stereotype_label" in df:
        counts = df["stereotype_label"].value_counts(dropna=False).to_dict()
        total = float(sum(v for v in counts.values() if v is not None))
        if total > 0:
            rate_0 = (counts.get(0, 0) / total) * 100.0
            rate_1 = (counts.get(1, 0) / total) * 100.0
            lines.append(
                "Stereotype label rates (%): "
                f"Non-biased={rate_0:.1f}, Biased={rate_1:.1f}"
            )

    for metric in ["p01", "p05", "p10", "pi", "pop_mean_pct", "cep", "uiop"]:
        if metric in df:
            series = pd.to_numeric(df[metric], errors="coerce")
            nonzero = float((series.fillna(0.0) > 0).mean())
            lines.append(f"{metric} nonzero rate: {nonzero:.3f}")

    if exposure:
        lines.append(
            f"Exposure: gini={exposure.get('gini')}, hhi={exposure.get('hhi')}"
        )

    lines.append("")
    lines.append("Top correlations (Spearman p<0.05):")
    top_corr = _summarize_correlations(corr, top_n=10)
    if top_corr.empty:
        lines.append("  none")
    else:
        for _, row in top_corr.iterrows():
            lines.append(
                f"  {row['metric']} vs {row['emotion']}: r={row['spearman_r']:.3f} (p={row['spearman_p']:.3g}, n={int(row['n'])})"
            )

    lines.append("")
    lines.append("Model summary (5-fold CV):")
    if models.empty:
        lines.append("  none (missing scikit-learn)")
    else:
        for _, row in models.iterrows():
            if row["model"] == "ridge_regression":
                lines.append(
                    f"  {row['metric']}: R^2={row['r2_mean']:.3f}, MAE={row['mae_mean']:.3f}, n={int(row['n'])}"
                )
            else:
                lines.append(
                    f"  {row['metric']}: AUC={row['roc_auc_mean']:.3f}, Acc={row['acc_mean']:.3f}, n={int(row['n'])}"
                )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _model_tables(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, r2_score
        from sklearn.model_selection import KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError(
            "Missing scikit-learn; install it to run modeling: pip install scikit-learn"
        ) from e

    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    rows = []
    if not emo_cols:
        return pd.DataFrame(rows)

    X = df[emo_cols].fillna(0.0)
    kf = KFold(n_splits=5, shuffle=True, random_state=7)

    for metric in metrics:
        if metric not in df:
            continue
        y = df[metric]
        if y.isna().all():
            continue
        if metric == "stereotype_label":
            yb = y.dropna().astype(int)
            Xb = X.loc[yb.index]
            if yb.nunique() < 2 or len(yb) < 10:
                continue
            model = Pipeline(
                steps=[
                    ("scale", StandardScaler(with_mean=False)),
                    ("clf", LogisticRegression(max_iter=200)),
                ]
            )
            aucs = []
            accs = []
            for train_idx, test_idx in kf.split(Xb):
                X_train, X_test = Xb.iloc[train_idx], Xb.iloc[test_idx]
                y_train, y_test = yb.iloc[train_idx], yb.iloc[test_idx]
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                pred = (prob >= 0.5).astype(int)
                aucs.append(roc_auc_score(y_test, prob))
                accs.append(accuracy_score(y_test, pred))
            rows.append(
                {
                    "metric": metric,
                    "model": "logistic_regression",
                    "n": int(len(yb)),
                    "roc_auc_mean": float(np.mean(aucs)),
                    "acc_mean": float(np.mean(accs)),
                }
            )
        else:
            yv = y.dropna().astype(float)
            Xv = X.loc[yv.index]
            if len(yv) < 10:
                continue
            model = Pipeline(
                steps=[
                    ("scale", StandardScaler(with_mean=False)),
                    ("reg", Ridge(alpha=1.0)),
                ]
            )
            r2s = []
            maes = []
            for train_idx, test_idx in kf.split(Xv):
                X_train, X_test = Xv.iloc[train_idx], Xv.iloc[test_idx]
                y_train, y_test = yv.iloc[train_idx], yv.iloc[test_idx]
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                r2s.append(r2_score(y_test, pred))
                maes.append(mean_absolute_error(y_test, pred))
            rows.append(
                {
                    "metric": metric,
                    "model": "ridge_regression",
                    "n": int(len(yv)),
                    "r2_mean": float(np.mean(r2s)),
                    "mae_mean": float(np.mean(maes)),
                }
            )

    return pd.DataFrame(rows)


def _load_redial(
    cache_root: Path,
    splits: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    from dataset.redial import ReDialDataset, get_speaker, safe_id

    base = cache_root / "emotion" / "go_emotions" / "redial" / "top5" / "titles"
    bias_base = cache_root / "bias"

    ds = ReDialDataset(cache_root=cache_root)
    records: List[Dict[str, Any]] = []
    exposure_counts: Dict[str, Counter] = defaultdict(Counter)
    exposure_episodes: Dict[str, int] = defaultdict(int)
    exposure_items = Counter()
    exposure_popularity: Dict[int, float] = {}

    for split in splits:
        emo_dir = base / split
        pop_dir = bias_base / "popularity" / "redial" / "ml-25m" / split
        epi_dir = bias_base / "episode_popularity" / "redial" / "ml-25m" / split
        genre_dir = bias_base / "genre" / "redial" / "ml-25m" / split
        year_dir = bias_base / "year_decade" / "redial" / "ml-25m" / split
        red_dir = bias_base / "redundancy" / "redial" / "ml-25m" / split

        stereo_base = bias_base / "stereotype" / "redial"
        stereo_dir = _top_dir(stereo_base) / split if _top_dir(stereo_base) else None

        for conv in ds.iter(split=split):
            cid = str(conv.conversation_id)
            sid = safe_id(cid)
            emo = _load_json(emo_dir / f"{sid}.json")
            if not emo:
                continue

            pop = _load_json(pop_dir / f"{sid}.json")
            epi = _load_json(epi_dir / f"{sid}.json")
            gen = _load_json(genre_dir / f"{sid}.json")
            year = _load_json(year_dir / f"{sid}.json")
            red = _load_json(red_dir / f"{sid}.json")
            stereo = _load_json(stereo_dir / f"{sid}.json") if stereo_dir else None

            emo_turns = {t["msg_idx"]: t for t in emo.get("turns", [])}
            pop_turns = {t["msg_idx"]: t for t in (pop or {}).get("turns", [])}
            epi_turns = {t["msg_idx"]: t for t in (epi or {}).get("turns", [])}
            gen_turns = {t["msg_idx"]: t for t in (gen or {}).get("turns", [])}
            year_turns = {t["msg_idx"]: t for t in (year or {}).get("turns", [])}
            red_turns = {t["msg_idx"]: t for t in (red or {}).get("turns", [])}
            stereo_turns = {t["msg_idx"]: t for t in (stereo or {}).get("turns", [])}

            for i, msg in enumerate(conv.messages):
                if get_speaker(msg, i) != "Recommender":
                    continue
                pop_obj = (pop_turns.get(i) or {}).get("popularity")
                if not pop_obj:
                    continue
                pop_items = (pop_turns.get(i) or {}).get("movielens_movie_ids") or []
                pop_pcts = pop_obj.get("percentiles") or []
                if pop_items and pop_pcts:
                    for mid, pct in zip(pop_items, pop_pcts):
                        try:
                            mid_int = int(mid)
                        except Exception:
                            continue
                        exposure_items[mid_int] += 1
                        if mid_int not in exposure_popularity:
                            try:
                                exposure_popularity[mid_int] = float(pct)
                            except Exception:
                                continue

                next_idx = None
                for j in range(i + 1, len(conv.messages)):
                    if get_speaker(conv.messages[j], j) == "Seeker":
                        next_idx = j
                        break
                if next_idx is None:
                    continue
                emo_obj = emo_turns.get(next_idx, {}).get("emotion")
                emo_scores, emo_top1, emo_top1_score = _emotion_vector(emo_obj)
                if not emo_scores:
                    continue

                pcts = pop_obj.get("percentiles") or []
                p01 = _coverage_at(pcts, 0.99)
                p05 = _coverage_at(pcts, 0.95)
                p10 = _coverage_at(pcts, 0.90)
                pi = _rank_utility(pcts)

                epi_obj = (epi_turns.get(i) or {}).get("episode_popularity") or {}
                cep = epi_obj.get("cep_similarity")
                uiop = epi_obj.get("uiop_similarity")

                gen_obj = (gen_turns.get(i) or {}).get("genre_bias") or {}
                year_obj = (year_turns.get(i) or {}).get("year_decade_bias") or {}
                red_obj = red_turns.get(i) or {}
                stereo_obj = (stereo_turns.get(i) or {}).get("bias") or []
                stereo_label, stereo_score_0, stereo_score_1 = _stereotype_scores(stereo_obj)

                record = {
                    "dataset": "redial",
                    "split": split,
                    "conversation_id": cid,
                    "turn_idx": i,
                    "emotion_top1": emo_top1,
                    "emotion_top1_score": emo_top1_score,
                    "p01": p01,
                    "p05": p05,
                    "p10": p10,
                    "pi": pi,
                    "cep": cep,
                    "uiop": uiop,
                    "pop_mean_pct": pop_obj.get("mean_percentile"),
                    "genre_js": gen_obj.get("js_divergence_vs_catalog"),
                    "genre_entropy": gen_obj.get("genres_entropy"),
                    "year_decade_js": year_obj.get("decades_js_divergence_vs_catalog"),
                    "year_decade_entropy": year_obj.get("decades_entropy"),
                    "mean_year": year_obj.get("mean_year"),
                    "redundancy_new": len(red_obj.get("new_items") or []),
                    "redundancy_repeated": len(red_obj.get("repeated_items") or []),
                    "stereotype_label": stereo_label,
                    "stereotype_score_0": stereo_score_0,
                    "stereotype_score_1": stereo_score_1,
                }
                for lab, score in emo_scores.items():
                    record[f"emo_{lab}"] = score
                records.append(record)

                if emo_top1 and pop_items:
                    exposure_episodes[emo_top1] += 1
                    exposure_counts[emo_top1].update(pop_items)

    exposure = _load_json(cache_root / "bias" / "exposure_concentration" / "redial" / "ml-25m" / "train.json") or {}
    df = pd.DataFrame(records)
    exposure_by_emotion = {"counts": exposure_counts, "episodes": exposure_episodes}
    exposure_items_meta = {"counts": exposure_items, "percentiles": exposure_popularity}
    return df, exposure, exposure_by_emotion, exposure_items_meta


def _load_cosrec(cache_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    from dataset.cosrec import CoSRecDataset, safe_id

    ds = CoSRecDataset(cache_root=cache_root)
    emo_dir = cache_root / "emotion" / "go_emotions" / "cosrec" / "top5" / "curated_turns"
    bias_base = cache_root / "bias"

    records: List[Dict[str, Any]] = []
    emo_cache: Dict[str, Dict[int, Any]] = {}
    exposure_counts: Dict[str, Counter] = defaultdict(Counter)
    exposure_episodes: Dict[str, int] = defaultdict(int)
    exposure_items = Counter()
    exposure_popularity: Dict[str, float] = {}

    def emo_for(conv_id: str) -> Dict[int, Any]:
        if conv_id in emo_cache:
            return emo_cache[conv_id]
        emo = _load_json(emo_dir / f"{safe_id(conv_id)}.json") or {}
        turns = {t["turn_idx"]: t for t in emo.get("turns", [])}
        emo_cache[conv_id] = turns
        return turns

    stereo_base = bias_base / "stereotype" / "cosrec"
    stereo_dir = _top_dir(stereo_base) / "curated" if _top_dir(stereo_base) else None

    for ep in ds.iter_rec_episodes(min_relevance=1):
        conv_id = str(ep.conversation_id)
        emo_turns = emo_for(conv_id)
        emo_obj = (emo_turns.get(ep.next_user_turn_idx) or {}).get("emotion")
        emo_scores, emo_top1, emo_top1_score = _emotion_vector(emo_obj)
        if not emo_scores:
            continue

        topic = safe_id(ep.topic_id)
        pop = _load_json(
            bias_base / "popularity" / "cosrec" / "amazon_2023" / "curated" / f"{topic}.json"
        )
        pop_obj = (pop or {}).get("popularity") or {}
        pcts = pop_obj.get("percentiles") or []
        if not pcts:
            continue

        p01 = _coverage_at(pcts, 0.99)
        p05 = _coverage_at(pcts, 0.95)
        p10 = _coverage_at(pcts, 0.90)
        pi = _rank_utility(pcts)

        epi = _load_json(
            bias_base / "episode_popularity" / "cosrec" / "amazon_2023" / "curated" / f"{topic}.json"
        )
        epi_obj = (epi or {}).get("episode_popularity") or {}
        cep = epi_obj.get("cep_similarity")
        uiop = epi_obj.get("uiop_similarity")
        pop_qrels = (pop or {}).get("qrels") or []
        pop_items = [str(q.get("asin")) for q in pop_qrels if q.get("asin")]
        missing = set(str(a) for a in (pop or {}).get("missing_asins") or [])
        mapped_asins = [asin for asin in pop_items if asin not in missing]
        if mapped_asins and pcts:
            for asin, pct in zip(mapped_asins, pcts):
                exposure_items[asin] += 1
                if asin not in exposure_popularity:
                    try:
                        exposure_popularity[asin] = float(pct)
                    except Exception:
                        continue

        gen = _load_json(
            bias_base / "genre" / "cosrec" / "amazon_2023" / "curated" / f"{topic}.json"
        )
        gen_obj = (gen or {}).get("summary") or {}

        rat = _load_json(
            bias_base / "rating" / "cosrec" / "amazon_2023" / "curated" / f"{topic}.json"
        )
        rat_obj = (rat or {}).get("rating") or {}

        red = _load_json(
            bias_base / "redundancy" / "cosrec" / "amazon_2023" / "curated" / f"{topic}.json"
        )

        stereo = _load_json(stereo_dir / f"{topic}.json") if stereo_dir else None
        stereo_obj = (stereo or {}).get("bias") or []
        stereo_label, stereo_score_0, stereo_score_1 = _stereotype_scores(stereo_obj)

        record = {
            "dataset": "cosrec",
            "conversation_id": conv_id,
            "topic_id": str(ep.topic_id),
            "system_turn_idx": int(ep.system_turn_idx),
            "utterance_idx": int(ep.utterance_idx),
            "user_index": int(ep.user_index),
            "emotion_top1": emo_top1,
            "emotion_top1_score": emo_top1_score,
            "p01": p01,
            "p05": p05,
            "p10": p10,
            "pi": pi,
            "cep": cep,
            "uiop": uiop,
            "pop_mean_pct": pop_obj.get("mean_percentile"),
            "rating_mean": rat_obj.get("mean_rating"),
            "rating_mean_pct": rat_obj.get("mean_percentile"),
            "genre_js": gen_obj.get("js_divergence_vs_catalog"),
            "genre_entropy": gen_obj.get("categories_entropy"),
            "redundancy_new": len((red or {}).get("new_items") or []),
            "redundancy_repeated": len((red or {}).get("repeated_items") or []),
            "stereotype_label": stereo_label,
            "stereotype_score_0": stereo_score_0,
            "stereotype_score_1": stereo_score_1,
        }
        for lab, score in emo_scores.items():
            record[f"emo_{lab}"] = score
        records.append(record)

        if emo_top1 and mapped_asins:
            exposure_episodes[emo_top1] += 1
            exposure_counts[emo_top1].update(mapped_asins)

    exposure = _load_json(
        cache_root / "bias" / "exposure_concentration" / "cosrec" / "amazon_2023" / "curated.json"
    ) or {}
    df = pd.DataFrame(records)
    exposure_by_emotion = {"counts": exposure_counts, "episodes": exposure_episodes}
    exposure_items_meta = {"counts": exposure_items, "percentiles": exposure_popularity}
    return df, exposure, exposure_by_emotion, exposure_items_meta


def analyze_stats(
    cache_root: Path | str = "./cache",
    out_dir: Path | str = "./results",
    splits: Iterable[str] = ("train", "test"),
) -> None:
    cache_root = Path(cache_root)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    _ensure_dir(tables_dir)
    _ensure_dir(figs_dir)
    final_dir = figs_dir / "final"
    _ensure_dir(final_dir)
    family_dirs = {
        "emotion": final_dir / "emotion",
        "popularity": final_dir / "popularity",
        "episode_popularity": final_dir / "episode_popularity",
        "genre": final_dir / "genre",
        "year_decade": final_dir / "year_decade",
        "redundancy": final_dir / "redundancy",
        "exposure": final_dir / "exposure",
        "rating": final_dir / "rating",
        "stereotype": final_dir / "stereotype",
        "summary": final_dir / "summary",
    }
    for p in family_dirs.values():
        _ensure_dir(p)

    redial_df, redial_exposure, redial_exposure_emotions, redial_exposure_items = _load_redial(
        cache_root, splits
    )
    cosrec_df, cosrec_exposure, cosrec_exposure_emotions, cosrec_exposure_items = _load_cosrec(cache_root)

    _add_turn_order(redial_df, "conversation_id", "turn_idx", "rec_turn_order")
    _add_turn_order(cosrec_df, "conversation_id", "system_turn_idx", "rec_turn_order")

    redial_df.to_csv(tables_dir / "episodes_redial.csv", index=False)
    cosrec_df.to_csv(tables_dir / "episodes_cosrec.csv", index=False)

    redial_metrics = [
        "p01",
        "p05",
        "p10",
        "pi",
        "cep",
        "uiop",
        "pop_mean_pct",
        "genre_js",
        "genre_entropy",
        "year_decade_js",
        "year_decade_entropy",
        "mean_year",
        "redundancy_new",
        "redundancy_repeated",
    ]
    cosrec_metrics = [
        "p01",
        "p05",
        "p10",
        "pi",
        "cep",
        "uiop",
        "pop_mean_pct",
        "rating_mean",
        "rating_mean_pct",
        "genre_js",
        "genre_entropy",
        "redundancy_new",
        "redundancy_repeated",
    ]

    redial_corr = _corr_table(redial_df, redial_metrics)
    cosrec_corr = _corr_table(cosrec_df, cosrec_metrics)
    redial_corr.to_csv(tables_dir / "correlations_redial.csv", index=False)
    cosrec_corr.to_csv(tables_dir / "correlations_cosrec.csv", index=False)

    redial_models = _model_tables(redial_df, redial_metrics + ["stereotype_label"])
    cosrec_models = _model_tables(cosrec_df, cosrec_metrics + ["stereotype_label"])
    redial_models.to_csv(tables_dir / "models_redial.csv", index=False)
    cosrec_models.to_csv(tables_dir / "models_cosrec.csv", index=False)

    redial_exp_df = _exposure_by_emotion_table(
        redial_exposure_emotions["counts"], redial_exposure_emotions["episodes"]
    )
    cosrec_exp_df = _exposure_by_emotion_table(
        cosrec_exposure_emotions["counts"], cosrec_exposure_emotions["episodes"]
    )
    redial_exp_df.to_csv(tables_dir / "exposure_by_emotion_redial.csv", index=False)
    cosrec_exp_df.to_csv(tables_dir / "exposure_by_emotion_cosrec.csv", index=False)

    redial_genre_pairs = _collect_redial_turn_dists(
        cache_root, redial_df, "genre", "genre_bias", "genres_dist"
    )
    cosrec_genre_pairs = _collect_cosrec_episode_dists(
        cache_root, cosrec_df, "genre", "categories_dist"
    )
    redial_genre_stats = _aggregate_dist_stats(redial_genre_pairs)
    cosrec_genre_stats = _aggregate_dist_stats(cosrec_genre_pairs)

    redial_decade_pairs = _collect_redial_turn_dists(
        cache_root, redial_df, "year_decade", "year_decade_bias", "decades_dist"
    )
    redial_decade_stats = _aggregate_dist_stats(redial_decade_pairs)

    redial_repeat_counts, redial_redundancy_progress = _collect_redial_redundancy_stats(
        cache_root, redial_df
    )
    cosrec_repeat_counts, cosrec_redundancy_progress = _collect_cosrec_redundancy_stats(
        cache_root, cosrec_df
    )

    _write_summary(
        tables_dir / "summary_redial.txt",
        "ReDial",
        redial_df,
        redial_corr,
        redial_models,
        redial_exposure,
    )
    _write_summary(
        tables_dir / "summary_cosrec.txt",
        "CoSRec",
        cosrec_df,
        cosrec_corr,
        cosrec_models,
        cosrec_exposure,
    )

    # Emotion signal graphs.
    _plot_emotion_distribution(
        redial_df,
        family_dirs["emotion"] / "emotion_distribution_redial.png",
        "ReDial: Emotion distribution",
    )
    _plot_emotion_distribution(
        cosrec_df,
        family_dirs["emotion"] / "emotion_distribution_cosrec.png",
        "CoSRec: Emotion distribution",
    )
    _plot_emotion_distribution_panels(
        redial_df,
        family_dirs["emotion"] / "emotion_top1_vs_scores_redial.png",
        "ReDial: Top-1 vs score distributions",
    )
    _plot_emotion_distribution_panels(
        cosrec_df,
        family_dirs["emotion"] / "emotion_top1_vs_scores_cosrec.png",
        "CoSRec: Top-1 vs score distributions",
    )
    _plot_emotion_cooccurrence(
        redial_df,
        family_dirs["emotion"] / "emotion_cooccurrence_redial.png",
        "ReDial: Emotion co-occurrence (top-3)",
    )
    _plot_emotion_cooccurrence(
        cosrec_df,
        family_dirs["emotion"] / "emotion_cooccurrence_cosrec.png",
        "CoSRec: Emotion co-occurrence (top-3)",
    )

    # Popularity bias graphs.
    _plot_violin_by_emotion(
        redial_df,
        "pop_mean_pct",
        family_dirs["popularity"] / "pop_mean_pct_by_emotion_redial.png",
        "ReDial: Mean popularity percentile by emotion",
    )
    _plot_violin_by_emotion(
        cosrec_df,
        "pop_mean_pct",
        family_dirs["popularity"] / "pop_mean_pct_by_emotion_cosrec.png",
        "CoSRec: Mean popularity percentile by emotion",
    )
    _plot_violin_by_emotion(
        redial_df,
        "pi",
        family_dirs["popularity"] / "pi_by_emotion_redial.png",
        "ReDial: PI by emotion",
    )
    _plot_violin_by_emotion(
        cosrec_df,
        "pi",
        family_dirs["popularity"] / "pi_by_emotion_cosrec.png",
        "CoSRec: PI by emotion",
    )
    _plot_threshold_profile_by_emotion(
        redial_df,
        family_dirs["popularity"] / "threshold_profile_redial.png",
        "ReDial: P@1/5/10 profile by emotion",
    )
    _plot_threshold_profile_by_emotion(
        cosrec_df,
        family_dirs["popularity"] / "threshold_profile_cosrec.png",
        "CoSRec: P@1/5/10 profile by emotion",
    )

    redial_emo_labels = sorted([c[len("emo_") :] for c in redial_df.columns if c.startswith("emo_")])
    cosrec_emo_labels = sorted([c[len("emo_") :] for c in cosrec_df.columns if c.startswith("emo_")])
    pop_score_dir_redial = family_dirs["popularity"] / "score_vs_pop_redial"
    pop_score_dir_cosrec = family_dirs["popularity"] / "score_vs_pop_cosrec"
    pi_score_dir_redial = family_dirs["popularity"] / "score_vs_pi_redial"
    pi_score_dir_cosrec = family_dirs["popularity"] / "score_vs_pi_cosrec"
    _ensure_dir(pop_score_dir_redial)
    _ensure_dir(pop_score_dir_cosrec)
    _ensure_dir(pi_score_dir_redial)
    _ensure_dir(pi_score_dir_cosrec)
    for emo in redial_emo_labels:
        _plot_score_vs_metric(
            redial_df,
            emo,
            "pop_mean_pct",
            pop_score_dir_redial / f"{emo}.png",
            f"ReDial: {emo} score vs popularity",
        )
        _plot_score_vs_metric(
            redial_df,
            emo,
            "pi",
            pi_score_dir_redial / f"{emo}.png",
            f"ReDial: {emo} score vs PI",
        )
    for emo in cosrec_emo_labels:
        _plot_score_vs_metric(
            cosrec_df,
            emo,
            "pop_mean_pct",
            pop_score_dir_cosrec / f"{emo}.png",
            f"CoSRec: {emo} score vs popularity",
        )
        _plot_score_vs_metric(
            cosrec_df,
            emo,
            "pi",
            pi_score_dir_cosrec / f"{emo}.png",
            f"CoSRec: {emo} score vs PI",
        )

    # Episode popularity dynamics.
    _plot_violin_by_emotion(
        redial_df,
        "cep",
        family_dirs["episode_popularity"] / "cep_by_emotion_redial.png",
        "ReDial: CEP by emotion",
    )
    _plot_violin_by_emotion(
        cosrec_df,
        "cep",
        family_dirs["episode_popularity"] / "cep_by_emotion_cosrec.png",
        "CoSRec: CEP by emotion",
    )
    _plot_violin_by_emotion(
        redial_df,
        "uiop",
        family_dirs["episode_popularity"] / "uiop_by_emotion_redial.png",
        "ReDial: UIOP by emotion",
    )
    _plot_violin_by_emotion(
        cosrec_df,
        "uiop",
        family_dirs["episode_popularity"] / "uiop_by_emotion_cosrec.png",
        "CoSRec: UIOP by emotion",
    )
    cep_uiop_dir_redial = family_dirs["episode_popularity"] / "cep_uiop_density_redial"
    cep_uiop_dir_cosrec = family_dirs["episode_popularity"] / "cep_uiop_density_cosrec"
    _plot_density_by_emotion(
        redial_df,
        "cep",
        "uiop",
        cep_uiop_dir_redial,
        "ReDial: CEP vs UIOP",
    )
    _plot_density_by_emotion(
        cosrec_df,
        "cep",
        "uiop",
        cep_uiop_dir_cosrec,
        "CoSRec: CEP vs UIOP",
    )
    _plot_turn_trajectory_by_emotion(
        redial_df,
        "pop_mean_pct",
        "rec_turn_order",
        family_dirs["episode_popularity"] / "popularity_trajectory_redial.png",
        "ReDial: Popularity trajectory by emotion",
    )
    _plot_turn_trajectory_by_emotion(
        cosrec_df,
        "pop_mean_pct",
        "rec_turn_order",
        family_dirs["episode_popularity"] / "popularity_trajectory_cosrec.png",
        "CoSRec: Popularity trajectory by emotion",
    )

    # Genre/category bias graphs.
    _plot_metric_scatter_by_emotion(
        redial_df,
        "genre_entropy",
        "genre_js",
        family_dirs["genre"] / "entropy_vs_js_redial.png",
        "ReDial: Genre entropy vs JS divergence",
    )
    _plot_metric_scatter_by_emotion(
        cosrec_df,
        "genre_entropy",
        "genre_js",
        family_dirs["genre"] / "entropy_vs_js_cosrec.png",
        "CoSRec: Category entropy vs JS divergence",
    )
    _plot_simple_violin_from_pairs(
        redial_genre_stats["top_shares"],
        family_dirs["genre"] / "top_genre_share_redial.png",
        "ReDial: Top genre share by emotion",
        "Top genre share",
    )
    _plot_simple_violin_from_pairs(
        cosrec_genre_stats["top_shares"],
        family_dirs["genre"] / "top_genre_share_cosrec.png",
        "CoSRec: Top category share by emotion",
        "Top category share",
    )
    _plot_simple_violin_from_pairs(
        redial_genre_stats["effective_nums"],
        family_dirs["genre"] / "effective_genres_redial.png",
        "ReDial: Effective number of genres by emotion",
        "exp(entropy)",
    )
    _plot_simple_violin_from_pairs(
        cosrec_genre_stats["effective_nums"],
        family_dirs["genre"] / "effective_genres_cosrec.png",
        "CoSRec: Effective number of categories by emotion",
        "exp(entropy)",
    )
    _plot_genre_lift_heatmap(
        redial_genre_stats,
        family_dirs["genre"] / "genre_lift_heatmap_redial.png",
        "ReDial: Genre lift by emotion",
    )
    _plot_genre_lift_heatmap(
        cosrec_genre_stats,
        family_dirs["genre"] / "genre_lift_heatmap_cosrec.png",
        "CoSRec: Category lift by emotion",
    )
    _plot_topk_uplift_dots(
        redial_genre_stats,
        family_dirs["genre"] / "top_uplift_redial",
        "ReDial",
    )
    _plot_topk_uplift_dots(
        cosrec_genre_stats,
        family_dirs["genre"] / "top_uplift_cosrec",
        "CoSRec",
    )
    _plot_alluvial_emotion_to_category(
        redial_genre_stats,
        family_dirs["genre"] / "emotion_to_genre_sankey_redial.png",
        "ReDial: Emotion → dominant genre",
    )
    _plot_alluvial_emotion_to_category(
        cosrec_genre_stats,
        family_dirs["genre"] / "emotion_to_genre_sankey_cosrec.png",
        "CoSRec: Emotion → dominant category",
    )

    # Year/decade bias (ReDial).
    _plot_year_ridgeline(
        redial_df,
        family_dirs["year_decade"] / "mean_year_ridgeline_redial.png",
        "ReDial: Mean release year by emotion",
    )
    _plot_decade_stacked_bars(
        redial_decade_stats,
        family_dirs["year_decade"] / "decade_distribution_redial.png",
        "ReDial: Decade distribution by emotion",
    )
    _plot_metric_scatter_by_emotion(
        redial_df,
        "pop_mean_pct",
        "mean_year",
        family_dirs["year_decade"] / "year_vs_popularity_redial.png",
        "ReDial: Year skew vs popularity",
    )

    # Rating bias (CoSRec).
    _plot_violin_by_emotion(
        cosrec_df,
        "rating_mean_pct",
        family_dirs["rating"] / "rating_pct_by_emotion_cosrec.png",
        "CoSRec: Rating percentile by emotion",
    )
    _plot_popularity_rating_joint(
        cosrec_df,
        family_dirs["rating"] / "popularity_vs_rating_cosrec.png",
        "CoSRec: Popularity vs rating (colored by emotion)",
        use_percentile=True,
        show_quadrants=True,
    )
    _plot_quadrant_heatmap(
        cosrec_df,
        "pop_mean_pct",
        "rating_mean_pct",
        family_dirs["rating"] / "rating_popularity_quadrants_cosrec.png",
        "CoSRec: Quadrant proportions by emotion",
    )

    # Redundancy bias.
    _plot_violin_by_emotion(
        redial_df,
        "redundancy_repeated",
        family_dirs["redundancy"] / "repeat_count_redial.png",
        "ReDial: Repeated items by emotion",
    )
    _plot_violin_by_emotion(
        cosrec_df,
        "redundancy_repeated",
        family_dirs["redundancy"] / "repeat_count_cosrec.png",
        "CoSRec: Repeated items by emotion",
    )
    _plot_repeat_rate_bar(
        redial_df,
        family_dirs["redundancy"] / "repeat_rate_redial.png",
        "ReDial: P(repeated > 0) by emotion",
    )
    _plot_repeat_rate_bar(
        cosrec_df,
        family_dirs["redundancy"] / "repeat_rate_cosrec.png",
        "CoSRec: P(repeated > 0) by emotion",
    )
    _plot_turn_trajectory_by_emotion(
        redial_redundancy_progress,
        "unique_items_so_far",
        "rec_turn_order",
        family_dirs["redundancy"] / "unique_items_trajectory_redial.png",
        "ReDial: Cumulative unique items by emotion",
    )
    _plot_turn_trajectory_by_emotion(
        cosrec_redundancy_progress,
        "unique_items_so_far",
        "rec_turn_order",
        family_dirs["redundancy"] / "unique_items_trajectory_cosrec.png",
        "CoSRec: Cumulative unique items by emotion",
    )
    _plot_pareto_repeated_items(
        redial_repeat_counts,
        family_dirs["redundancy"] / "repeat_pareto_redial.png",
        "ReDial: Repeat-item Pareto by emotion",
    )
    _plot_pareto_repeated_items(
        cosrec_repeat_counts,
        family_dirs["redundancy"] / "repeat_pareto_cosrec.png",
        "CoSRec: Repeat-item Pareto by emotion",
    )

    # Exposure concentration.
    _plot_exposure_emotion_curves(
        redial_exposure_items["counts"],
        redial_exposure_emotions["counts"],
        redial_exposure_items["percentiles"],
        redial_exposure_emotions["episodes"],
        family_dirs["exposure"] / "exposure_gini_redial.png",
        "ReDial: Exposure vs popularity (Gini)",
        metric="gini",
    )
    _plot_exposure_emotion_curves(
        cosrec_exposure_items["counts"],
        cosrec_exposure_emotions["counts"],
        cosrec_exposure_items["percentiles"],
        cosrec_exposure_emotions["episodes"],
        family_dirs["exposure"] / "exposure_gini_cosrec.png",
        "CoSRec: Exposure vs popularity (Gini)",
        metric="gini",
    )
    _plot_exposure_topk_share_by_emotion(
        redial_exposure_emotions["counts"],
        redial_exposure_emotions["episodes"],
        family_dirs["exposure"] / "exposure_topk_share_redial.png",
        "ReDial: Exposure top-K share by emotion",
    )
    _plot_exposure_topk_share_by_emotion(
        cosrec_exposure_emotions["counts"],
        cosrec_exposure_emotions["episodes"],
        family_dirs["exposure"] / "exposure_topk_share_cosrec.png",
        "CoSRec: Exposure top-K share by emotion",
    )
    _plot_gini_hhi_scatter(
        redial_exp_df,
        family_dirs["exposure"] / "gini_hhi_scatter_redial.png",
        "ReDial: Gini vs HHI by emotion",
    )
    _plot_gini_hhi_scatter(
        cosrec_exp_df,
        family_dirs["exposure"] / "gini_hhi_scatter_cosrec.png",
        "CoSRec: Gini vs HHI by emotion",
    )

    # Stereotype/biasful language.
    _plot_stereotype_by_emotion(
        redial_df,
        family_dirs["stereotype"] / "stereotype_by_emotion_redial.png",
        "ReDial: Stereotype by emotion (top-1)",
    )
    _plot_stereotype_by_emotion(
        cosrec_df,
        family_dirs["stereotype"] / "stereotype_by_emotion_cosrec.png",
        "CoSRec: Stereotype by emotion (top-1)",
    )
    _plot_bias_rate_by_turn(
        redial_df,
        family_dirs["stereotype"] / "bias_rate_by_turn_redial.png",
        "ReDial: Bias rate by conversation position",
        turn_col="rec_turn_order",
    )
    _plot_bias_rate_by_turn(
        cosrec_df,
        family_dirs["stereotype"] / "bias_rate_by_turn_cosrec.png",
        "CoSRec: Bias rate by conversation position",
        turn_col="rec_turn_order",
    )
    _plot_bias_vs_stereotype(
        redial_df,
        ["pop_mean_pct", "genre_js", "redundancy_repeated"],
        family_dirs["stereotype"] / "bias_vs_stereotype_redial.png",
        "ReDial: Bias metrics vs stereotype label",
    )
    _plot_bias_vs_stereotype(
        cosrec_df,
        ["pop_mean_pct", "genre_js", "redundancy_repeated"],
        family_dirs["stereotype"] / "bias_vs_stereotype_cosrec.png",
        "CoSRec: Bias metrics vs stereotype label",
    )
    _plot_stereotype_score_by_emotion(
        redial_df,
        family_dirs["stereotype"] / "stereotype_score_by_emotion_redial.png",
        "ReDial: LABEL_1 score by emotion",
    )
    _plot_stereotype_score_by_emotion(
        cosrec_df,
        family_dirs["stereotype"] / "stereotype_score_by_emotion_cosrec.png",
        "CoSRec: LABEL_1 score by emotion",
    )

    # Cross-bias summary visuals.
    _plot_effect_size_heatmap(
        redial_df,
        redial_metrics,
        family_dirs["summary"] / "effect_size_heatmap_redial.png",
        "ReDial: Emotion × metric effect sizes",
    )
    _plot_effect_size_heatmap(
        cosrec_df,
        cosrec_metrics,
        family_dirs["summary"] / "effect_size_heatmap_cosrec.png",
        "CoSRec: Emotion × metric effect sizes",
    )
    _plot_parallel_coordinates(
        redial_df,
        redial_metrics,
        family_dirs["summary"] / "parallel_coordinates_redial.png",
        "ReDial: Parallel coordinates (bias profiles)",
    )
    _plot_parallel_coordinates(
        cosrec_df,
        cosrec_metrics,
        family_dirs["summary"] / "parallel_coordinates_cosrec.png",
        "CoSRec: Parallel coordinates (bias profiles)",
    )
    _plot_radar_per_emotion(
        redial_df,
        ["pop_mean_pct", "p05", "cep", "uiop", "genre_js", "year_decade_js", "redundancy_repeated"],
        family_dirs["summary"] / "radar_redial",
        "ReDial radar",
    )
    _plot_radar_per_emotion(
        cosrec_df,
        ["pop_mean_pct", "p05", "cep", "uiop", "genre_js", "rating_mean_pct", "redundancy_repeated"],
        family_dirs["summary"] / "radar_cosrec",
        "CoSRec radar",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stats and figures for bias/emotion analysis.")
    parser.add_argument("--cache-root", default=str(Path(__file__).resolve().parents[1] / "cache"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--splits", default="train,test")
    args = parser.parse_args()

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    analyze_stats(cache_root=args.cache_root, out_dir=args.out_dir, splits=splits)


if __name__ == "__main__":
    main()
