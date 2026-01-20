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


def _plot_emotion_distribution(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    counts = df["emotion_top1"].value_counts().sort_values(ascending=False)
    if counts.empty:
        return
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = counts.index.tolist()
    values = counts.values
    y = np.arange(len(labels))
    ax.hlines(y=y, xmin=0, xmax=values, color="#c7c7c7", linewidth=1.0)
    ax.plot(values, y, "o", color="#4c78a8", markersize=5)
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x")
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
    labels = ["LABEL_0", "LABEL_1"]
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


def _plot_exposure_by_emotion(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
    top_n: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    if df.empty or metric not in df or "emotion" not in df:
        return
    plot_df = df.sort_values("episodes", ascending=False).head(top_n)
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("episodes", ascending=True)
    labels = [f"{row.emotion} (n={int(row.episodes)})" for row in plot_df.itertuples()]
    values = plot_df[metric].tolist()
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel(metric.upper())
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_exposure_popularity_curve(
    exposure_counts: Counter,
    popularity_percentiles: Dict[Any, float],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

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
        return

    pairs.sort(key=lambda x: x[0])
    total = sum(c for _, c in pairs)
    if total <= 0:
        return

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

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(xs, ys, color="#4c78a8", linewidth=2)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#999999", linewidth=1)
    ax.set_title(f"{title} (Gini={gini:.2f})")
    ax.set_xlabel("Popularity percentile (items sorted by popularity)")
    ax.set_ylabel("Cumulative exposure share")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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
                f"LABEL_0={rate_0:.1f}, LABEL_1={rate_1:.1f}"
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

                stereo_label = None
                if isinstance(stereo_obj, list) and stereo_obj:
                    lab = stereo_obj[0].get("label")
                    if lab == "LABEL_1":
                        stereo_label = 1
                    elif lab == "LABEL_0":
                        stereo_label = 0

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
    from dataset.amazon_reviews import AmazonReviews2023Index
    from dataset.cosrec import CoSRecDataset, safe_id

    ds = CoSRecDataset(cache_root=cache_root)
    amazon = AmazonReviews2023Index(cache_root=cache_root)
    amazon.ensure_index()
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
        for asin in pop_items:
            meta = amazon.get(asin)
            if not meta:
                continue
            try:
                num = int(meta.get("num_ratings", 0))
            except Exception:
                continue
            exposure_items[asin] += 1
            if asin not in exposure_popularity:
                exposure_popularity[asin] = float(amazon.popularity_percentile(num))

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
        stereo_label = None
        if isinstance(stereo_obj, list) and stereo_obj:
            lab = stereo_obj[0].get("label")
            if lab == "LABEL_1":
                stereo_label = 1
            elif lab == "LABEL_0":
                stereo_label = 0

        record = {
            "dataset": "cosrec",
            "conversation_id": conv_id,
            "topic_id": str(ep.topic_id),
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
        }
        for lab, score in emo_scores.items():
            record[f"emo_{lab}"] = score
        records.append(record)

        if emo_top1 and pop_items:
            exposure_episodes[emo_top1] += 1
            exposure_counts[emo_top1].update(pop_items)

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

    redial_df, redial_exposure, redial_exposure_emotions, redial_exposure_items = _load_redial(
        cache_root, splits
    )
    cosrec_df, cosrec_exposure, cosrec_exposure_emotions, cosrec_exposure_items = _load_cosrec(cache_root)

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

    _plot_emotion_distribution(
        redial_df,
        figs_dir / "emotion_top1_redial.png",
        "ReDial: Top-1 emotion distribution",
    )
    _plot_emotion_distribution(
        cosrec_df,
        figs_dir / "emotion_top1_cosrec.png",
        "CoSRec: Top-1 emotion distribution",
    )

    _plot_threshold_box(
        redial_df,
        figs_dir / "popularity_thresholds_redial.png",
        "ReDial: Popularity coverage at thresholds",
    )
    _plot_threshold_box(
        cosrec_df,
        figs_dir / "popularity_thresholds_cosrec.png",
        "CoSRec: Popularity coverage at thresholds",
    )

    for metric in ["pi", "cep", "uiop"]:
        _plot_metric_hist(
            redial_df,
            metric,
            figs_dir / f"{metric}_redial.png",
            f"ReDial: {metric} distribution",
        )
        _plot_metric_hist(
            cosrec_df,
            metric,
            figs_dir / f"{metric}_cosrec.png",
            f"CoSRec: {metric} distribution",
        )

    _plot_metric_hist(
        redial_df,
        "pop_mean_pct",
        figs_dir / "pop_mean_pct_redial.png",
        "ReDial: Mean popularity percentile",
    )
    _plot_metric_hist(
        cosrec_df,
        "pop_mean_pct",
        figs_dir / "pop_mean_pct_cosrec.png",
        "CoSRec: Mean popularity percentile",
    )

    _plot_metric_hist(
        redial_df,
        "genre_js",
        figs_dir / "genre_js_redial.png",
        "ReDial: Genre JS divergence",
    )
    _plot_metric_hist(
        cosrec_df,
        "genre_js",
        figs_dir / "genre_js_cosrec.png",
        "CoSRec: Genre JS divergence",
    )

    _plot_stereotype_distribution(
        redial_df,
        figs_dir / "stereotype_redial.png",
        "ReDial: Stereotype labels",
    )
    _plot_stereotype_distribution(
        cosrec_df,
        figs_dir / "stereotype_cosrec.png",
        "CoSRec: Stereotype labels",
    )

    _plot_exposure_popularity_curve(
        redial_exposure_items["counts"],
        redial_exposure_items["percentiles"],
        figs_dir / "exposure_gini_redial.png",
        "ReDial: Exposure vs popularity",
    )
    _plot_exposure_popularity_curve(
        cosrec_exposure_items["counts"],
        cosrec_exposure_items["percentiles"],
        figs_dir / "exposure_gini_cosrec.png",
        "CoSRec: Exposure vs popularity",
    )

    _plot_exposure_by_emotion(
        redial_exp_df,
        "gini",
        figs_dir / "exposure_gini_by_emotion_redial.png",
        "ReDial: Exposure gini by emotion",
    )
    _plot_exposure_by_emotion(
        redial_exp_df,
        "hhi",
        figs_dir / "exposure_hhi_by_emotion_redial.png",
        "ReDial: Exposure HHI by emotion",
    )
    _plot_exposure_by_emotion(
        cosrec_exp_df,
        "gini",
        figs_dir / "exposure_gini_by_emotion_cosrec.png",
        "CoSRec: Exposure gini by emotion",
    )
    _plot_exposure_by_emotion(
        cosrec_exp_df,
        "hhi",
        figs_dir / "exposure_hhi_by_emotion_cosrec.png",
        "CoSRec: Exposure HHI by emotion",
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
