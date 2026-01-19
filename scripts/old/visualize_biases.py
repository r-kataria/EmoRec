#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# ----------------------------
# Styling (simple ACM-ish: grayscale, serif, clean)
# ----------------------------
def set_acm_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
        }
    )


# ----------------------------
# Basic IO
# ----------------------------
def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def glob_jsons(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("*.json") if p.is_file()])


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


# ----------------------------
# Stats helpers
# ----------------------------
def quantiles(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    idx = int(q * (len(s) - 1))
    return float(s[idx])


def stats(vals: List[float]) -> Dict[str, Optional[float]]:
    clean = [v for v in vals if v is not None]
    clean = [float(v) for v in clean if float(v) == float(v)]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    s = sorted(clean)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {
        "n": n,
        "mean": float(mean),
        "median": float(median),
        "p25": quantiles(s, 0.25),
        "p75": quantiles(s, 0.75),
    }


def gini(values: List[int]) -> float:
    if not values:
        return 0.0
    x = sorted([int(v) for v in values if int(v) >= 0])
    n = len(x)
    s = sum(x)
    if s == 0 or n == 0:
        return 0.0
    num = 0.0
    for i, v in enumerate(x, start=1):
        num += i * v
    return float((2.0 * num) / (n * s) - (n + 1) / n)


def hhi(values: List[int]) -> float:
    s = sum(values)
    if s == 0:
        return 0.0
    return float(sum((v / s) ** 2 for v in values))


def lorenz_curve(values: List[int]) -> Tuple[List[float], List[float]]:
    """Returns (population_share, value_share). Includes (0,0)."""
    vals = sorted([int(v) for v in values if int(v) >= 0])
    n = len(vals)
    if n == 0:
        return [0.0, 1.0], [0.0, 1.0]
    total = sum(vals)
    if total == 0:
        return [0.0, 1.0], [0.0, 1.0]
    cum = 0
    xs = [0.0]
    ys = [0.0]
    for i, v in enumerate(vals, start=1):
        cum += v
        xs.append(i / n)
        ys.append(cum / total)
    return xs, ys


# ----------------------------
# Plot helpers (grayscale)
# ----------------------------
def plot_boxplot(groups: List[Tuple[str, List[float]]], title: str, ylabel: str, out_path: Path) -> None:
    data = [g[1] for g in groups if g[1]]
    labels = [g[0] for g in groups if g[1]]
    if not data:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cdf(groups: List[Tuple[str, List[float], str]], title: str, xlabel: str, out_path: Path) -> None:
    """
    groups: (label, values, linestyle)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plotted = False
    for label, vals, ls in groups:
        if not vals:
            continue
        s = sorted(vals)
        n = len(s)
        ys = [(i + 1) / n for i in range(n)]
        plt.plot(s, ys, linestyle=ls, color="black", label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_grouped_bars(categories: List[str], series: List[Tuple[str, List[float]]], title: str, ylabel: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not categories:
        return
    k = len(series)
    if k == 0:
        return

    x = list(range(len(categories)))
    width = 0.8 / max(1, k)
    offsets = [(i - (k - 1) / 2) * width for i in range(k)]

    plt.figure(figsize=(max(8, len(categories) * 0.9), 5))
    for idx, (label, vals) in enumerate(series):
        if len(vals) != len(categories):
            continue
        plt.bar([xi + offsets[idx] for xi in x], vals, width=width, color=str(0.2 + 0.2 * idx), label=label)

    plt.xticks(x, categories, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_lorenz(groups: List[Tuple[str, List[int], str]], title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for label, vals, ls in groups:
        xs, ys = lorenz_curve(vals)
        plt.plot(xs, ys, linestyle=ls, color="black", label=label)
    # equality line
    plt.plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=1.0)
    plt.title(title)
    plt.xlabel("Cumulative share of items")
    plt.ylabel("Cumulative share of mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_hbar(labels: List[str], values: List[float], title: str, xlabel: str, out_path: Path) -> None:
    if not labels or not values:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, max(4, 0.35 * len(labels))))
    y = list(range(len(labels)))[::-1]
    plt.barh(y, values[::-1], color="0.3")
    plt.yticks(y, labels[::-1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------
# Load MovieLens metadata
# ----------------------------
def load_rating_counts(cache_root: Path) -> Dict[int, int]:
    p = cache_root / "datasets" / "movielens" / "ml-25m" / "rating_counts.json"
    if not p.exists():
        return {}
    raw = read_json(p)
    return {int(k): int(v) for k, v in raw.items()}


def load_movies_titles(cache_root: Path) -> Dict[int, str]:
    # extracted file: <root>/datasets/movielens/ml-25m/ml-25m/movies.csv
    p = cache_root / "datasets" / "movielens" / "ml-25m" / "ml-25m" / "movies.csv"
    if not p.exists():
        return {}
    out: Dict[int, str] = {}
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mid = int(row["movieId"])
                out[mid] = row.get("title", "") or str(mid)
            except Exception:
                continue
    return out


def load_genre_baseline(cache_root: Path) -> Dict[str, float]:
    p = cache_root / "datasets" / "movielens" / "ml-25m" / "genre_baseline.json"
    if not p.exists():
        return {}
    raw = read_json(p)
    return {str(k): float(v) for k, v in raw.items()}


def percentile_from_sorted(sorted_vals: List[int], x: int) -> float:
    if not sorted_vals:
        return 0.0
    i = bisect.bisect_left(sorted_vals, int(x))
    return float(i) / float(len(sorted_vals))


def threshold_from_sorted(sorted_vals: List[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(q * (len(sorted_vals) - 1))
    return int(sorted_vals[idx])


# ----------------------------
# Scan cached popularity bias (turn-level + item-level)
# ----------------------------
def scan_popularity(cache_root: Path, split: str) -> Dict[str, Any]:
    base = cache_root / "bias" / "popularity" / "redial" / "ml-25m" / split
    files = glob_jsons(base)

    # per-conv summary metrics already computed by PopularityBias
    conv_mean_pct = []
    conv_mean_count = []

    # per-turn (Recommender turns) mean percentile, mean count, head share
    turn_mean_pct = []
    turn_mean_count = []
    turn_head10 = []
    turn_novelty = []

    # per-item percentiles and counts across all recommender turns
    item_pct = []
    item_count = []

    # mapping coverage
    total_redial_mentions = 0
    total_mapped_mentions = 0
    total_unmapped_mentions = 0

    # movieId set for restricted baseline
    mapped_ids_set = set()

    # for per-turn restricted computation later
    per_turn_counts: List[List[int]] = []  # list of counts per recommender turn

    convs_with_mapped_recs = 0

    for fp in files:
        try:
            rec = read_json(fp)
        except Exception:
            continue

        s = rec.get("summary", {}) or {}
        mp = safe_float(s.get("mean_percentile_all"))
        mc = safe_float(s.get("mean_count_all"))
        n_mapped = s.get("num_movies_mapped_total") or 0
        if n_mapped and n_mapped > 0:
            convs_with_mapped_recs += 1
            if mp is not None:
                conv_mean_pct.append(mp)
            if mc is not None:
                conv_mean_count.append(mc)

        turns = rec.get("turns", []) or []
        for t in turns:
            if t.get("speaker") != "Recommender":
                continue

            redial_ids = t.get("redial_movie_ids") or []
            ml_ids = t.get("movielens_movie_ids") or []
            unmapped = t.get("unmapped_redial_movie_ids") or []
            total_redial_mentions += len(redial_ids)
            total_mapped_mentions += len(ml_ids)
            total_unmapped_mentions += len(unmapped)

            for mid in ml_ids:
                try:
                    mapped_ids_set.add(int(mid))
                except Exception:
                    pass

            pop = t.get("popularity")
            if not pop:
                continue

            counts = pop.get("counts") or []
            pcts = pop.get("percentiles") or []
            if counts:
                per_turn_counts.append([int(c) for c in counts])
                item_count.extend([int(c) for c in counts])
            if pcts:
                item_pct.extend([float(p) for p in pcts])

            mpt = safe_float(pop.get("mean_percentile"))
            mct = safe_float(pop.get("mean_count"))
            h10 = safe_float(pop.get("head_share_top10pct"))
            nov = safe_float(pop.get("novelty_mean"))

            if mpt is not None:
                turn_mean_pct.append(mpt)
            if mct is not None:
                turn_mean_count.append(mct)
            if h10 is not None:
                turn_head10.append(h10)
            if nov is not None:
                turn_novelty.append(nov)

    return {
        "cached_conversations": len(files),
        "conversations_with_mapped_recs": convs_with_mapped_recs,
        "conv_mean_percentile_all": conv_mean_pct,
        "conv_mean_count_all": conv_mean_count,
        "turn_mean_percentile": turn_mean_pct,
        "turn_mean_count": turn_mean_count,
        "turn_head_share_top10_global": turn_head10,
        "turn_novelty_mean": turn_novelty,
        "item_percentile_global": item_pct,
        "item_count": item_count,
        "per_turn_counts": per_turn_counts,
        "mapping": {
            "total_redial_mentions": total_redial_mentions,
            "total_mapped_mentions": total_mapped_mentions,
            "total_unmapped_mentions": total_unmapped_mentions,
        },
        "mapped_ids_set": mapped_ids_set,
    }


# ----------------------------
# Scan redundancy cache
# ----------------------------
def scan_redundancy(cache_root: Path, split: str) -> Dict[str, Any]:
    base = cache_root / "bias" / "redundancy" / "redial" / "ml-25m" / split
    files = glob_jsons(base)

    redundancy_rates = []
    total_items = []
    convs_with_items = 0
    convs_with_repeats = 0

    # aggregate repeated items (count repeats beyond first)
    repeated_excess = Counter()

    # per-turn repeated items count (recommender turns)
    repeated_items_per_turn = []

    for fp in files:
        try:
            rec = read_json(fp)
        except Exception:
            continue
        s = rec.get("summary", {}) or {}
        tot = s.get("total_items") or 0
        rr = safe_float(s.get("redundancy_rate"))

        if tot and tot > 0:
            convs_with_items += 1
            total_items.append(float(tot))
            if rr is not None:
                redundancy_rates.append(rr)
            if rr is not None and rr > 0:
                convs_with_repeats += 1

        # item-level repeats
        ric = s.get("repeated_item_counts") or {}
        if isinstance(ric, dict):
            for mid_str, cnt in ric.items():
                try:
                    cnt_i = int(cnt)
                    # "excess repeats" = occurrences beyond the first
                    excess = max(0, cnt_i - 1)
                    if excess > 0:
                        repeated_excess[int(mid_str)] += excess
                except Exception:
                    continue

        # per-turn repeats
        turns = rec.get("turns", []) or []
        for t in turns:
            if t.get("speaker") != "Recommender":
                continue
            rep = t.get("repeated_items") or []
            if isinstance(rep, list):
                repeated_items_per_turn.append(len(rep))

    total_repeat_events = int(sum(repeated_excess.values()))
    unique_repeated_items = int(len(repeated_excess))

    return {
        "cached_conversations": len(files),
        "conversations_with_items": convs_with_items,
        "conversations_with_repeats": convs_with_repeats,
        "redundancy_rate": redundancy_rates,
        "total_items": total_items,
        "repeated_excess_counts": repeated_excess,
        "total_repeat_events": total_repeat_events,
        "unique_repeated_items": unique_repeated_items,
        "repeated_items_per_turn": repeated_items_per_turn,
    }


# ----------------------------
# Scan genre cache
# ----------------------------
def scan_genre(cache_root: Path, split: str) -> Dict[str, Any]:
    base = cache_root / "bias" / "genre" / "redial" / "ml-25m" / split
    files = glob_jsons(base)

    jsd_vals = []
    convs_with_items = 0

    # weighted aggregate genre dist by total_items
    agg = defaultdict(float)
    wsum = 0.0

    # per-turn JSD distribution (recommender turns)
    turn_jsd = []

    for fp in files:
        try:
            rec = read_json(fp)
        except Exception:
            continue
        s = rec.get("summary", {}) or {}
        tot = s.get("total_items") or 0
        jsd = safe_float(s.get("js_divergence_vs_catalog"))
        dist = s.get("genres_dist") or {}

        if tot and tot > 0:
            convs_with_items += 1

        if jsd is not None:
            jsd_vals.append(jsd)

        if isinstance(dist, dict) and tot and tot > 0:
            w = float(tot)
            wsum += w
            for g, v in dist.items():
                try:
                    agg[str(g)] += w * float(v)
                except Exception:
                    continue

        # per-turn
        for t in rec.get("turns", []) or []:
            if t.get("speaker") != "Recommender":
                continue
            gb = t.get("genre_bias") or {}
            if isinstance(gb, dict):
                tj = safe_float(gb.get("js_divergence_vs_catalog"))
                if tj is not None:
                    turn_jsd.append(tj)

    agg_norm = {g: (agg[g] / wsum) for g in agg} if wsum > 0 else {}

    return {
        "cached_conversations": len(files),
        "conversations_with_items": convs_with_items,
        "js_divergence_vs_catalog": jsd_vals,
        "genres_dist_weighted": agg_norm,
        "turn_jsd": turn_jsd,
    }


# ----------------------------
# Exposure concentration (derive from popularity caches so we can do top% shares & Lorenz)
# ----------------------------
def exposure_from_popularity(cache_root: Path, split: str) -> Dict[str, Any]:
    base = cache_root / "bias" / "popularity" / "redial" / "ml-25m" / split
    files = glob_jsons(base)
    counts = Counter()

    for fp in files:
        try:
            rec = read_json(fp)
        except Exception:
            continue
        for t in rec.get("turns", []) or []:
            if t.get("speaker") != "Recommender":
                continue
            for mid in t.get("movielens_movie_ids") or []:
                try:
                    counts[int(mid)] += 1
                except Exception:
                    continue

    freqs = list(counts.values())
    total_mentions = int(sum(freqs))
    unique_items = int(len(freqs))
    top_sorted = sorted(freqs, reverse=True)

    def top_k_share(k: int) -> float:
        if total_mentions == 0:
            return 0.0
        return float(sum(top_sorted[: min(k, len(top_sorted))]) / total_mentions)

    def top_p_share(p: float) -> float:
        if total_mentions == 0 or unique_items == 0:
            return 0.0
        k = int(math.ceil(p * unique_items))
        k = max(1, min(k, unique_items))
        return float(sum(top_sorted[:k]) / total_mentions)

    return {
        "unique_items": unique_items,
        "total_mentions": total_mentions,
        "top10_share": top_k_share(10),
        "top50_share": top_k_share(50),
        "top100_share": top_k_share(100),
        "top1pct_items_share": top_p_share(0.01),
        "top5pct_items_share": top_p_share(0.05),
        "top10pct_items_share": top_p_share(0.10),
        "gini": gini(freqs),
        "hhi": hhi(freqs),
        "freqs": freqs,  # for plots
    }


# ----------------------------
# Popularity baselines (global vs ReDial-restricted)
# ----------------------------
def compute_popularity_baselines(
    rating_counts: Dict[int, int],
    mapped_ids_union: set,
) -> Dict[str, Any]:
    all_counts_sorted = sorted(rating_counts.values())

    mapped_counts = [rating_counts.get(mid, 0) for mid in mapped_ids_union]
    mapped_counts_sorted = sorted(mapped_counts)

    # thresholds by rating-count quantiles
    thr90_all = threshold_from_sorted(all_counts_sorted, 0.90)
    thr95_all = threshold_from_sorted(all_counts_sorted, 0.95)

    thr90_map = threshold_from_sorted(mapped_counts_sorted, 0.90) if mapped_counts_sorted else 0
    thr95_map = threshold_from_sorted(mapped_counts_sorted, 0.95) if mapped_counts_sorted else 0

    return {
        "all_counts_sorted": all_counts_sorted,
        "mapped_counts_sorted": mapped_counts_sorted,
        "thr90_all": thr90_all,
        "thr95_all": thr95_all,
        "thr90_mapped": thr90_map,
        "thr95_mapped": thr95_map,
    }


def enrich_restricted_popularity(
    per_turn_counts: List[List[int]],
    mapped_counts_sorted: List[int],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Returns:
      - item_percentile_restricted (flattened across all items)
      - turn_mean_percentile_restricted (per recommender turn)
      - turn_head_share_top10_restricted (per recommender turn, based on 90th percentile threshold of mapped set)
    """
    item_pct_r = []
    turn_mean_pct_r = []
    turn_head10_r = []

    if not mapped_counts_sorted:
        return item_pct_r, turn_mean_pct_r, turn_head10_r

    thr90 = threshold_from_sorted(mapped_counts_sorted, 0.90)

    for counts in per_turn_counts:
        if not counts:
            continue
        pcts = [percentile_from_sorted(mapped_counts_sorted, c) for c in counts]
        item_pct_r.extend(pcts)
        turn_mean_pct_r.append(sum(pcts) / len(pcts))
        head = sum(1 for c in counts if c >= thr90) / len(counts)
        turn_head10_r.append(head)

    return item_pct_r, turn_mean_pct_r, turn_head10_r


# ----------------------------
# Markdown table builder
# ----------------------------
def fmt(x: Any, nd: int = 3, pct: bool = False) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
        if pct:
            v = 100.0 * v
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


# ----------------------------
# Main summarization + plotting
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, default="../cache")
    ap.add_argument("--out_dir", type=str, default="./cache/viz_bias")
    ap.add_argument("--top_genres", type=int, default=12)
    ap.add_argument("--no_acm_style", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_acm_style:
        set_acm_style()

    # Load ML metadata needed for restricted baselines + titles
    rating_counts = load_rating_counts(cache_root)
    titles = load_movies_titles(cache_root)
    genre_baseline = load_genre_baseline(cache_root)

    # -------- Scan per-split caches --------
    pop_train = scan_popularity(cache_root, "train")
    pop_test = scan_popularity(cache_root, "test")
    red_train = scan_redundancy(cache_root, "train")
    red_test = scan_redundancy(cache_root, "test")
    gen_train = scan_genre(cache_root, "train")
    gen_test = scan_genre(cache_root, "test")

    # Union of mapped ML ids (for restricted baseline)
    mapped_union = set(pop_train["mapped_ids_set"]) | set(pop_test["mapped_ids_set"])

    baselines = compute_popularity_baselines(rating_counts, mapped_union) if rating_counts else {
        "all_counts_sorted": [],
        "mapped_counts_sorted": [],
        "thr90_all": 0,
        "thr95_all": 0,
        "thr90_mapped": 0,
        "thr95_mapped": 0,
    }

    # Restricted percentiles (computed from per-turn counts)
    item_pct_r_train, turn_mean_pct_r_train, turn_head10_r_train = enrich_restricted_popularity(
        pop_train["per_turn_counts"], baselines["mapped_counts_sorted"]
    )
    item_pct_r_test, turn_mean_pct_r_test, turn_head10_r_test = enrich_restricted_popularity(
        pop_test["per_turn_counts"], baselines["mapped_counts_sorted"]
    )

    # Global (catalog) head shares at item-level for top5/top10 by rating-count threshold
    def item_head_shares(item_counts: List[int], thr90: int, thr95: int) -> Dict[str, float]:
        if not item_counts:
            return {"top10pct": 0.0, "top5pct": 0.0}
        n = len(item_counts)
        top10 = sum(1 for c in item_counts if c >= thr90) / n if thr90 > 0 else 0.0
        top5 = sum(1 for c in item_counts if c >= thr95) / n if thr95 > 0 else 0.0
        return {"top10pct": float(top10), "top5pct": float(top5)}

    head_train_global = item_head_shares(pop_train["item_count"], baselines["thr90_all"], baselines["thr95_all"])
    head_test_global = item_head_shares(pop_test["item_count"], baselines["thr90_all"], baselines["thr95_all"])
    head_all_global = item_head_shares(pop_train["item_count"] + pop_test["item_count"], baselines["thr90_all"], baselines["thr95_all"])

    # Restricted head shares based on mapped thresholds
    head_train_restricted = item_head_shares(pop_train["item_count"], baselines["thr90_mapped"], baselines["thr95_mapped"])
    head_test_restricted = item_head_shares(pop_test["item_count"], baselines["thr90_mapped"], baselines["thr95_mapped"])
    head_all_restricted = item_head_shares(pop_train["item_count"] + pop_test["item_count"], baselines["thr90_mapped"], baselines["thr95_mapped"])

    # -------- Exposure concentration derived from popularity caches (term-wise mapping) --------
    exp_train = exposure_from_popularity(cache_root, "train")
    exp_test = exposure_from_popularity(cache_root, "test")
    # "overall"
    exp_all = {
        "unique_items": exp_train["unique_items"] + 0,  # placeholder, replaced below
        "total_mentions": exp_train["total_mentions"] + exp_test["total_mentions"],
    }
    # recompute "all" freqs by combining counts (do it properly)
    # easiest: combine freqs lists (not identical items), but for Lorenz/Gini we need item counts per item.
    # We rebuild from cache by scanning both splits again in one pass.
    # (fast enough and avoids needing raw item->count dict persisted)
    def exposure_all() -> Dict[str, Any]:
        counts = Counter()
        for split in ("train", "test"):
            base = cache_root / "bias" / "popularity" / "redial" / "ml-25m" / split
            for fp in glob_jsons(base):
                try:
                    rec = read_json(fp)
                except Exception:
                    continue
                for t in rec.get("turns", []) or []:
                    if t.get("speaker") != "Recommender":
                        continue
                    for mid in t.get("movielens_movie_ids") or []:
                        try:
                            counts[int(mid)] += 1
                        except Exception:
                            continue
        freqs = list(counts.values())
        total_mentions = int(sum(freqs))
        unique_items = int(len(freqs))
        top_sorted = sorted(freqs, reverse=True)

        def top_k_share(k: int) -> float:
            if total_mentions == 0:
                return 0.0
            return float(sum(top_sorted[: min(k, len(top_sorted))]) / total_mentions)

        def top_p_share(p: float) -> float:
            if total_mentions == 0 or unique_items == 0:
                return 0.0
            k = int(math.ceil(p * unique_items))
            k = max(1, min(k, unique_items))
            return float(sum(top_sorted[:k]) / total_mentions)

        return {
            "unique_items": unique_items,
            "total_mentions": total_mentions,
            "top10_share": top_k_share(10),
            "top50_share": top_k_share(50),
            "top100_share": top_k_share(100),
            "top1pct_items_share": top_p_share(0.01),
            "top5pct_items_share": top_p_share(0.05),
            "top10pct_items_share": top_p_share(0.10),
            "gini": gini(freqs),
            "hhi": hhi(freqs),
            "freqs": freqs,
        }

    exp_all = exposure_all()

    # -------- Genre distributions (overall) + deltas --------
    # Weighted dist per split; combine and renormalize
    def combine_dists(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
        c = defaultdict(float)
        for k, v in a.items():
            c[k] += float(v)
        for k, v in b.items():
            c[k] += float(v)
        s = sum(c.values())
        return {k: (c[k] / s) for k in c} if s > 0 else {}

    rec_genre_overall = combine_dists(gen_train["genres_dist_weighted"], gen_test["genres_dist_weighted"])
    # delta = rec - baseline (both already sum to ~1 over their supports)
    genre_delta = {g: rec_genre_overall.get(g, 0.0) - genre_baseline.get(g, 0.0) for g in set(rec_genre_overall) | set(genre_baseline)}

    top_delta = sorted(genre_delta.items(), key=lambda kv: abs(kv[1]), reverse=True)[: max(10, args.top_genres)]
    top_genres = [g for g, _ in top_delta]

    # -------- Redundancy: top repeated items --------
    rep_all = red_train["repeated_excess_counts"] + red_test["repeated_excess_counts"]
    top_repeated = rep_all.most_common(20)
    top_repeated_labels = [titles.get(mid, str(mid)) for mid, _ in top_repeated]
    top_repeated_vals = [float(cnt) for _, cnt in top_repeated]

    # ----------------------------
    # Plots
    # ----------------------------
    # Popularity: CDF of item popularity percentiles (global ML baseline)
    # Convert to 0..100 for readability
    plot_cdf(
        groups=[
            ("train", [100.0 * v for v in pop_train["item_percentile_global"]], "--"),
            ("test", [100.0 * v for v in pop_test["item_percentile_global"]], ":"),
            ("all", [100.0 * v for v in (pop_train["item_percentile_global"] + pop_test["item_percentile_global"])], "-"),
        ],
        title="Popularity of recommended items (MovieLens percentile, item-level)",
        xlabel="Popularity percentile (0=least-rated, 100=most-rated)",
        out_path=out_dir / "popularity_item_percentile_cdf_global.png",
    )

    # Popularity: restricted baseline CDF
    plot_cdf(
        groups=[
            ("train", [100.0 * v for v in item_pct_r_train], "--"),
            ("test", [100.0 * v for v in item_pct_r_test], ":"),
            ("all", [100.0 * v for v in (item_pct_r_train + item_pct_r_test)], "-"),
        ],
        title="Popularity of recommended items (percentile among ReDial-supported items)",
        xlabel="Popularity percentile within ReDial item-support (0..100)",
        out_path=out_dir / "popularity_item_percentile_cdf_restricted.png",
    )

    # Popularity: boxplot of per-turn mean percentile (global baseline)
    plot_boxplot(
        groups=[
            ("train", [100.0 * v for v in pop_train["turn_mean_percentile"]]),
            ("test", [100.0 * v for v in pop_test["turn_mean_percentile"]]),
            ("all", [100.0 * v for v in (pop_train["turn_mean_percentile"] + pop_test["turn_mean_percentile"])]),
        ],
        title="Popularity bias (per recommender turn)",
        ylabel="Mean popularity percentile per recommender turn (0..100)",
        out_path=out_dir / "popularity_turn_mean_percentile_box.png",
    )

    # Popularity: head share bars (global vs restricted)
    plot_grouped_bars(
        categories=["top10% (by ML popularity)", "top5% (by ML popularity)"],
        series=[
            ("train", [100.0 * head_train_global["top10pct"], 100.0 * head_train_global["top5pct"]]),
            ("test", [100.0 * head_test_global["top10pct"], 100.0 * head_test_global["top5pct"]]),
            ("all", [100.0 * head_all_global["top10pct"], 100.0 * head_all_global["top5pct"]]),
        ],
        title="Head-share of recommended items (MovieLens catalog baseline)",
        ylabel="Share of recommended items (%)",
        out_path=out_dir / "popularity_head_share_global.png",
    )

    plot_grouped_bars(
        categories=["top10% (within ReDial support)", "top5% (within ReDial support)"],
        series=[
            ("train", [100.0 * head_train_restricted["top10pct"], 100.0 * head_train_restricted["top5pct"]]),
            ("test", [100.0 * head_test_restricted["top10pct"], 100.0 * head_test_restricted["top5pct"]]),
            ("all", [100.0 * head_all_restricted["top10pct"], 100.0 * head_all_restricted["top5pct"]]),
        ],
        title="Head-share of recommended items (restricted to ReDial-supported items)",
        ylabel="Share of recommended items (%)",
        out_path=out_dir / "popularity_head_share_restricted.png",
    )

    # Popularity: "Gini graph" via Lorenz curve on rating-counts of recommended items (item-level)
    plot_lorenz(
        groups=[
            ("train (Gini={:.3f})".format(gini(pop_train["item_count"])), pop_train["item_count"], "--"),
            ("test (Gini={:.3f})".format(gini(pop_test["item_count"])), pop_test["item_count"], ":"),
            ("all (Gini={:.3f})".format(gini(pop_train["item_count"] + pop_test["item_count"])), pop_train["item_count"] + pop_test["item_count"], "-"),
        ],
        title="Popularity concentration in recommended items (Lorenz curve over MovieLens rating counts)",
        out_path=out_dir / "popularity_lorenz_rating_counts.png",
    )

    # Redundancy: boxplot + share-with-repeats
    plot_boxplot(
        groups=[
            ("train", red_train["redundancy_rate"]),
            ("test", red_test["redundancy_rate"]),
            ("all", red_train["redundancy_rate"] + red_test["redundancy_rate"]),
        ],
        title="Redundancy bias (conversation-level)",
        ylabel="Redundancy rate (1 - unique/total recommended items)",
        out_path=out_dir / "redundancy_rate_box.png",
    )

    def repeat_share(red: Dict[str, Any]) -> float:
        denom = red.get("conversations_with_items") or 0
        if denom == 0:
            return 0.0
        return float((red.get("conversations_with_repeats") or 0) / denom)

    plot_grouped_bars(
        categories=["% conversations with repeats"],
        series=[
            ("train", [100.0 * repeat_share(red_train)]),
            ("test", [100.0 * repeat_share(red_test)]),
            ("all", [100.0 * repeat_share({
                "conversations_with_items": (red_train["conversations_with_items"] + red_test["conversations_with_items"]),
                "conversations_with_repeats": (red_train["conversations_with_repeats"] + red_test["conversations_with_repeats"]),
            })]),
        ],
        title="Redundancy incidence",
        ylabel="Share of conversations (%)",
        out_path=out_dir / "redundancy_share_with_repeats.png",
    )

    # Redundancy: top repeated items
    plot_hbar(
        labels=top_repeated_labels,
        values=top_repeated_vals,
        title="Top repeated items within dialogues (excess repeat count)",
        xlabel="Repeat events (occurrences beyond first within a dialogue)",
        out_path=out_dir / "redundancy_top_repeated_items.png",
    )

    # Genre: distribution shift chart + JSD boxplot + delta chart
    # Genre mix: choose top genres by recommendation share
    source = rec_genre_overall if rec_genre_overall else genre_baseline
    top_for_mix = sorted(source.items(), key=lambda kv: kv[1], reverse=True)[: max(8, args.top_genres)]
    mix_genres = [g for g, _ in top_for_mix]
    mix_base = [float(genre_baseline.get(g, 0.0)) for g in mix_genres]
    mix_rec = [float(rec_genre_overall.get(g, 0.0)) for g in mix_genres]

    plot_grouped_bars(
        categories=mix_genres,
        series=[("MovieLens catalog", mix_base), ("ReDial recs", mix_rec)],
        title=f"Genre mix: ReDial recommendations vs MovieLens catalog (top {len(mix_genres)})",
        ylabel="Share",
        out_path=out_dir / "genre_mix_top.png",
    )

    plot_boxplot(
        groups=[
            ("train", gen_train["js_divergence_vs_catalog"]),
            ("test", gen_test["js_divergence_vs_catalog"]),
            ("all", gen_train["js_divergence_vs_catalog"] + gen_test["js_divergence_vs_catalog"]),
        ],
        title="Genre distribution shift (JS divergence vs MovieLens catalog)",
        ylabel="JS divergence",
        out_path=out_dir / "genre_jsd_box.png",
    )

    delta_vals = [float(genre_delta.get(g, 0.0)) for g in top_genres]
    # show delta (rec - base) for most-changed genres
    plot_hbar(
        labels=top_genres,
        values=[abs(v) for v in delta_vals],
        title="Largest genre distribution shifts |ReDial - MovieLens|",
        xlabel="Absolute delta in share",
        out_path=out_dir / "genre_delta_abs_top.png",
    )

    # Exposure: topK and top% shares + Lorenz curve
    plot_grouped_bars(
        categories=["top10 items", "top50 items", "top100 items"],
        series=[
            ("train", [100.0 * exp_train["top10_share"], 100.0 * exp_train["top50_share"], 100.0 * exp_train["top100_share"]]),
            ("test", [100.0 * exp_test["top10_share"], 100.0 * exp_test["top50_share"], 100.0 * exp_test["top100_share"]]),
            ("all", [100.0 * exp_all["top10_share"], 100.0 * exp_all["top50_share"], 100.0 * exp_all["top100_share"]]),
        ],
        title="Exposure concentration: share of mentions captured by top-K items",
        ylabel="Share of total mentions (%)",
        out_path=out_dir / "exposure_topK_shares.png",
    )

    plot_grouped_bars(
        categories=["top1% items", "top5% items", "top10% items"],
        series=[
            ("train", [100.0 * exp_train["top1pct_items_share"], 100.0 * exp_train["top5pct_items_share"], 100.0 * exp_train["top10pct_items_share"]]),
            ("test", [100.0 * exp_test["top1pct_items_share"], 100.0 * exp_test["top5pct_items_share"], 100.0 * exp_test["top10pct_items_share"]]),
            ("all", [100.0 * exp_all["top1pct_items_share"], 100.0 * exp_all["top5pct_items_share"], 100.0 * exp_all["top10pct_items_share"]]),
        ],
        title="Exposure concentration: share of mentions captured by top-% of items",
        ylabel="Share of total mentions (%)",
        out_path=out_dir / "exposure_topP_shares.png",
    )

    plot_lorenz(
        groups=[
            ("train (Gini={:.3f})".format(exp_train["gini"]), exp_train["freqs"], "--"),
            ("test (Gini={:.3f})".format(exp_test["gini"]), exp_test["freqs"], ":"),
            ("all (Gini={:.3f})".format(exp_all["gini"]), exp_all["freqs"], "-"),
        ],
        title="Exposure concentration (Lorenz curve over mention counts per item)",
        out_path=out_dir / "exposure_lorenz.png",
    )

    # ----------------------------
    # Build summary.json (richer) + summary.md (tables)
    # ----------------------------
    # Popularity mapping coverage
    def mapping_block(pop: Dict[str, Any]) -> Dict[str, Any]:
        m = pop["mapping"]
        total = m["total_redial_mentions"]
        mapped = m["total_mapped_mentions"]
        unmapped = m["total_unmapped_mentions"]
        rate = (mapped / total) if total > 0 else None
        return {
            "total_redial_mentions": total,
            "mapped_mentions": mapped,
            "unmapped_mentions": unmapped,
            "mapping_rate": rate,
        }

    summary: Dict[str, Any] = {
        "note": {
            "popularity_definition": (
                "Popularity bias is measured on ReDial recommendations: we extract movies mentioned in "
                "Recommender turns, map them to MovieLens items, then use MovieLens rating-count as a popularity proxy. "
                "We report percentiles (0..1) under two baselines: (i) full MovieLens catalog, and "
                "(ii) restricted to items that appear in ReDial (mapped support)."
            ),
            "redundancy_definition": (
                "Redundancy measures repeated recommendations within the same dialogue (same MovieLens item appearing multiple times)."
            ),
        },
        "popularity": {
            "baseline_thresholds": {
                "movieLens_catalog_thr90_count": baselines["thr90_all"],
                "movieLens_catalog_thr95_count": baselines["thr95_all"],
                "redial_support_thr90_count": baselines["thr90_mapped"],
                "redial_support_thr95_count": baselines["thr95_mapped"],
            },
            "train": {
                "conversations_cached": pop_train["cached_conversations"],
                "conversations_with_mapped_recs": pop_train["conversations_with_mapped_recs"],
                "mapping": mapping_block(pop_train),
                "conversation_level_mean_percentile_all": stats(pop_train["conv_mean_percentile_all"]),
                "turn_level_mean_percentile": stats(pop_train["turn_mean_percentile"]),
                "item_level_percentile_catalog": stats(pop_train["item_percentile_global"]),
                "item_level_percentile_redial_support": stats(item_pct_r_train),
                "head_share_catalog": head_train_global,
                "head_share_redial_support": head_train_restricted,
            },
            "test": {
                "conversations_cached": pop_test["cached_conversations"],
                "conversations_with_mapped_recs": pop_test["conversations_with_mapped_recs"],
                "mapping": mapping_block(pop_test),
                "conversation_level_mean_percentile_all": stats(pop_test["conv_mean_percentile_all"]),
                "turn_level_mean_percentile": stats(pop_test["turn_mean_percentile"]),
                "item_level_percentile_catalog": stats(pop_test["item_percentile_global"]),
                "item_level_percentile_redial_support": stats(item_pct_r_test),
                "head_share_catalog": head_test_global,
                "head_share_redial_support": head_test_restricted,
            },
            "overall": {
                "mapping": mapping_block({
                    "mapping": {
                        "total_redial_mentions": pop_train["mapping"]["total_redial_mentions"] + pop_test["mapping"]["total_redial_mentions"],
                        "total_mapped_mentions": pop_train["mapping"]["total_mapped_mentions"] + pop_test["mapping"]["total_mapped_mentions"],
                        "total_unmapped_mentions": pop_train["mapping"]["total_unmapped_mentions"] + pop_test["mapping"]["total_unmapped_mentions"],
                    }
                }),
                "item_level_percentile_catalog": stats(pop_train["item_percentile_global"] + pop_test["item_percentile_global"]),
                "item_level_percentile_redial_support": stats(item_pct_r_train + item_pct_r_test),
                "head_share_catalog": head_all_global,
                "head_share_redial_support": head_all_restricted,
                "gini_over_rating_counts_of_recommended_items": gini(pop_train["item_count"] + pop_test["item_count"]),
            },
        },
        "redundancy": {
            "train": {
                "conversations_cached": red_train["cached_conversations"],
                "conversations_with_items": red_train["conversations_with_items"],
                "conversations_with_repeats": red_train["conversations_with_repeats"],
                "share_with_repeats": repeat_share(red_train),
                "redundancy_rate": stats(red_train["redundancy_rate"]),
                "repeated_items_per_turn": stats([float(x) for x in red_train["repeated_items_per_turn"]]),
                "total_repeat_events": red_train["total_repeat_events"],
                "unique_repeated_items": red_train["unique_repeated_items"],
            },
            "test": {
                "conversations_cached": red_test["cached_conversations"],
                "conversations_with_items": red_test["conversations_with_items"],
                "conversations_with_repeats": red_test["conversations_with_repeats"],
                "share_with_repeats": repeat_share(red_test),
                "redundancy_rate": stats(red_test["redundancy_rate"]),
                "repeated_items_per_turn": stats([float(x) for x in red_test["repeated_items_per_turn"]]),
                "total_repeat_events": red_test["total_repeat_events"],
                "unique_repeated_items": red_test["unique_repeated_items"],
            },
            "overall": {
                "top_repeated_items": [
                    {"movieId": mid, "title": titles.get(mid, str(mid)), "repeat_events": int(cnt)}
                    for mid, cnt in top_repeated
                ],
            },
        },
        "genre": {
            "train": {
                "conversations_cached": gen_train["cached_conversations"],
                "conversations_with_items": gen_train["conversations_with_items"],
                "js_divergence_vs_catalog": stats(gen_train["js_divergence_vs_catalog"]),
                "turn_jsd": stats(gen_train["turn_jsd"]),
            },
            "test": {
                "conversations_cached": gen_test["cached_conversations"],
                "conversations_with_items": gen_test["conversations_with_items"],
                "js_divergence_vs_catalog": stats(gen_test["js_divergence_vs_catalog"]),
                "turn_jsd": stats(gen_test["turn_jsd"]),
            },
            "overall": {
                "recommendation_genre_dist": rec_genre_overall,
                "top_genre_shifts_abs": [{"genre": g, "delta": float(genre_delta.get(g, 0.0))} for g, _ in top_delta],
            },
        },
        "exposure_concentration": {
            "train": {k: exp_train[k] for k in exp_train if k != "freqs"},
            "test": {k: exp_test[k] for k in exp_test if k != "freqs"},
            "overall": {k: exp_all[k] for k in exp_all if k != "freqs"},
        },
    }

    write_json(out_dir / "summary.json", summary)

    # ----------------------------
    # summary.md tables
    # ----------------------------
    md = []
    md.append("# Bias summaries (from cached results)\n")
    md.append("## Interpretation notes\n")
    md.append("- **Popularity bias** here means: *how popular are the items recommended in ReDial*, using **MovieLens rating-count** as a popularity proxy.\n")
    md.append("- Percentiles are reported as **0..100** in the tables below for readability.\n")
    md.append("- We report two popularity baselines:\n")
    md.append("  - **Catalog baseline**: percentile among *all* MovieLens items.\n")
    md.append("  - **ReDial-support baseline**: percentile among only items that appear in ReDial (mapped set).\n")

    # Popularity: mapping table
    pop_map_rows = []
    for split_name, pop in [("train", pop_train), ("test", pop_test)]:
        m = pop["mapping"]
        total = m["total_redial_mentions"]
        mapped = m["total_mapped_mentions"]
        rate = (mapped / total) if total > 0 else 0.0
        pop_map_rows.append([
            split_name,
            pop["cached_conversations"],
            pop["conversations_with_mapped_recs"],
            total,
            mapped,
            fmt(rate, nd=2, pct=True) + "%",
        ])
    md.append("\n## Popularity bias: mapping coverage\n")
    md.append(md_table(
        ["Split", "Cached convs", "Convs w/ mapped recs", "Total @mentions", "Mapped mentions", "Mapping rate"],
        pop_map_rows,
    ))

    # Popularity: item-level percentiles (catalog vs restricted)
    def pop_stat_row(split_name: str, vals: List[float]) -> List[Any]:
        s = stats(vals)
        return [split_name, s["n"], fmt(s["mean"], pct=True), fmt(s["median"], pct=True), fmt(s["p25"], pct=True), fmt(s["p75"], pct=True)]

    md.append("\n## Popularity bias: item-level popularity percentiles\n")
    md.append("### Catalog baseline (all MovieLens items)\n")
    md.append(md_table(
        ["Split", "N items", "Mean %ile", "Median %ile", "P25 %ile", "P75 %ile"],
        [
            pop_stat_row("train", pop_train["item_percentile_global"]),
            pop_stat_row("test", pop_test["item_percentile_global"]),
            pop_stat_row("all", pop_train["item_percentile_global"] + pop_test["item_percentile_global"]),
        ],
    ))

    md.append("\n### ReDial-support baseline (only items appearing in ReDial)\n")
    md.append(md_table(
        ["Split", "N items", "Mean %ile", "Median %ile", "P25 %ile", "P75 %ile"],
        [
            pop_stat_row("train", item_pct_r_train),
            pop_stat_row("test", item_pct_r_test),
            pop_stat_row("all", item_pct_r_train + item_pct_r_test),
        ],
    ))

    # Popularity: head share table
    md.append("\n## Popularity bias: head-share of recommended items\n")
    md.append(md_table(
        ["Split", "Top10% share (catalog)", "Top5% share (catalog)", "Top10% share (ReDial-support)", "Top5% share (ReDial-support)"],
        [
            ["train", fmt(head_train_global["top10pct"], pct=True) + "%", fmt(head_train_global["top5pct"], pct=True) + "%",
             fmt(head_train_restricted["top10pct"], pct=True) + "%", fmt(head_train_restricted["top5pct"], pct=True) + "%"],
            ["test", fmt(head_test_global["top10pct"], pct=True) + "%", fmt(head_test_global["top5pct"], pct=True) + "%",
             fmt(head_test_restricted["top10pct"], pct=True) + "%", fmt(head_test_restricted["top5pct"], pct=True) + "%"],
            ["all", fmt(head_all_global["top10pct"], pct=True) + "%", fmt(head_all_global["top5pct"], pct=True) + "%",
             fmt(head_all_restricted["top10pct"], pct=True) + "%", fmt(head_all_restricted["top5pct"], pct=True) + "%"],
        ],
    ))
    md.append("\n**Gini over rating-counts of recommended items (all splits):** {:.3f}\n".format(summary["popularity"]["overall"]["gini_over_rating_counts_of_recommended_items"]))

    # Redundancy tables
    md.append("\n## Redundancy bias (repeats within a dialogue)\n")
    red_rows = []
    for split_name, red in [("train", red_train), ("test", red_test)]:
        denom = red["conversations_with_items"]
        repeats = red["conversations_with_repeats"]
        share = (repeats / denom) if denom > 0 else 0.0
        red_rows.append([
            split_name,
            red["cached_conversations"],
            denom,
            repeats,
            fmt(share, nd=2, pct=True) + "%",
            red["total_repeat_events"],
            red["unique_repeated_items"],
        ])
    md.append(md_table(
        ["Split", "Cached convs", "Convs w/ items", "Convs w/ repeats", "% w/ repeats", "Total repeat events", "Unique repeated items"],
        red_rows,
    ))

    md.append("\n### Top repeated items (overall)\n")
    top_rows = []
    for mid, cnt in top_repeated:
        top_rows.append([mid, titles.get(mid, str(mid)), cnt])
    if top_rows:
        md.append(md_table(["movieId", "title", "repeat events"], top_rows[:15]))
    else:
        md.append("_No repeated items found in cached redundancy files._\n")

    # Genre tables
    md.append("\n## Genre bias (distribution shift vs MovieLens catalog)\n")
    def jsd_row(split_name: str, vals: List[float], cached_convs: int, convs_with_items: int) -> List[Any]:
        s = stats(vals)
        return [split_name, cached_convs, convs_with_items, s["n"], fmt(s["mean"]), fmt(s["median"]), fmt(s["p25"]), fmt(s["p75"])]

    md.append(md_table(
        ["Split", "Cached convs", "Convs w/ items", "N", "Mean JSD", "Median JSD", "P25", "P75"],
        [
            jsd_row("train", gen_train["js_divergence_vs_catalog"], gen_train["cached_conversations"], gen_train["conversations_with_items"]),
            jsd_row("test", gen_test["js_divergence_vs_catalog"], gen_test["cached_conversations"], gen_test["conversations_with_items"]),
            jsd_row("all", gen_train["js_divergence_vs_catalog"] + gen_test["js_divergence_vs_catalog"],
                    gen_train["cached_conversations"] + gen_test["cached_conversations"],
                    gen_train["conversations_with_items"] + gen_test["conversations_with_items"]),
        ],
    ))

    md.append("\n### Largest genre share shifts |ReDial - MovieLens|\n")
    delta_rows = []
    for g, d in top_delta[:15]:
        delta_rows.append([g, fmt(genre_baseline.get(g, 0.0), nd=4), fmt(rec_genre_overall.get(g, 0.0), nd=4), fmt(d, nd=4)])
    md.append(md_table(["Genre", "MovieLens share", "ReDial share", "Delta"], delta_rows))

    # Exposure concentration tables (corpus-level)
    md.append("\n## Exposure concentration (corpus-level)\n")
    def exp_row(name: str, exp: Dict[str, Any]) -> List[Any]:
        return [
            name,
            exp["unique_items"],
            exp["total_mentions"],
            fmt(exp["top10_share"], pct=True) + "%",
            fmt(exp["top50_share"], pct=True) + "%",
            fmt(exp["top100_share"], pct=True) + "%",
            fmt(exp["top5pct_items_share"], pct=True) + "%",
            fmt(exp["top10pct_items_share"], pct=True) + "%",
            fmt(exp["gini"]),
            fmt(exp["hhi"]),
        ]

    md.append(md_table(
        ["Split", "Unique items", "Total mentions", "Top10 items", "Top50 items", "Top100 items", "Top5% items", "Top10% items", "Gini", "HHI"],
        [
            exp_row("train", exp_train),
            exp_row("test", exp_test),
            exp_row("all", exp_all),
        ],
    ))

    write_text(out_dir / "summary.md", "\n".join(md))


if __name__ == "__main__":
    main()
