#!/usr/bin/env python3
"""
Co-recommendation community analysis (ReDial, using your cached MovieLens-mapped rec turns).

Reads:
  ./cache/bias/popularity/redial/ml-25m/{train|test}/*.json

Builds a co-recommendation graph:
  - nodes = MovieLens movieId (only items that appear in your cached popularity files)
  - edge weight = #times two items co-occur in the SAME recommender turn (episode)

Runs community detection (default: greedy modularity) and outputs:
  - JSON: node->community mapping + community stats
  - Markdown summary with tables
  - Plots (ACM-ish grayscale):
      - top community exposure shares
      - Lorenz curve (community exposure concentration)
      - size_share vs exposure_share scatter
      - community size histogram
  - JSONL with per-episode community mixture (for later emotion↔bias correlation)

Usage:
  python scripts/visualize_corec.py --cache_root ./cache --out_dir ./cache/viz_corec

Notes:
  - This uses NO new datasets beyond what you already cached.
  - You must have run your popularity bias caching first (so the JSON files exist).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


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


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def fmt_pct(x: float, nd: int = 2) -> str:
    return f"{100.0 * float(x):.{nd}f}%"


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


def plot_lorenz(exposure: List[int], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs, ys = lorenz_curve(exposure)
    plt.figure()
    plt.plot(xs, ys, color="black")
    plt.plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=1.0)
    plt.title(title)
    plt.xlabel("Cumulative share of communities")
    plt.ylabel("Cumulative share of exposure")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar_top(values: List[Tuple[str, float]], out_path: Path, title: str, xlabel: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        return
    labels = [k for k, _ in values]
    vals = [v for _, v in values]
    plt.figure(figsize=(max(8, 0.5 * len(labels)), 4.5))
    plt.bar(range(len(labels)), vals, color="0.3")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(x: List[float], y: List[float], out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not x or not y or len(x) != len(y):
        return
    plt.figure()
    plt.scatter(x, y, s=10, color="0.3")
    # diagonal
    lo = min(min(x), min(y))
    hi = max(max(x), max(y))
    plt.plot([lo, hi], [lo, hi], linestyle=":", color="black", linewidth=1.0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_hist(vals: List[int], out_path: Path, title: str, xlabel: str, bins: int = 30) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not vals:
        return
    plt.figure()
    plt.hist(vals, bins=bins, color="0.3")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_movielens_titles(cache_root: Path) -> Dict[int, str]:
    """
    Optional: used only to make the tables nicer.
    Expects the extracted file:
      <cache_root>/datasets/movielens/ml-25m/ml-25m/movies.csv
    """
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


def iter_popularity_turns(cache_root: Path, split: str) -> Iterable[Tuple[str, int, List[int]]]:
    """
    Yields (conversationId, msg_idx, movielens_movie_ids) for recommender turns.
    Reads cached popularity outputs:
      <cache_root>/bias/popularity/redial/ml-25m/{split}/*.json
    """
    base = cache_root / "bias" / "popularity" / "redial" / "ml-25m" / split
    for fp in glob_jsons(base):
        rec = read_json(fp)
        cid = str(rec.get("conversationId", ""))
        for t in rec.get("turns", []) or []:
            if t.get("speaker") != "Recommender":
                continue
            ml_ids = t.get("movielens_movie_ids") or []
            if not ml_ids:
                continue
            try:
                msg_idx = int(t.get("msg_idx"))
            except Exception:
                continue
            out_ids = []
            for x in ml_ids:
                try:
                    out_ids.append(int(x))
                except Exception:
                    pass
            if out_ids:
                yield cid, msg_idx, out_ids


def build_corec_graph(cache_root: Path, splits: List[str], min_edge_weight: int = 2) -> Tuple[Counter, Dict[Tuple[int, int], int], List[Tuple[str, int, List[int]]]]:
    """
    Returns:
      - node_exposure: movieId -> mention count (by occurrence in recommender turns)
      - edge_weights: (u,v) -> co-occurrence count (u < v)
      - episodes: list of (cid, msg_idx, unique_sorted_movieIds) for later community mixture export
    """
    node_exposure = Counter()
    edge_weights: Dict[Tuple[int, int], int] = defaultdict(int)
    episodes: List[Tuple[str, int, List[int]]] = []

    for split in splits:
        for cid, msg_idx, ml_ids in iter_popularity_turns(cache_root, split):
            # node exposures: count each mention (including duplicates in the list, if any)
            for mid in ml_ids:
                node_exposure[mid] += 1

            uniq = sorted(set(ml_ids))
            if len(uniq) < 2:
                episodes.append((cid, msg_idx, uniq))
                continue

            # co-occurrence edges within THIS recommender turn
            for i in range(len(uniq)):
                u = uniq[i]
                for j in range(i + 1, len(uniq)):
                    v = uniq[j]
                    edge_weights[(u, v)] += 1

            episodes.append((cid, msg_idx, uniq))

    if min_edge_weight > 1:
        edge_weights = {k: w for k, w in edge_weights.items() if w >= min_edge_weight}

    return node_exposure, edge_weights, episodes


def detect_communities(node_exposure: Counter, edge_weights: Dict[Tuple[int, int], int]) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    Community detection on weighted undirected graph.
    Returns:
      - node2comm: movieId -> community_id
      - communities: list of node lists
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: networkx. Install with: pip install networkx"
        ) from e

    G = nx.Graph()
    for mid, exp in node_exposure.items():
        G.add_node(int(mid), exposure=int(exp))

    for (u, v), w in edge_weights.items():
        G.add_edge(int(u), int(v), weight=int(w))

    # Greedy modularity communities (deterministic given graph)
    comms = list(greedy_modularity_communities(G, weight="weight"))

    node2comm: Dict[int, int] = {}
    communities: List[List[int]] = []
    for cid, s in enumerate(comms):
        nodes = sorted(int(x) for x in s)
        communities.append(nodes)
        for n in nodes:
            node2comm[n] = cid

    # Any isolated nodes might not appear depending on algorithm; ensure coverage
    for n in G.nodes():
        if n not in node2comm:
            cid = len(communities)
            node2comm[n] = cid
            communities.append([int(n)])

    return node2comm, communities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, default="../cache")
    ap.add_argument("--out_dir", type=str, default="./cache/viz_corec")
    ap.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    ap.add_argument("--min_edge_weight", type=int, default=2)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--no_acm_style", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_acm_style:
        set_acm_style()

    splits = ["train", "test"] if args.split == "all" else [args.split]

    # Build graph from cached popularity files (MovieLens-mapped)
    node_exposure, edge_weights, episodes = build_corec_graph(
        cache_root=cache_root,
        splits=splits,
        min_edge_weight=int(args.min_edge_weight),
    )

    # Detect communities
    node2comm, communities = detect_communities(node_exposure, edge_weights)

    # Titles are optional (if MovieLens downloaded)
    titles = load_movielens_titles(cache_root)

    total_exposure = int(sum(node_exposure.values()))
    n_nodes = int(len(node_exposure))
    n_edges = int(len(edge_weights))

    # Community stats
    comm_exposure = Counter()
    comm_size = Counter()
    for mid, exp in node_exposure.items():
        c = node2comm.get(int(mid), -1)
        if c >= 0:
            comm_exposure[c] += int(exp)
            comm_size[c] += 1

    exposure_values = [int(comm_exposure[c]) for c in sorted(comm_exposure.keys())]
    size_values = [int(comm_size[c]) for c in sorted(comm_size.keys())]

    # Over-exposure ratio: exposure_share / size_share
    comm_stats = []
    for c in sorted(comm_exposure.keys()):
        exp = int(comm_exposure[c])
        sz = int(comm_size[c])
        exp_share = (exp / total_exposure) if total_exposure > 0 else 0.0
        sz_share = (sz / n_nodes) if n_nodes > 0 else 0.0
        ratio = (exp_share / sz_share) if sz_share > 0 else None

        # Top movies in community by exposure
        members = communities[c] if c < len(communities) else []
        top_movies = sorted(members, key=lambda m: node_exposure.get(m, 0), reverse=True)[:8]
        top_movies_disp = [titles.get(m, str(m)) for m in top_movies]

        comm_stats.append(
            {
                "community": c,
                "size": sz,
                "exposure": exp,
                "exposure_share": exp_share,
                "size_share": sz_share,
                "overexposure_ratio": ratio,
                "top_movies": top_movies_disp,
            }
        )

    # Save JSON artifacts
    out_json = {
        "meta": {
            "split": args.split,
            "min_edge_weight": int(args.min_edge_weight),
            "nodes": n_nodes,
            "edges": n_edges,
            "total_exposure_mentions": total_exposure,
            "num_communities": int(len(comm_stats)),
            "gini_community_exposure": gini(exposure_values),
            "hhi_community_exposure": hhi(exposure_values),
        },
        "node2community": {str(mid): int(cid) for mid, cid in node2comm.items()},
        "communities": comm_stats,
    }
    write_json(out_dir / "communities.json", out_json)

    # Episode-level community mixture for later emotion correlation
    # One line per recommender turn: dominant community + distribution
    jsonl_path = out_dir / "turn_communities.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for cid, msg_idx, mids in episodes:
            comms = [node2comm.get(int(m), -1) for m in mids]
            comms = [c for c in comms if c >= 0]
            if not comms:
                continue
            cc = Counter(comms)
            dominant, dom_cnt = cc.most_common(1)[0]
            total = sum(cc.values())
            dist = {str(k): (v / total) for k, v in cc.items()}
            f.write(
                json.dumps(
                    {
                        "conversationId": cid,
                        "msg_idx": int(msg_idx),
                        "movieLens_movieIds": mids,
                        "communities": [int(c) for c in comms],
                        "dominant_community": int(dominant),
                        "dominant_share": float(dom_cnt / total),
                        "community_dist": dist,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Markdown summary (tables)
    by_exposure = sorted(comm_stats, key=lambda d: d["exposure_share"], reverse=True)[: args.top_k]
    by_ratio = sorted([d for d in comm_stats if d["overexposure_ratio"] is not None], key=lambda d: d["overexposure_ratio"], reverse=True)[: args.top_k]

    md = []
    md.append("# Co-recommendation community bias (ReDial)\n")
    md.append("**Graph definition**: nodes are MovieLens movieIds; edges connect items co-mentioned in the same Recommender turn; edge weight = co-occurrence count.\n")
    md.append(f"- Split: `{args.split}`\n- Min edge weight: `{int(args.min_edge_weight)}`\n")
    md.append("## Summary\n")
    md.append(md_table(
        ["Metric", "Value"],
        [
            ["Nodes (items)", n_nodes],
            ["Edges (co-occurrence)", n_edges],
            ["Total recommendation mentions", total_exposure],
            ["# Communities", len(comm_stats)],
            ["Gini (community exposure)", f"{gini(exposure_values):.3f}"],
            ["HHI (community exposure)", f"{hhi(exposure_values):.3f}"],
        ],
    ))

    md.append("\n## Top communities by exposure share\n")
    rows = []
    for d in by_exposure:
        rows.append([
            d["community"],
            d["size"],
            d["exposure"],
            fmt_pct(d["exposure_share"]),
            fmt_pct(d["size_share"]),
            f"{d['overexposure_ratio']:.2f}" if d["overexposure_ratio"] is not None else "—",
            "; ".join(d["top_movies"][:4]),
        ])
    md.append(md_table(
        ["Community", "Size", "Exposure", "Exposure share", "Size share", "Overexp. ratio", "Top movies (examples)"],
        rows
    ))

    md.append("\n## Most over-exposed communities (exposure_share / size_share)\n")
    rows = []
    for d in by_ratio:
        rows.append([
            d["community"],
            d["size"],
            d["exposure"],
            fmt_pct(d["exposure_share"]),
            fmt_pct(d["size_share"]),
            f"{d['overexposure_ratio']:.2f}",
            "; ".join(d["top_movies"][:4]),
        ])
    md.append(md_table(
        ["Community", "Size", "Exposure", "Exposure share", "Size share", "Overexp. ratio", "Top movies (examples)"],
        rows
    ))

    write_text(out_dir / "summary.md", "\n".join(md))

    # Plots
    # 1) top community exposure shares
    top_bar = [(f"C{d['community']}", float(d["exposure_share"])) for d in by_exposure]
    plot_bar_top(
        values=top_bar,
        out_path=out_dir / "top_community_exposure_share.png",
        title=f"Top communities by exposure share ({args.split})",
        xlabel="Exposure share",
    )

    # 2) Lorenz curve for community exposure
    plot_lorenz(
        exposure=exposure_values,
        out_path=out_dir / "community_exposure_lorenz.png",
        title="Community exposure concentration (Lorenz curve)",
    )

    # 3) scatter size_share vs exposure_share
    xs = [d["size_share"] for d in comm_stats]
    ys = [d["exposure_share"] for d in comm_stats]
    plot_scatter(
        x=xs,
        y=ys,
        out_path=out_dir / "size_share_vs_exposure_share.png",
        title="Community over/under-exposure",
        xlabel="Size share (#items in community / #items total)",
        ylabel="Exposure share (#mentions in community / #mentions total)",
    )

    # 4) community size histogram
    plot_hist(
        vals=[d["size"] for d in comm_stats],
        out_path=out_dir / "community_size_hist.png",
        title="Community sizes",
        xlabel="#items in community",
        bins=30,
    )


if __name__ == "__main__":
    main()
