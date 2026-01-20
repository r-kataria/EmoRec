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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.anova import anova_table as _anova_table
from analysis.correlations import corr_table as _corr_table
from analysis.exposure import exposure_by_emotion_table as _exposure_by_emotion_table
from analysis.models import model_tables as _model_tables
from plots.build_all import build_all_plots


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


def _format_numeric_table(
    df: pd.DataFrame,
    p_cols: Optional[Iterable[str]] = None,
    decimals: int = 3,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    p_cols = set(p_cols or [])

    def is_int_like(series: pd.Series) -> bool:
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            return False
        return bool(np.all(np.isclose(vals, np.round(vals))))

    def fmt_val(x: Any) -> str:
        if pd.isna(x):
            return ""
        try:
            return f"{float(x):.{decimals}f}"
        except Exception:
            return str(x)

    def fmt_int(x: Any) -> str:
        if pd.isna(x):
            return ""
        try:
            return str(int(round(float(x))))
        except Exception:
            return str(x)

    def fmt_p(x: Any) -> str:
        if pd.isna(x):
            return ""
        try:
            val = float(x)
        except Exception:
            return str(x)
        if val < 0.001:
            return "<0.001"
        return f"{val:.{decimals}f}"

    for col in out.columns:
        series = out[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if col in p_cols:
            out[col] = series.map(fmt_p)
        elif is_int_like(series):
            out[col] = series.map(fmt_int)
        else:
            out[col] = series.map(fmt_val)
    return out


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


def _aggregate_dist_stats(pairs: List[Tuple[str, Dict[str, float]]]) -> Dict[str, Any]:
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


def _collect_redial_redundancy_stats(cache_root: Path, df: pd.DataFrame) -> Tuple[Dict[str, Counter], pd.DataFrame]:
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
                    counts_by_emotion[emotion][str(item)] += 1
                new_items = t.get("new_items") or []
                for item in new_items:
                    try:
                        seen.add(int(item))
                    except Exception:
                        continue
                rows.append({"emotion_top1": emotion, "rec_turn_order": rec_order, "unique_items_so_far": len(seen)})
    return counts_by_emotion, pd.DataFrame(rows)


def _collect_cosrec_redundancy_stats(cache_root: Path, df: pd.DataFrame) -> Tuple[Dict[str, Counter], pd.DataFrame]:
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
                rows.append({"emotion_top1": emo, "rec_turn_order": int(rec_order), "unique_items_so_far": int(unique_so_far)})
    return counts_by_emotion, pd.DataFrame(rows)


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
            lines.append("Stereotype label rates (%): " f"Non-biased={rate_0:.1f}, Biased={rate_1:.1f}")

    for metric in ["p01", "p05", "p10", "pi", "pop_mean_pct", "cep", "uiop"]:
        if metric in df:
            series = pd.to_numeric(df[metric], errors="coerce")
            nonzero = float((series.fillna(0.0) > 0).mean())
            lines.append(f"{metric} nonzero rate: {nonzero:.3f}")

    if exposure:
        lines.append(f"Exposure: gini={exposure.get('gini')}")

    lines.append("")
    lines.append("Top correlations (Spearman p<0.05):")
    top_corr = _summarize_correlations(corr, top_n=10)
    if top_corr.empty:
        lines.append("  none")
    else:
        for _, row in top_corr.iterrows():
            lines.append(
                f"  {row['metric']} vs {row['emotion']}: r={row['spearman_r']:.3f} "
                f"(p={row['spearman_p']:.3g}, n={int(row['n'])})"
            )

    lines.append("")
    lines.append("Model summary (5-fold CV):")
    if models.empty:
        lines.append("  none (missing scikit-learn)")
    else:
        for _, row in models.iterrows():
            if row["model"] == "ridge_regression":
                lines.append(f"  {row['metric']}: R^2={row['r2_mean']:.3f}, MAE={row['mae_mean']:.3f}, n={int(row['n'])}")
            else:
                lines.append(f"  {row['metric']}: AUC={row['roc_auc_mean']:.3f}, Acc={row['acc_mean']:.3f}, n={int(row['n'])}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_redial(cache_root: Path, splits: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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
            red = _load_json(red_dir / f"{sid}.json")
            stereo = _load_json(stereo_dir / f"{sid}.json") if stereo_dir else None

            emo_turns = {t["msg_idx"]: t for t in emo.get("turns", [])}
            pop_turns = {t["msg_idx"]: t for t in (pop or {}).get("turns", [])}
            epi_turns = {t["msg_idx"]: t for t in (epi or {}).get("turns", [])}
            gen_turns = {t["msg_idx"]: t for t in (gen or {}).get("turns", [])}
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

    exposure = (
        _load_json(cache_root / "bias" / "exposure_concentration" / "redial" / "ml-25m" / "train.json") or {}
    )
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
        pop = _load_json(bias_base / "popularity" / "cosrec" / "amazon_2023" / "curated" / f"{topic}.json")
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

        gen = _load_json(bias_base / "genre" / "cosrec" / "amazon_2023" / "curated" / f"{topic}.json")
        gen_obj = (gen or {}).get("summary") or {}

        red = _load_json(bias_base / "redundancy" / "cosrec" / "amazon_2023" / "curated" / f"{topic}.json")

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

    exposure = (
        _load_json(cache_root / "bias" / "exposure_concentration" / "cosrec" / "amazon_2023" / "curated.json") or {}
    )
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
    _ensure_dir(tables_dir)

    graphs_dir = ROOT.parent / "graphs"
    _ensure_dir(graphs_dir)

    redial_df, redial_exposure, redial_exposure_emotions, redial_exposure_items = _load_redial(cache_root, splits)
    cosrec_df, cosrec_exposure, cosrec_exposure_emotions, cosrec_exposure_items = _load_cosrec(cache_root)

    _add_turn_order(redial_df, "conversation_id", "turn_idx", "rec_turn_order")
    _add_turn_order(cosrec_df, "conversation_id", "system_turn_idx", "rec_turn_order")

    _format_numeric_table(redial_df).to_csv(tables_dir / "episodes_redial.csv", index=False)
    _format_numeric_table(cosrec_df).to_csv(tables_dir / "episodes_cosrec.csv", index=False)

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
        "genre_js",
        "genre_entropy",
        "redundancy_new",
        "redundancy_repeated",
    ]

    redial_corr = _corr_table(redial_df, redial_metrics)
    cosrec_corr = _corr_table(cosrec_df, cosrec_metrics)
    _format_numeric_table(redial_corr, p_cols=["pearson_p", "spearman_p"]).to_csv(
        tables_dir / "correlations_redial.csv", index=False
    )
    _format_numeric_table(cosrec_corr, p_cols=["pearson_p", "spearman_p"]).to_csv(
        tables_dir / "correlations_cosrec.csv", index=False
    )

    redial_anova = _anova_table(redial_df, redial_metrics)
    cosrec_anova = _anova_table(cosrec_df, cosrec_metrics)
    _format_numeric_table(redial_anova, p_cols=["p_value"]).to_csv(tables_dir / "anova_redial.csv", index=False)
    _format_numeric_table(cosrec_anova, p_cols=["p_value"]).to_csv(tables_dir / "anova_cosrec.csv", index=False)

    redial_models = _model_tables(redial_df, redial_metrics + ["stereotype_label"])
    cosrec_models = _model_tables(cosrec_df, cosrec_metrics + ["stereotype_label"])
    _format_numeric_table(redial_models).to_csv(tables_dir / "models_redial.csv", index=False)
    _format_numeric_table(cosrec_models).to_csv(tables_dir / "models_cosrec.csv", index=False)

    redial_exp_df = _exposure_by_emotion_table(redial_exposure_emotions["counts"], redial_exposure_emotions["episodes"])
    cosrec_exp_df = _exposure_by_emotion_table(cosrec_exposure_emotions["counts"], cosrec_exposure_emotions["episodes"])
    _format_numeric_table(redial_exp_df).to_csv(tables_dir / "exposure_by_emotion_redial.csv", index=False)
    _format_numeric_table(cosrec_exp_df).to_csv(tables_dir / "exposure_by_emotion_cosrec.csv", index=False)

    redial_genre_pairs = _collect_redial_turn_dists(cache_root, redial_df, "genre", "genre_bias", "genres_dist")
    cosrec_genre_pairs = _collect_cosrec_episode_dists(cache_root, cosrec_df, "genre", "categories_dist")
    redial_genre_stats = _aggregate_dist_stats(redial_genre_pairs)
    cosrec_genre_stats = _aggregate_dist_stats(cosrec_genre_pairs)

    _write_summary(tables_dir / "summary_redial.txt", "ReDial", redial_df, redial_corr, redial_models, redial_exposure)
    _write_summary(tables_dir / "summary_cosrec.txt", "CoSRec", cosrec_df, cosrec_corr, cosrec_models, cosrec_exposure)

    build_all_plots(
        redial_df=redial_df,
        cosrec_df=cosrec_df,
        redial_genre_stats=redial_genre_stats,
        cosrec_genre_stats=cosrec_genre_stats,
        redial_exposure_items=redial_exposure_items,
        cosrec_exposure_items=cosrec_exposure_items,
        redial_exposure_emotions=redial_exposure_emotions,
        cosrec_exposure_emotions=cosrec_exposure_emotions,
        out_dir=graphs_dir,
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
