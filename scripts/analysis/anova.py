from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def anova_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    if df.empty or "emotion_top1" not in df:
        return pd.DataFrame([])
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        if metric not in df:
            continue
        sub = df[["emotion_top1", metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna()
        if sub.empty:
            continue

        groups = []
        group_counts: Dict[str, int] = {}
        group_means: Dict[str, float] = {}
        for emo, grp in sub.groupby("emotion_top1"):
            vals = grp[metric].to_numpy()
            if len(vals) < 2:
                continue
            groups.append(vals)
            group_counts[str(emo)] = int(len(vals))
            group_means[str(emo)] = float(np.mean(vals))

        if len(groups) < 2:
            continue

        n_total = int(sum(group_counts.values()))
        overall_mean = (
            float(sum(group_means[emo] * n for emo, n in group_counts.items()) / n_total)
            if n_total > 0
            else float("nan")
        )
        ss_between = sum(n * (group_means[emo] - overall_mean) ** 2 for emo, n in group_counts.items())
        ss_within = float(
            sum(float(((vals - float(np.mean(vals))) ** 2).sum()) for vals in groups)
        )
        ss_total = float(ss_between + ss_within)
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else float("nan")

        df_between = max(1, len(groups) - 1)
        df_within = max(1, n_total - len(groups))
        ms_between = ss_between / df_between if df_between > 0 else float("nan")
        ms_within = ss_within / df_within if df_within > 0 else float("nan")
        f_stat = float(ms_between / ms_within) if ms_within > 0 else float("nan")

        p_val = float("nan")
        try:
            from scipy.stats import f as f_dist  # type: ignore

            p_val = float(f_dist.sf(f_stat, df_between, df_within))
        except Exception:
            pass

        rows.append(
            {
                "metric": metric,
                "n": n_total,
                "k": int(len(groups)),
                "f_stat": float(f_stat),
                "p_value": float(p_val),
                "eta_sq": eta_sq,
            }
        )
    return pd.DataFrame(rows)
