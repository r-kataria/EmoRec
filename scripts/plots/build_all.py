from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ._common import ensure_dir


def build_all_plots(
    *,
    redial_df: pd.DataFrame,
    cosrec_df: pd.DataFrame,
    redial_metrics: List[str],
    cosrec_metrics: List[str],
    redial_genre_stats: Dict[str, Any],
    cosrec_genre_stats: Dict[str, Any],
    redial_decade_stats: Dict[str, Any],
    redial_repeat_counts: Dict[str, Counter],
    cosrec_repeat_counts: Dict[str, Counter],
    redial_redundancy_progress: pd.DataFrame,
    cosrec_redundancy_progress: pd.DataFrame,
    redial_exposure_items: Dict[str, Any],
    cosrec_exposure_items: Dict[str, Any],
    redial_exposure_emotions: Dict[str, Any],
    cosrec_exposure_emotions: Dict[str, Any],
    family_dirs: Dict[str, Path],
) -> None:
    from .alluvial_emotion_to_category import plot as alluvial_emotion_to_category
    from .bias_rate_by_turn import plot as bias_rate_by_turn
    from .bias_vs_stereotype import plot as bias_vs_stereotype
    from .decade_stacked_bars import plot as decade_stacked_bars
    from .density_by_emotion import plot as density_by_emotion
    from .effect_size_heatmap import plot as effect_size_heatmap
    from .emotion_cooccurrence import plot as emotion_cooccurrence
    from .emotion_distribution import plot as emotion_distribution
    from .emotion_top1_vs_scores import plot as emotion_top1_vs_scores
    from .exposure_emotion_curves import plot as exposure_emotion_curves
    from .exposure_topk_share_by_emotion import plot as exposure_topk_share_by_emotion
    from .genre_lift_heatmap import plot as genre_lift_heatmap
    from .metric_scatter_by_emotion import plot as metric_scatter_by_emotion
    from .parallel_coordinates import plot as parallel_coordinates
    from .pareto_repeated_items import plot as pareto_repeated_items
    from .radar_per_emotion import plot as radar_per_emotion
    from .repeat_rate_bar import plot as repeat_rate_bar
    from .score_vs_metric import plot as score_vs_metric
    from .simple_violin_from_pairs import plot as simple_violin_from_pairs
    from .stereotype_by_emotion import plot as stereotype_by_emotion
    from .stereotype_score_by_emotion import plot as stereotype_score_by_emotion
    from .threshold_profile_by_emotion import plot as threshold_profile_by_emotion
    from .topk_uplift_dots import plot as topk_uplift_dots
    from .turn_trajectory_by_emotion import plot as turn_trajectory_by_emotion
    from .violin_by_emotion import plot as violin_by_emotion
    from .year_ridgeline import plot as year_ridgeline

    # Emotion signal graphs.
    emotion_distribution(redial_df, family_dirs["emotion"] / "emotion_distribution_redial.png", "ReDial: Emotion distribution")
    emotion_distribution(cosrec_df, family_dirs["emotion"] / "emotion_distribution_cosrec.png", "CoSRec: Emotion distribution")
    emotion_top1_vs_scores(redial_df, family_dirs["emotion"] / "emotion_top1_vs_scores_redial.png", "ReDial: Top-1 vs score distributions")
    emotion_top1_vs_scores(cosrec_df, family_dirs["emotion"] / "emotion_top1_vs_scores_cosrec.png", "CoSRec: Top-1 vs score distributions")
    emotion_cooccurrence(redial_df, family_dirs["emotion"] / "emotion_cooccurrence_redial.png", "ReDial: Emotion co-occurrence (top-3)")
    emotion_cooccurrence(cosrec_df, family_dirs["emotion"] / "emotion_cooccurrence_cosrec.png", "CoSRec: Emotion co-occurrence (top-3)")

    # Popularity bias graphs.
    violin_by_emotion(redial_df, "pop_mean_pct", family_dirs["popularity"] / "pop_mean_pct_by_emotion_redial.png", "ReDial: Mean popularity percentile by emotion")
    violin_by_emotion(cosrec_df, "pop_mean_pct", family_dirs["popularity"] / "pop_mean_pct_by_emotion_cosrec.png", "CoSRec: Mean popularity percentile by emotion")
    violin_by_emotion(redial_df, "pi", family_dirs["popularity"] / "pi_by_emotion_redial.png", "ReDial: PI by emotion")
    violin_by_emotion(cosrec_df, "pi", family_dirs["popularity"] / "pi_by_emotion_cosrec.png", "CoSRec: PI by emotion")
    threshold_profile_by_emotion(redial_df, family_dirs["popularity"] / "threshold_profile_redial.png", "ReDial: P@1/5/10 profile by emotion")
    threshold_profile_by_emotion(cosrec_df, family_dirs["popularity"] / "threshold_profile_cosrec.png", "CoSRec: P@1/5/10 profile by emotion")

    redial_emo_labels = sorted([c[len("emo_") :] for c in redial_df.columns if c.startswith("emo_")])
    cosrec_emo_labels = sorted([c[len("emo_") :] for c in cosrec_df.columns if c.startswith("emo_")])
    pop_score_dir_redial = family_dirs["popularity"] / "score_vs_pop_redial"
    pop_score_dir_cosrec = family_dirs["popularity"] / "score_vs_pop_cosrec"
    pi_score_dir_redial = family_dirs["popularity"] / "score_vs_pi_redial"
    pi_score_dir_cosrec = family_dirs["popularity"] / "score_vs_pi_cosrec"
    ensure_dir(pop_score_dir_redial)
    ensure_dir(pop_score_dir_cosrec)
    ensure_dir(pi_score_dir_redial)
    ensure_dir(pi_score_dir_cosrec)
    for emo in redial_emo_labels:
        score_vs_metric(redial_df, emo, "pop_mean_pct", pop_score_dir_redial / f"{emo}.png", f"ReDial: {emo} score vs popularity")
        score_vs_metric(redial_df, emo, "pi", pi_score_dir_redial / f"{emo}.png", f"ReDial: {emo} score vs PI")
    for emo in cosrec_emo_labels:
        score_vs_metric(cosrec_df, emo, "pop_mean_pct", pop_score_dir_cosrec / f"{emo}.png", f"CoSRec: {emo} score vs popularity")
        score_vs_metric(cosrec_df, emo, "pi", pi_score_dir_cosrec / f"{emo}.png", f"CoSRec: {emo} score vs PI")

    # Episode popularity dynamics.
    violin_by_emotion(redial_df, "cep", family_dirs["episode_popularity"] / "cep_by_emotion_redial.png", "ReDial: CEP by emotion")
    violin_by_emotion(cosrec_df, "cep", family_dirs["episode_popularity"] / "cep_by_emotion_cosrec.png", "CoSRec: CEP by emotion")
    violin_by_emotion(redial_df, "uiop", family_dirs["episode_popularity"] / "uiop_by_emotion_redial.png", "ReDial: UIOP by emotion")
    violin_by_emotion(cosrec_df, "uiop", family_dirs["episode_popularity"] / "uiop_by_emotion_cosrec.png", "CoSRec: UIOP by emotion")
    density_by_emotion(redial_df, "cep", "uiop", family_dirs["episode_popularity"] / "cep_uiop_density_redial", "ReDial: CEP vs UIOP")
    density_by_emotion(cosrec_df, "cep", "uiop", family_dirs["episode_popularity"] / "cep_uiop_density_cosrec", "CoSRec: CEP vs UIOP")
    turn_trajectory_by_emotion(redial_df, "pop_mean_pct", "rec_turn_order", family_dirs["episode_popularity"] / "popularity_trajectory_redial.png", "ReDial: Popularity trajectory by emotion")
    turn_trajectory_by_emotion(cosrec_df, "pop_mean_pct", "rec_turn_order", family_dirs["episode_popularity"] / "popularity_trajectory_cosrec.png", "CoSRec: Popularity trajectory by emotion")

    # Genre/category bias graphs.
    metric_scatter_by_emotion(redial_df, "genre_entropy", "genre_js", family_dirs["genre"] / "entropy_vs_js_redial.png", "ReDial: Genre entropy vs JS divergence")
    metric_scatter_by_emotion(cosrec_df, "genre_entropy", "genre_js", family_dirs["genre"] / "entropy_vs_js_cosrec.png", "CoSRec: Category entropy vs JS divergence")
    simple_violin_from_pairs(redial_genre_stats["top_shares"], family_dirs["genre"] / "top_genre_share_redial.png", "ReDial: Top genre share by emotion", "Top genre share")
    simple_violin_from_pairs(cosrec_genre_stats["top_shares"], family_dirs["genre"] / "top_genre_share_cosrec.png", "CoSRec: Top category share by emotion", "Top category share")
    simple_violin_from_pairs(redial_genre_stats["effective_nums"], family_dirs["genre"] / "effective_genres_redial.png", "ReDial: Effective number of genres by emotion", "exp(entropy)")
    simple_violin_from_pairs(cosrec_genre_stats["effective_nums"], family_dirs["genre"] / "effective_genres_cosrec.png", "CoSRec: Effective number of categories by emotion", "exp(entropy)")
    genre_lift_heatmap(redial_genre_stats, family_dirs["genre"] / "genre_lift_heatmap_redial.png", "ReDial: Genre lift by emotion")
    genre_lift_heatmap(cosrec_genre_stats, family_dirs["genre"] / "genre_lift_heatmap_cosrec.png", "CoSRec: Category lift by emotion")
    topk_uplift_dots(redial_genre_stats, family_dirs["genre"] / "top_uplift_redial", "ReDial")
    topk_uplift_dots(cosrec_genre_stats, family_dirs["genre"] / "top_uplift_cosrec", "CoSRec")
    alluvial_emotion_to_category(redial_genre_stats, family_dirs["genre"] / "emotion_to_genre_sankey_redial.png", "ReDial: Emotion → dominant genre")
    alluvial_emotion_to_category(cosrec_genre_stats, family_dirs["genre"] / "emotion_to_genre_sankey_cosrec.png", "CoSRec: Emotion → dominant category")

    # Year/decade bias (ReDial).
    year_ridgeline(redial_df, family_dirs["year_decade"] / "mean_year_ridgeline_redial.png", "ReDial: Mean release year by emotion")
    decade_stacked_bars(redial_decade_stats, family_dirs["year_decade"] / "decade_distribution_redial.png", "ReDial: Decade distribution by emotion")
    metric_scatter_by_emotion(redial_df, "pop_mean_pct", "mean_year", family_dirs["year_decade"] / "year_vs_popularity_redial.png", "ReDial: Year skew vs popularity")

    # Redundancy bias.
    violin_by_emotion(redial_df, "redundancy_repeated", family_dirs["redundancy"] / "repeat_count_redial.png", "ReDial: Repeated items by emotion")
    violin_by_emotion(cosrec_df, "redundancy_repeated", family_dirs["redundancy"] / "repeat_count_cosrec.png", "CoSRec: Repeated items by emotion")
    repeat_rate_bar(redial_df, family_dirs["redundancy"] / "repeat_rate_redial.png", "ReDial: P(repeated > 0) by emotion")
    repeat_rate_bar(cosrec_df, family_dirs["redundancy"] / "repeat_rate_cosrec.png", "CoSRec: P(repeated > 0) by emotion")
    turn_trajectory_by_emotion(redial_redundancy_progress, "unique_items_so_far", "rec_turn_order", family_dirs["redundancy"] / "unique_items_trajectory_redial.png", "ReDial: Cumulative unique items by emotion")
    turn_trajectory_by_emotion(cosrec_redundancy_progress, "unique_items_so_far", "rec_turn_order", family_dirs["redundancy"] / "unique_items_trajectory_cosrec.png", "CoSRec: Cumulative unique items by emotion")
    pareto_repeated_items(redial_repeat_counts, family_dirs["redundancy"] / "repeat_pareto_redial.png", "ReDial: Repeat-item Pareto by emotion")
    pareto_repeated_items(cosrec_repeat_counts, family_dirs["redundancy"] / "repeat_pareto_cosrec.png", "CoSRec: Repeat-item Pareto by emotion")

    # Exposure concentration.
    exposure_emotion_curves(
        redial_exposure_items["counts"],
        redial_exposure_emotions["counts"],
        redial_exposure_items["percentiles"],
        redial_exposure_emotions["episodes"],
        family_dirs["exposure"] / "exposure_gini_redial.png",
        "ReDial: Exposure vs popularity (Gini)",
    )
    exposure_emotion_curves(
        cosrec_exposure_items["counts"],
        cosrec_exposure_emotions["counts"],
        cosrec_exposure_items["percentiles"],
        cosrec_exposure_emotions["episodes"],
        family_dirs["exposure"] / "exposure_gini_cosrec.png",
        "CoSRec: Exposure vs popularity (Gini)",
    )
    exposure_topk_share_by_emotion(redial_exposure_emotions["counts"], redial_exposure_emotions["episodes"], family_dirs["exposure"] / "exposure_topk_share_redial.png", "ReDial: Exposure top-K share by emotion")
    exposure_topk_share_by_emotion(cosrec_exposure_emotions["counts"], cosrec_exposure_emotions["episodes"], family_dirs["exposure"] / "exposure_topk_share_cosrec.png", "CoSRec: Exposure top-K share by emotion")

    # Stereotype/biasful language.
    stereotype_by_emotion(redial_df, family_dirs["stereotype"] / "stereotype_by_emotion_redial.png", "ReDial: Stereotype by emotion (top-1)")
    stereotype_by_emotion(cosrec_df, family_dirs["stereotype"] / "stereotype_by_emotion_cosrec.png", "CoSRec: Stereotype by emotion (top-1)")
    bias_rate_by_turn(redial_df, family_dirs["stereotype"] / "bias_rate_by_turn_redial.png", "ReDial: Bias rate by conversation position", turn_col="rec_turn_order")
    bias_rate_by_turn(cosrec_df, family_dirs["stereotype"] / "bias_rate_by_turn_cosrec.png", "CoSRec: Bias rate by conversation position", turn_col="rec_turn_order")
    bias_vs_stereotype(redial_df, ["pop_mean_pct", "genre_js", "redundancy_repeated"], family_dirs["stereotype"] / "bias_vs_stereotype_redial.png", "ReDial: Bias metrics vs stereotype label")
    bias_vs_stereotype(cosrec_df, ["pop_mean_pct", "genre_js", "redundancy_repeated"], family_dirs["stereotype"] / "bias_vs_stereotype_cosrec.png", "CoSRec: Bias metrics vs stereotype label")
    stereotype_score_by_emotion(redial_df, family_dirs["stereotype"] / "stereotype_score_by_emotion_redial.png", "ReDial: LABEL_1 score by emotion")
    stereotype_score_by_emotion(cosrec_df, family_dirs["stereotype"] / "stereotype_score_by_emotion_cosrec.png", "CoSRec: LABEL_1 score by emotion")

    # Cross-bias summary visuals.
    effect_size_heatmap(redial_df, redial_metrics, family_dirs["summary"] / "effect_size_heatmap_redial.png", "ReDial: Emotion × metric effect sizes")
    effect_size_heatmap(cosrec_df, cosrec_metrics, family_dirs["summary"] / "effect_size_heatmap_cosrec.png", "CoSRec: Emotion × metric effect sizes")
    parallel_coordinates(redial_df, redial_metrics, family_dirs["summary"] / "parallel_coordinates_redial.png", "ReDial: Parallel coordinates (bias profiles)")
    parallel_coordinates(cosrec_df, cosrec_metrics, family_dirs["summary"] / "parallel_coordinates_cosrec.png", "CoSRec: Parallel coordinates (bias profiles)")
    radar_per_emotion(redial_df, ["pop_mean_pct", "p05", "cep", "uiop", "genre_js", "year_decade_js", "redundancy_repeated"], family_dirs["summary"] / "radar_redial", "ReDial radar")
    radar_per_emotion(cosrec_df, ["pop_mean_pct", "p05", "cep", "uiop", "genre_js", "redundancy_repeated"], family_dirs["summary"] / "radar_cosrec", "CoSRec radar")

