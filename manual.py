#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import CoSRecDataset, ReDialDataset  # noqa: E402
from dataset.cosrec import safe_id as cosrec_safe_id  # noqa: E402
from dataset.redial import extract_movie_ids, get_speaker as redial_get_speaker  # noqa: E402


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_json_file(folder: Optional[Path]) -> Optional[Path]:
    if folder is None or not folder.exists():
        return None
    files = sorted(folder.glob("*.json"))
    return files[0] if files else None


def _find_top_dir(root: Path, preferred: Optional[str]) -> Optional[Path]:
    if preferred:
        p = root / preferred
        if p.exists():
            return p
    tops = sorted(root.glob("top*"))
    return tops[0] if tops else None


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        v = float(value)
    except Exception:
        return "na"
    return f"{v:.{digits}f}"


def _score_value(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _fmt_preds(preds: Any, max_k: int = 3) -> str:
    if not preds:
        return "none"
    items = []
    if isinstance(preds, dict):
        items = [(str(preds.get("label", "")), preds.get("score", None))]
    elif isinstance(preds, list):
        for d in preds:
            if isinstance(d, dict):
                items.append((str(d.get("label", "")), d.get("score", None)))
    items.sort(key=lambda x: float(x[1] or 0.0), reverse=True)
    parts = []
    for label, score in items[:max_k]:
        if score is None:
            parts.append(label)
        else:
            parts.append(f"{label} ({_fmt_float(score)})")
    return ", ".join(parts) if parts else "none"


def _fmt_popularity(obj: Optional[Dict[str, Any]]) -> str:
    if not obj:
        return "none"
    return (
        f"mean_pct={_fmt_float(obj.get('mean_percentile'))} "
        f"head_top10={_fmt_float(obj.get('head_share_top10pct'))} "
        f"mean_count={_fmt_float(obj.get('mean_count'), 1)}"
    )


def _fmt_rating(obj: Optional[Dict[str, Any]]) -> str:
    if not obj:
        return "none"
    return (
        f"mean_rating={_fmt_float(obj.get('mean_rating'))} "
        f"mean_pct={_fmt_float(obj.get('mean_percentile'))}"
    )


def _fmt_genre(obj: Optional[Dict[str, Any]], js_key: str) -> str:
    if not obj:
        return "none"
    if "genres_entropy" in obj:
        entropy = obj.get("genres_entropy")
    else:
        entropy = obj.get("categories_entropy")
    if "genre_coverage" in obj:
        coverage = obj.get("genre_coverage")
    else:
        coverage = obj.get("category_coverage")
    return (
        f"js={_fmt_float(obj.get(js_key))} "
        f"entropy={_fmt_float(entropy)} "
        f"coverage={coverage or 0}"
    )


def _fmt_year_decade(obj: Optional[Dict[str, Any]]) -> str:
    if not obj:
        return "none"
    return (
        f"mean_year={_fmt_float(obj.get('mean_year'), 1)} "
        f"js_year={_fmt_float(obj.get('years_js_divergence_vs_catalog'))} "
        f"js_decade={_fmt_float(obj.get('decades_js_divergence_vs_catalog'))}"
    )


def _fmt_redundancy(new_items: Iterable[Any], repeated_items: Iterable[Any]) -> str:
    new_n = len(list(new_items)) if new_items is not None else 0
    rep_n = len(list(repeated_items)) if repeated_items is not None else 0
    return f"new={new_n} repeated={rep_n}"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _label_score(preds: Any, label: str) -> Optional[float]:
    if preds is None:
        return None
    if isinstance(preds, dict):
        return float(preds.get("score", 0.0)) if str(preds.get("label")) == label else 0.0
    if isinstance(preds, list):
        for d in preds:
            if not isinstance(d, dict):
                continue
            if str(d.get("label")) == label:
                return float(d.get("score", 0.0))
        return 0.0
    return None


def _aggregate_emotions(preds_list: List[Any]) -> Optional[List[Dict[str, float]]]:
    valid = [p for p in preds_list if p]
    if not valid:
        return None
    totals: Dict[str, float] = {}
    for preds in valid:
        items = [preds] if isinstance(preds, dict) else preds
        if not isinstance(items, list):
            continue
        for d in items:
            if not isinstance(d, dict):
                continue
            label = str(d.get("label", ""))
            score = _score_value(d.get("score", 0.0))
            totals[label] = totals.get(label, 0.0) + score
    count = len(valid)
    averaged = [{"label": k, "score": v / count} for k, v in totals.items()]
    averaged.sort(key=lambda x: x["score"], reverse=True)
    return averaged


def _aggregate_stereotype(preds_list: List[Any]) -> Optional[Dict[str, float]]:
    valid = [p for p in preds_list if p is not None]
    if not valid:
        return None
    scores: List[float] = []
    hits = 0
    for preds in valid:
        score = _label_score(preds, "LABEL_1") or 0.0
        if score > 0:
            hits += 1
        scores.append(score)
    return {
        "label1_rate": hits / len(valid),
        "label1_avg_score": sum(scores) / len(scores),
    }


def _aggregate_popularity(pop_list: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not pop_list:
        return None
    return {
        "mean_percentile": _mean([p.get("mean_percentile") for p in pop_list]),
        "head_share_top10pct": _mean([p.get("head_share_top10pct") for p in pop_list]),
        "mean_count": _mean([p.get("mean_count") for p in pop_list]),
    }


def _aggregate_rating(rating_list: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not rating_list:
        return None
    return {
        "mean_rating": _mean([r.get("mean_rating") for r in rating_list]),
        "mean_percentile": _mean([r.get("mean_percentile") for r in rating_list]),
    }


def _aggregate_genre(genre_list: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not genre_list:
        return None
    return {
        "js_divergence_vs_catalog": _mean([g.get("js_divergence_vs_catalog") for g in genre_list]),
        "categories_entropy": _mean([g.get("categories_entropy") for g in genre_list]),
        "category_coverage": _mean([g.get("category_coverage") for g in genre_list]),
    }


def _aggregate_redundancy(red_list: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not red_list:
        return None
    new_counts = [len(r.get("new_items") or []) for r in red_list]
    rep_counts = [len(r.get("repeated_items") or []) for r in red_list]
    return {
        "new_avg": _mean(new_counts),
        "repeated_avg": _mean(rep_counts),
    }


# ---------------- ReDial helpers ----------------


def _find_redial_emotion_path(cache_root: Path, split: str, conv_id: str) -> Optional[Path]:
    root = cache_root / "emotion" / "go_emotions" / "redial"
    top_dir = _find_top_dir(root, "top5")
    for mode in ("titles", "raw"):
        if top_dir:
            p = top_dir / mode / split / f"{conv_id}.json"
            if p.exists():
                return p
        for td in sorted(root.glob("top*")):
            p = td / mode / split / f"{conv_id}.json"
            if p.exists():
                return p
    # fallback: old layout without top_k directory
    for mode in ("titles", "raw"):
        p = root / mode / split / f"{conv_id}.json"
        if p.exists():
            return p
    return None


def _find_redial_stereo_path(cache_root: Path, split: str, conv_id: str) -> Optional[Path]:
    root = cache_root / "bias" / "stereotype" / "redial"
    top_dir = _find_top_dir(root, "top1")
    if top_dir:
        p = top_dir / split / f"{conv_id}.json"
        if p.exists():
            return p
    for td in sorted(root.glob("top*")):
        p = td / split / f"{conv_id}.json"
        if p.exists():
            return p
    return None


def _pick_redial_conv_id(cache_root: Path, split: str) -> Optional[str]:
    root = cache_root / "emotion" / "go_emotions" / "redial"
    for td in sorted(root.glob("top*")):
        for mode in ("titles", "raw"):
            f = _first_json_file(td / mode / split)
            if f:
                data = _load_json(f)
                if data and data.get("conversationId"):
                    return str(data["conversationId"])
    # fallback: old layout without top_k directory
    for mode in ("titles", "raw"):
        f = _first_json_file(root / mode / split)
        if f:
            data = _load_json(f)
            if data and data.get("conversationId"):
                return str(data["conversationId"])
    return None


def _redial_bias_path(cache_root: Path, bias: str, split: str, conv_id: str) -> Path:
    candidates = [
        cache_root / "bias" / bias / "redial" / "ml-25m" / split / f"{conv_id}.json",
        cache_root / "bias" / bias / "ml-25m" / split / f"{conv_id}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _load_redial_biases(cache_root: Path, split: str, conv_id: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for bias in ("popularity", "redundancy", "genre", "year_decade"):
        p = _redial_bias_path(cache_root, bias, split, conv_id)
        data = _load_json(p)
        if data:
            out[bias] = data
    stereo_p = _find_redial_stereo_path(cache_root, split, conv_id)
    stereo = _load_json(stereo_p)
    if stereo:
        out["stereotype"] = stereo
    return out


def _print_redial(cache_root: Path, split: str, conv_id: Optional[str]) -> None:
    ds = ReDialDataset(cache_root=cache_root)
    if conv_id is None:
        conv_id = _pick_redial_conv_id(cache_root, split)

    conv = None
    if conv_id:
        for c in ds.iter(split=split):
            if str(c.conversation_id) == str(conv_id):
                conv = c
                break
    if conv is None:
        for c in ds.iter(split=split):
            conv = c
            break

    if conv is None:
        print("No ReDial conversation found.")
        return

    conv_id = str(conv.conversation_id)
    emo_p = _find_redial_emotion_path(cache_root, split, conv_id)
    emo = _load_json(emo_p) or {}
    emo_turns = {t.get("msg_idx"): t.get("emotion") for t in emo.get("turns", []) if isinstance(t, dict)}

    biases = _load_redial_biases(cache_root, split, conv_id)
    pop_turns = {t.get("msg_idx"): t for t in biases.get("popularity", {}).get("turns", [])}
    red_turns = {
        t.get("msg_idx"): (t.get("new_items"), t.get("repeated_items"))
        for t in biases.get("redundancy", {}).get("turns", [])
    }
    genre_turns = {
        t.get("msg_idx"): t.get("genre_bias") for t in biases.get("genre", {}).get("turns", [])
    }
    yd_turns = {
        t.get("msg_idx"): t.get("year_decade_bias")
        for t in biases.get("year_decade", {}).get("turns", [])
    }
    has_year_decade = "year_decade" in biases
    stereo_turns = {
        t.get("msg_idx"): t.get("bias") for t in biases.get("stereotype", {}).get("turns", [])
    }

    print(f"== ReDial ({split}) conversation {conv_id} ==")
    print("Legend: A=Seeker, B=Recommender\n")

    for i, msg in enumerate(conv.messages):
        text = msg.get("text", "") if isinstance(msg, dict) else ""
        speaker = redial_get_speaker(msg, i)
        tag = "A" if speaker == "Seeker" else "B"
        print(f"{tag}: {text}")

        if speaker == "Seeker":
            if i in emo_turns:
                preds = emo_turns.get(i)
                print(f"emotion: {_fmt_preds(preds)}")
            else:
                print("emotion: missing cache")
        else:
            movie_tags = extract_movie_ids(text)
            if movie_tags:
                print(f"movies: {', '.join('@' + m for m in movie_tags)}")
            if i in stereo_turns:
                print(f"bias stereotype: {_fmt_preds(stereo_turns.get(i))}")
            if i in pop_turns:
                t = pop_turns.get(i) or {}
                pop_obj = t.get("popularity")
                unmapped = t.get("unmapped_redial_movie_ids") or []
                if pop_obj:
                    print(f"bias popularity: {_fmt_popularity(pop_obj)}")
                else:
                    reason = ""
                    if unmapped:
                        reason = f" (unmapped {', '.join('@' + m for m in unmapped)})"
                    elif movie_tags:
                        reason = " (no MovieLens mapping)"
                    else:
                        reason = " (no movie tags)"
                    print(f"bias popularity: none{reason}")
            if i in genre_turns:
                obj = genre_turns.get(i)
                if obj:
                    print(f"bias genre: {_fmt_genre(obj, 'js_divergence_vs_catalog')}")
                elif movie_tags:
                    print("bias genre: none (no MovieLens mapping)")
                else:
                    print("bias genre: none (no movie tags)")
            if i in yd_turns:
                obj = yd_turns.get(i)
                if obj:
                    print(f"bias year/decade: {_fmt_year_decade(obj)}")
                elif movie_tags:
                    print("bias year/decade: none (no MovieLens mapping)")
                else:
                    print("bias year/decade: none (no movie tags)")
            elif movie_tags and not has_year_decade:
                print("bias year/decade: missing cache (run biases)")
            if i in red_turns:
                new_items, repeated_items = red_turns.get(i) or ([], [])
                print(f"bias redundancy: {_fmt_redundancy(new_items, repeated_items)}")
        print("")


# ---------------- CoSRec helpers ----------------


def _pick_cosrec_conv_id(cache_root: Path) -> Optional[str]:
    emo_root = cache_root / "emotion" / "go_emotions" / "cosrec"
    top_dir = _find_top_dir(emo_root, "top5")
    if top_dir:
        f = _first_json_file(top_dir / "curated")
        if f:
            data = _load_json(f)
            if data and data.get("conversation_id"):
                return str(data["conversation_id"])
    stereo_root = cache_root / "bias" / "stereotype" / "cosrec"
    top_dir = _find_top_dir(stereo_root, "top1")
    if top_dir:
        f = _first_json_file(top_dir / "curated")
        if f:
            data = _load_json(f)
            if data and data.get("conversation_id"):
                return str(data["conversation_id"])
    return None


def _find_cosrec_emotion_path(cache_root: Path, topic_id: str) -> Optional[Path]:
    key = cosrec_safe_id(topic_id)
    root = cache_root / "emotion" / "go_emotions" / "cosrec"
    top_dir = _find_top_dir(root, "top5")
    if top_dir:
        p = top_dir / "curated" / f"{key}.json"
        if p.exists():
            return p
    for td in sorted(root.glob("top*")):
        p = td / "curated" / f"{key}.json"
        if p.exists():
            return p
    return None


def _find_cosrec_stereo_path(cache_root: Path, topic_id: str) -> Optional[Path]:
    key = cosrec_safe_id(topic_id)
    root = cache_root / "bias" / "stereotype" / "cosrec"
    top_dir = _find_top_dir(root, "top1")
    if top_dir:
        p = top_dir / "curated" / f"{key}.json"
        if p.exists():
            return p
    for td in sorted(root.glob("top*")):
        p = td / "curated" / f"{key}.json"
        if p.exists():
            return p
    return None


def _cosrec_bias_path(cache_root: Path, bias: str, topic_id: str) -> Path:
    base = cache_root / "bias" / bias / "cosrec" / "amazon_2023" / "curated"
    return base / f"{cosrec_safe_id(topic_id)}.json"


def _load_cosrec_biases(cache_root: Path, topic_id: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for bias in ("popularity", "rating", "genre", "redundancy"):
        data = _load_json(_cosrec_bias_path(cache_root, bias, topic_id))
        if data:
            out[bias] = data
    stereo = _load_json(_find_cosrec_stereo_path(cache_root, topic_id))
    if stereo:
        out["stereotype"] = stereo
    return out


def _print_cosrec(
    cache_root: Path,
    conv_id: Optional[str],
    intent_type: str,
    min_relevance: int,
    mode: str,
) -> None:
    ds = CoSRecDataset(cache_root=cache_root)
    if conv_id is None:
        conv_id = _pick_cosrec_conv_id(cache_root)

    conv = None
    if conv_id:
        for c in ds.iter_conversations("curated"):
            if str(c.conversation_id) == str(conv_id):
                conv = c
                break
    if conv is None:
        for c in ds.iter_conversations("curated"):
            conv = c
            break

    if conv is None:
        print("No CoSRec conversation found.")
        return

    conv_id = str(conv.conversation_id)

    eps = []
    for ep in ds.iter_rec_episodes(min_relevance=min_relevance, require_next_user=False):
        if intent_type and ep.intent_type != intent_type:
            continue
        if str(ep.conversation_id) == conv_id:
            eps.append(ep)

    by_sys: Dict[int, List[Dict[str, Any]]] = {}
    by_user: Dict[int, List[Dict[str, Any]]] = {}

    for ep in eps:
        topic_id = str(ep.topic_id)
        emo = _load_json(_find_cosrec_emotion_path(cache_root, topic_id))
        biases = _load_cosrec_biases(cache_root, topic_id)

        by_sys.setdefault(int(ep.system_turn_idx), []).append(
            {
                "topic_id": topic_id,
                "biases": biases,
            }
        )
        if emo:
            by_user.setdefault(int(ep.next_user_turn_idx), []).append(
                {
                    "topic_id": topic_id,
                    "emotion": emo.get("emotion"),
                }
            )

    print(f"== CoSRec (curated) conversation {conv_id} ==")
    print("Legend: A=User, B=System\n")

    for i, t in enumerate(conv.turns):
        speaker = t.get("speaker", "?")
        tag = "A" if speaker == "U" else "B"
        text = t.get("text", "")
        print(f"{tag}: {text}")

        if i in by_user:
            emo_items = by_user[i]
            if mode == "episodes":
                for item in emo_items:
                    topic = item.get("topic_id", "")[:20]
                    print(f"emotion ({topic}): {_fmt_preds(item.get('emotion'))}")
            else:
                agg = _aggregate_emotions([e.get("emotion") for e in emo_items])
                if agg:
                    print(f"emotion: {_fmt_preds(agg)} (episodes={len(emo_items)})")
                else:
                    print("emotion: none")
        elif speaker == "U":
            print("emotion: none (no rec episode)")

        if i in by_sys:
            items = by_sys[i]
            if mode == "episodes":
                for item in items:
                    topic = item.get("topic_id", "")[:20]
                    biases = item.get("biases") or {}
                    stereo = biases.get("stereotype", {}).get("bias")
                    pop = biases.get("popularity", {}).get("popularity")
                    rating = biases.get("rating", {}).get("rating")
                    genre = biases.get("genre", {}).get("summary")
                    red = biases.get("redundancy", {})
                    new_items = red.get("new_items") or []
                    rep_items = red.get("repeated_items") or []

                    if stereo is not None:
                        print(f"bias stereotype ({topic}): {_fmt_preds(stereo)}")
                    if pop is not None:
                        print(f"bias popularity ({topic}): {_fmt_popularity(pop)}")
                    if rating is not None:
                        print(f"bias rating ({topic}): {_fmt_rating(rating)}")
                    if genre:
                        print(f"bias genre ({topic}): {_fmt_genre(genre, 'js_divergence_vs_catalog')}")
                    if new_items or rep_items:
                        print(f"bias redundancy ({topic}): {_fmt_redundancy(new_items, rep_items)}")
            else:
                bias_items = [item.get("biases") or {} for item in items]
                stereo_list = [b.get("stereotype", {}).get("bias") for b in bias_items if b.get("stereotype")]
                pop_list = [b.get("popularity", {}).get("popularity") for b in bias_items if b.get("popularity")]
                rating_list = [b.get("rating", {}).get("rating") for b in bias_items if b.get("rating")]
                genre_list = [b.get("genre", {}).get("summary") for b in bias_items if b.get("genre")]
                red_list = [b.get("redundancy") for b in bias_items if b.get("redundancy")]

                stereo_agg = _aggregate_stereotype(stereo_list)
                if stereo_agg:
                    print(
                        "bias stereotype: "
                        f"label1_rate={_fmt_float(stereo_agg.get('label1_rate'))} "
                        f"label1_avg={_fmt_float(stereo_agg.get('label1_avg_score'))} "
                        f"(episodes={len(stereo_list)})"
                    )

                pop_agg = _aggregate_popularity([p for p in pop_list if p])
                if pop_agg:
                    print(
                        "bias popularity: "
                        f"mean_pct={_fmt_float(pop_agg.get('mean_percentile'))} "
                        f"head_top10={_fmt_float(pop_agg.get('head_share_top10pct'))} "
                        f"mean_count={_fmt_float(pop_agg.get('mean_count'), 1)} "
                        f"(episodes={len(pop_list)})"
                    )

                rating_agg = _aggregate_rating([r for r in rating_list if r])
                if rating_agg:
                    print(
                        "bias rating: "
                        f"mean_rating={_fmt_float(rating_agg.get('mean_rating'))} "
                        f"mean_pct={_fmt_float(rating_agg.get('mean_percentile'))} "
                        f"(episodes={len(rating_list)})"
                    )

                genre_agg = _aggregate_genre([g for g in genre_list if g])
                if genre_agg:
                    print(
                        "bias genre: "
                        f"js={_fmt_float(genre_agg.get('js_divergence_vs_catalog'))} "
                        f"entropy={_fmt_float(genre_agg.get('categories_entropy'))} "
                        f"coverage={_fmt_float(genre_agg.get('category_coverage'), 1)} "
                        f"(episodes={len(genre_list)})"
                    )

                red_agg = _aggregate_redundancy([r for r in red_list if r])
                if red_agg:
                    print(
                        "bias redundancy: "
                        f"new_avg={_fmt_float(red_agg.get('new_avg'), 1)} "
                        f"repeated_avg={_fmt_float(red_agg.get('repeated_avg'), 1)} "
                        f"(episodes={len(red_list)})"
                    )
        elif speaker == "S":
            print("bias: none (no rec episode)")

        print("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=str(ROOT / "cache"))
    parser.add_argument("--redial-split", default="train")
    parser.add_argument("--redial-id", default=None)
    parser.add_argument("--cosrec-id", default=None)
    parser.add_argument("--intent-type", default="recommendation")
    parser.add_argument("--min-relevance", type=int, default=1)
    parser.add_argument("--cosrec-mode", choices=["aggregate", "episodes"], default="aggregate")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    _print_redial(cache_root, args.redial_split, args.redial_id)
    _print_cosrec(cache_root, args.cosrec_id, args.intent_type, args.min_relevance, args.cosrec_mode)


if __name__ == "__main__":
    main()
