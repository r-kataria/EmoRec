#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import CoSRecDataset, ReDialDataset  # noqa: E402
from dataset.cosrec import safe_id as cosrec_safe_id  # noqa: E402
from dataset.redial import get_speaker, replace_movie_tags_with_titles, safe_id as redial_safe_id  # noqa: E402


def _ascii(text: Any) -> str:
    if text is None:
        return ""
    return str(text).encode("ascii", "replace").decode("ascii")


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sorted_top_dirs(base: Path) -> List[Path]:
    if not base.exists():
        return []
    items = []
    for p in base.iterdir():
        if not p.is_dir() or not p.name.startswith("top"):
            continue
        suffix = p.name[3:]
        try:
            order = int(suffix)
        except Exception:
            order = -1
        items.append((order, p.name, p))
    items.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, _, p in items]


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _pick_cache_id_from_dirs(dirs: Iterable[Path]) -> Optional[str]:
    for d in dirs:
        if not d.exists() or not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix == ".json":
                return p.stem
    return None


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "none"


def _format_scores(items: Any, limit: Optional[int] = None) -> str:
    if not isinstance(items, list) or not items:
        return "none"
    out = []
    for d in items[:limit] if limit else items:
        if not isinstance(d, dict):
            continue
        label = _ascii(d.get("label", ""))
        score = d.get("score")
        if score is None:
            out.append(label)
        else:
            out.append(f"{label} ({_fmt_float(score)})")
    if limit and len(items) > limit:
        out.append(f"+{len(items) - limit} more")
    return ", ".join(out) if out else "none"


def _index_turns(cache: Optional[Dict[str, Any]], idx_key: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not cache:
        return out
    for t in cache.get("turns", []) or []:
        try:
            idx = int(t.get(idx_key))
        except Exception:
            continue
        out[idx] = t
    return out


def _redial_emotion_path(cache_root: Path, split: str, conv_id: str) -> Optional[Path]:
    safe = redial_safe_id(conv_id)
    path = (
        cache_root
        / "emotion"
        / "go_emotions"
        / "redial"
        / "top5"
        / "titles"
        / split
        / f"{safe}.json"
    )
    return path if path.exists() else None


def _redial_stereotype_path(cache_root: Path, split: str, conv_id: str) -> Optional[Path]:
    safe = redial_safe_id(conv_id)
    base = cache_root / "bias" / "stereotype" / "redial"
    candidates = [base / split / f"{safe}.json"]
    for top_dir in _sorted_top_dirs(base):
        candidates.append(top_dir / split / f"{safe}.json")
    return _first_existing(candidates)


def _redial_bias_path(cache_root: Path, bias: str, split: str, conv_id: str) -> Path:
    safe = redial_safe_id(conv_id)
    return cache_root / "bias" / bias / "redial" / "ml-25m" / split / f"{safe}.json"


def _redial_exposure_path(cache_root: Path, split: str) -> Path:
    return cache_root / "bias" / "exposure_concentration" / "redial" / "ml-25m" / f"{split}.json"


def _cosrec_turn_emotion_path(cache_root: Path, conv_id: str) -> Optional[Path]:
    safe = cosrec_safe_id(conv_id)
    base = cache_root / "emotion" / "go_emotions" / "cosrec"
    candidates = []
    for top_dir in _sorted_top_dirs(base):
        candidates.append(top_dir / "curated_turns" / f"{safe}.json")
    return _first_existing(candidates)


def _cosrec_bias_path(cache_root: Path, bias: str, topic_id: str) -> Optional[Path]:
    safe = cosrec_safe_id(topic_id)
    if bias == "stereotype":
        base = cache_root / "bias" / "stereotype" / "cosrec"
        candidates = [base / "curated" / f"{safe}.json"]
        for top_dir in _sorted_top_dirs(base):
            candidates.append(top_dir / "curated" / f"{safe}.json")
        return _first_existing(candidates)
    return cache_root / "bias" / bias / "cosrec" / "amazon_2023" / "curated" / f"{safe}.json"


def _cosrec_exposure_path(cache_root: Path) -> Path:
    return cache_root / "bias" / "exposure_concentration" / "cosrec" / "amazon_2023" / "curated.json"


def _load_catalogue_index(cache_root: Path) -> Dict[str, Dict[str, Any]]:
    p = cache_root / "datasets" / "amazon_2023" / "processed" / "catalogue_index.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _find_redial_conversation(
    ds: ReDialDataset,
    split: str,
    conv_id: Optional[str],
    cache_root: Path,
) -> Optional[Any]:
    target = conv_id
    if target is None:
        dirs = []
        dirs.append(cache_root / "emotion" / "go_emotions" / "redial" / "titles" / split)
        base_emo = cache_root / "emotion" / "go_emotions" / "redial"
        for top_dir in _sorted_top_dirs(base_emo):
            dirs.append(top_dir / "titles" / split)
        dirs.append(cache_root / "bias" / "popularity" / "redial" / "ml-25m" / split)
        dirs.append(cache_root / "bias" / "genre" / "redial" / "ml-25m" / split)
        dirs.append(cache_root / "bias" / "redundancy" / "redial" / "ml-25m" / split)
        dirs.append(cache_root / "bias" / "year_decade" / "redial" / "ml-25m" / split)
        base_st = cache_root / "bias" / "stereotype" / "redial"
        for top_dir in _sorted_top_dirs(base_st):
            dirs.append(top_dir / split)
        target = _pick_cache_id_from_dirs(dirs)

    for conv in ds.iter(split=split):
        if target is None or str(conv.conversation_id) == str(target):
            return conv
    return None


def _find_cosrec_conversation(
    ds: CoSRecDataset,
    conv_id: Optional[str],
    cache_root: Path,
) -> Optional[Any]:
    target = conv_id
    if target is None:
        dirs = []
        base = cache_root / "emotion" / "go_emotions" / "cosrec"
        for top_dir in _sorted_top_dirs(base):
            dirs.append(top_dir / "curated_turns")
        target = _pick_cache_id_from_dirs(dirs)

    for conv in ds.iter_conversations("curated"):
        if target is None or str(conv.conversation_id) == str(target):
            return conv
    return None


def _format_redial_biases(
    idx: int,
    stereo_turns: Dict[int, Dict[str, Any]],
    pop_turns: Dict[int, Dict[str, Any]],
    epi_turns: Dict[int, Dict[str, Any]],
    genre_turns: Dict[int, Dict[str, Any]],
    year_turns: Dict[int, Dict[str, Any]],
    red_turns: Dict[int, Dict[str, Any]],
    missing: List[str],
) -> List[str]:
    lines = []
    if "stereotype" in missing:
        lines.append("bias stereotype: missing cache")
    else:
        bias = (stereo_turns.get(idx) or {}).get("bias")
        lines.append(f"bias stereotype: {_format_scores(bias)}")
    if "popularity" in missing:
        lines.append("bias popularity: missing cache")
    else:
        pop = (pop_turns.get(idx) or {}).get("popularity") or {}
        if pop:
            mean_pct = _fmt_float(pop.get("mean_percentile"))
            head = _fmt_float(pop.get("head_share_top10pct"))
            mean_count = _fmt_float(pop.get("mean_count"), 1)
            lines.append(f"bias popularity: mean_pct={mean_pct} head_top10={head} mean_count={mean_count}")
        else:
            lines.append("bias popularity: none")
    if "episode_popularity" in missing:
        lines.append("bias episode_popularity: missing cache")
    else:
        ep = (epi_turns.get(idx) or {}).get("episode_popularity") or {}
        if ep:
            p = _fmt_float(ep.get("p_coverage"))
            pi = _fmt_float(ep.get("pi_rank_utility"))
            cep = _fmt_float(ep.get("cep_similarity"))
            uiop = _fmt_float(ep.get("uiop_similarity"))
            lines.append(f"bias episode_popularity: P={p} pi={pi} CEP={cep} UIOP={uiop}")
        else:
            lines.append("bias episode_popularity: none")
    if "genre" in missing:
        lines.append("bias genre: missing cache")
    else:
        gb = (genre_turns.get(idx) or {}).get("genre_bias") or {}
        if gb:
            js = _fmt_float(gb.get("js_divergence_vs_catalog"))
            ent = _fmt_float(gb.get("genres_entropy"))
            cov = gb.get("genre_coverage")
            lines.append(f"bias genre: js={js} entropy={ent} coverage={cov}")
        else:
            lines.append("bias genre: none")
    if "year_decade" in missing:
        lines.append("bias year/decade: missing cache")
    else:
        yd = (year_turns.get(idx) or {}).get("year_decade_bias") or {}
        if yd:
            js = _fmt_float(yd.get("decades_js_divergence_vs_catalog"))
            ent = _fmt_float(yd.get("decades_entropy"))
            cov = yd.get("decade_coverage")
            mean_year = _fmt_float(yd.get("mean_year"), 1)
            lines.append(f"bias year/decade: decade_js={js} decade_entropy={ent} coverage={cov} mean_year={mean_year}")
        else:
            lines.append("bias year/decade: none")
    if "redundancy" in missing:
        lines.append("bias redundancy: missing cache")
    else:
        rb = red_turns.get(idx) or {}
        new_items = rb.get("new_items") or []
        rep_items = rb.get("repeated_items") or []
        lines.append(f"bias redundancy: new={len(new_items)} repeated={len(rep_items)}")
    return lines


def _format_cosrec_biases(
    cache_root: Path,
    topic_id: str,
) -> List[str]:
    lines = []
    st_path = _cosrec_bias_path(cache_root, "stereotype", topic_id)
    st = _load_json(st_path)
    if not st_path or not st_path.exists():
        lines.append("bias stereotype: missing cache")
    else:
        bias = (st or {}).get("bias")
        lines.append(f"bias stereotype: {_format_scores(bias)}")

    pop_path = _cosrec_bias_path(cache_root, "popularity", topic_id)
    pop = _load_json(pop_path)
    if not pop_path or not pop_path.exists():
        lines.append("bias popularity: missing cache")
    else:
        obj = (pop or {}).get("popularity") or {}
        if obj:
            mean_pct = _fmt_float(obj.get("mean_percentile"))
            head = _fmt_float(obj.get("head_share_top10pct"))
            mean_count = _fmt_float(obj.get("mean_count"), 1)
            lines.append(f"bias popularity: mean_pct={mean_pct} head_top10={head} mean_count={mean_count}")
        else:
            lines.append("bias popularity: none")

    rat_path = _cosrec_bias_path(cache_root, "rating", topic_id)
    rat = _load_json(rat_path)
    if not rat_path or not rat_path.exists():
        lines.append("bias rating: missing cache")
    else:
        obj = (rat or {}).get("rating") or {}
        if obj:
            mean_rating = _fmt_float(obj.get("mean_rating"))
            mean_pct = _fmt_float(obj.get("mean_percentile"))
            lines.append(f"bias rating: mean_rating={mean_rating} mean_pct={mean_pct}")
        else:
            lines.append("bias rating: none")

    gen_path = _cosrec_bias_path(cache_root, "genre", topic_id)
    gen = _load_json(gen_path)
    if not gen_path or not gen_path.exists():
        lines.append("bias genre: missing cache")
    else:
        summary = (gen or {}).get("summary") or {}
        if summary:
            js = _fmt_float(summary.get("js_divergence_vs_catalog"))
            ent = _fmt_float(summary.get("categories_entropy"))
            cov = summary.get("category_coverage")
            lines.append(f"bias genre: js={js} entropy={ent} coverage={cov}")
        else:
            lines.append("bias genre: none")

    red_path = _cosrec_bias_path(cache_root, "redundancy", topic_id)
    red = _load_json(red_path)
    if not red_path or not red_path.exists():
        lines.append("bias redundancy: missing cache")
    else:
        new_items = (red or {}).get("new_items") or []
        rep_items = (red or {}).get("repeated_items") or []
        lines.append(f"bias redundancy: new={len(new_items)} repeated={len(rep_items)}")

    epi_path = _cosrec_bias_path(cache_root, "episode_popularity", topic_id)
    epi = _load_json(epi_path)
    if not epi_path or not epi_path.exists():
        lines.append("bias episode_popularity: missing cache")
    else:
        ep = (epi or {}).get("episode_popularity") or {}
        if ep:
            p = _fmt_float(ep.get("p_coverage"))
            pi = _fmt_float(ep.get("pi_rank_utility"))
            cep = _fmt_float(ep.get("cep_similarity"))
            uiop = _fmt_float(ep.get("uiop_similarity"))
            lines.append(f"bias episode_popularity: P={p} pi={pi} CEP={cep} UIOP={uiop}")
        else:
            lines.append("bias episode_popularity: none")
    return lines


def _format_items(
    qrels: List[tuple[str, int]],
    meta_index: Dict[str, Dict[str, Any]],
    limit: int,
) -> str:
    parts = []
    for asin, rel in qrels[:limit]:
        meta = meta_index.get(str(asin), {})
        item_parts = [str(asin), f"rel={rel}"]
        price = meta.get("price")
        if isinstance(price, (int, float)) and price > 0:
            item_parts.append(f"price={_fmt_float(price, 2)}")
        rating = meta.get("avg_valid_ratings") or meta.get("avg_ratings")
        if rating is not None:
            item_parts.append(f"rating={_fmt_float(rating)}")
        count = meta.get("num_valid_ratings") or meta.get("num_ratings")
        if count:
            item_parts.append(f"ratings={count}")
        store = meta.get("store")
        if store:
            item_parts.append(f"store={_ascii(store)}")
        cats = meta.get("categories") or []
        if isinstance(cats, list) and cats:
            item_parts.append(f"cats={_ascii('/'.join(cats[:2]))}")
        parts.append(f"{', '.join(item_parts)}")
    if len(qrels) > limit:
        parts.append(f"+{len(qrels) - limit} more")
    return "; ".join(parts) if parts else "none"


def render_redial(
    cache_root: Path,
    split: str,
    conv_id: Optional[str],
) -> List[str]:
    ds = ReDialDataset(cache_root=cache_root)
    conv = _find_redial_conversation(ds, split, conv_id, cache_root)
    if conv is None:
        return [f"== ReDial ({split}) ==", "No conversation found."]

    emotion_path = _redial_emotion_path(cache_root, split, conv.conversation_id)
    emotion_cache = _load_json(emotion_path)
    emo_turns = _index_turns(emotion_cache, "msg_idx")

    missing = []
    stereo_path = _redial_stereotype_path(cache_root, split, conv.conversation_id)
    pop_path = _redial_bias_path(cache_root, "popularity", split, conv.conversation_id)
    epi_path = _redial_bias_path(cache_root, "episode_popularity", split, conv.conversation_id)
    genre_path = _redial_bias_path(cache_root, "genre", split, conv.conversation_id)
    year_path = _redial_bias_path(cache_root, "year_decade", split, conv.conversation_id)
    red_path = _redial_bias_path(cache_root, "redundancy", split, conv.conversation_id)

    stereo_cache = _load_json(stereo_path)
    pop_cache = _load_json(pop_path)
    epi_cache = _load_json(epi_path)
    genre_cache = _load_json(genre_path)
    year_cache = _load_json(year_path)
    red_cache = _load_json(red_path)

    if not stereo_cache:
        missing.append("stereotype")
    if not pop_cache:
        missing.append("popularity")
    if not epi_cache:
        missing.append("episode_popularity")
    if not genre_cache:
        missing.append("genre")
    if not year_cache:
        missing.append("year_decade")
    if not red_cache:
        missing.append("redundancy")

    stereo_turns = _index_turns(stereo_cache, "msg_idx")
    pop_turns = _index_turns(pop_cache, "msg_idx")
    epi_turns = _index_turns(epi_cache, "msg_idx")
    genre_turns = _index_turns(genre_cache, "msg_idx")
    year_turns = _index_turns(year_cache, "msg_idx")
    red_turns = _index_turns(red_cache, "msg_idx")

    exposure_cache = _load_json(_redial_exposure_path(cache_root, split))

    mentions = ds.movie_mentions_map()
    mentions.update(conv.movie_mentions or {})

    lines = [
        f"== ReDial ({split}) conversation {_ascii(conv.conversation_id)} ==",
        "Legend: A=Seeker, B=Recommender",
    ]
    if emotion_cache:
        lines.append("Cache: emotion=ok")
    else:
        lines.append("Cache: emotion=missing")
    if missing:
        lines.append(f"Missing bias caches: {', '.join(missing)}")
    if exposure_cache:
        top10 = _fmt_float(exposure_cache.get("top10_share"))
        gini = _fmt_float(exposure_cache.get("gini"))
        hhi = _fmt_float(exposure_cache.get("hhi"))
        lines.append(f"Exposure (dataset-level): top10_share={top10} gini={gini} hhi={hhi}")
    lines.append("")

    for i, msg in enumerate(conv.messages):
        raw_text = msg.get("text", "") if isinstance(msg, dict) else ""
        text = replace_movie_tags_with_titles(raw_text, mentions)
        speaker = get_speaker(msg, i)
        prefix = "A" if speaker == "Seeker" else "B"
        lines.append(f"{prefix}: {_ascii(text)}")
        if speaker == "Seeker":
            if not emotion_cache:
                lines.append("emotion: missing cache")
            else:
                emo = (emo_turns.get(i) or {}).get("emotion")
                lines.append(f"emotion: {_format_scores(emo)}")
        else:
            lines.extend(
                _format_redial_biases(
                    i,
                    stereo_turns,
                    pop_turns,
                    epi_turns,
                    genre_turns,
                    year_turns,
                    red_turns,
                    missing,
                )
            )
        lines.append("")
    return lines


def render_cosrec(
    cache_root: Path,
    conv_id: Optional[str],
    item_limit: int,
) -> List[str]:
    ds = CoSRecDataset(cache_root=cache_root)
    conv = _find_cosrec_conversation(ds, conv_id, cache_root)
    if conv is None:
        return ["== CoSRec (curated) ==", "No conversation found."]

    emotion_path = _cosrec_turn_emotion_path(cache_root, conv.conversation_id)
    emotion_cache = _load_json(emotion_path)
    emo_turns = _index_turns(emotion_cache, "turn_idx")

    exposure_cache = _load_json(_cosrec_exposure_path(cache_root))
    meta_index = _load_catalogue_index(cache_root)

    episodes = []
    for ep in ds.iter_rec_episodes(min_relevance=1):
        if str(ep.conversation_id) == str(conv.conversation_id):
            episodes.append(ep)

    by_system: Dict[int, List[Any]] = {}
    for ep in episodes:
        by_system.setdefault(int(ep.system_turn_idx), []).append(ep)

    for eps in by_system.values():
        eps.sort(key=lambda e: (int(e.utterance_idx), int(e.user_index), str(e.topic_id)))

    lines = [
        f"== CoSRec (curated) conversation {_ascii(conv.conversation_id)} ==",
        "Legend: A=User, B=System",
    ]
    if emotion_cache:
        lines.append("Cache: turn_emotion=ok")
    else:
        lines.append("Cache: turn_emotion=missing")
    if exposure_cache:
        top10 = _fmt_float(exposure_cache.get("top10_share"))
        gini = _fmt_float(exposure_cache.get("gini"))
        hhi = _fmt_float(exposure_cache.get("hhi"))
        lines.append(f"Exposure (dataset-level): top10_share={top10} gini={gini} hhi={hhi}")
    lines.append("")

    for i, turn in enumerate(conv.turns):
        speaker = turn.get("speaker")
        text = turn.get("text", "")
        prefix = "A" if speaker == "U" else "B" if speaker == "S" else "?"
        lines.append(f"{prefix}: {_ascii(text)}")

        if speaker == "U":
            if not emotion_cache:
                lines.append("emotion: missing cache")
            else:
                emo = (emo_turns.get(i) or {}).get("emotion")
                lines.append(f"emotion: {_format_scores(emo)}")
        elif speaker == "S":
            eps = by_system.get(i, [])
            if not eps:
                lines.append("bias: none (no rec episode)")
            for ep in eps:
                lines.append(
                    f"episode {_ascii(ep.topic_id)} (intent={ep.intent_type}, user_index={ep.user_index}, utterance_idx={ep.utterance_idx})"
                )
                if ep.product:
                    lines.append(f"product: {_ascii(ep.product)}")
                ordered = sorted(ep.qrels, key=lambda x: (-int(x[1]), str(x[0])))
                items = _format_items([(str(a), int(r)) for a, r in ordered], meta_index, item_limit)
                lines.append(f"items: {items}")
                lines.extend(_format_cosrec_biases(cache_root, ep.topic_id))
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print one cached ReDial and CoSRec conversation with emotions/biases."
    )
    parser.add_argument("--cache-root", default=str(ROOT / "cache"))
    parser.add_argument("--redial-split", default="train")
    parser.add_argument("--redial-id", default=None)
    parser.add_argument("--cosrec-id", default=None)
    parser.add_argument("--item-limit", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "man.txt"))
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    lines: List[str] = []
    lines.extend(render_redial(cache_root, args.redial_split, args.redial_id))
    lines.append("")
    lines.extend(render_cosrec(cache_root, args.cosrec_id, args.item_limit))

    text = "\n".join(lines).strip() + "\n"
    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
