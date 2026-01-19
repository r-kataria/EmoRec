from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from collections import defaultdict


_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)


def safe_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))[:240] or "unknown"


def _read_jsonl_single_key(path: Path) -> Iterator[Tuple[str, Any]]:
    """
    CoSRec jsonl files store each record as a dict with a single key:
      { "<conversation_id>": <value> }
    """
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if not isinstance(obj, dict) or len(obj) != 1:
                continue
            (k, v), = obj.items()
            yield str(k), v


def _parse_conversation_text(text: str) -> List[Dict[str, str]]:
    """
    "U: ...\nS: ...\nU: ..." -> [{"speaker":"U","text":...}, ...]
    """
    turns: List[Dict[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("U:"):
            turns.append({"speaker": "U", "text": line[2:].strip()})
        elif line.startswith("S:"):
            turns.append({"speaker": "S", "text": line[2:].strip()})
        else:
            turns.append({"speaker": "?", "text": line})
    return turns


def _user_utterance_index_to_turn_index(turns: List[Dict[str, str]], utterance_idx: int) -> Optional[int]:
    """
    CoSRec 'utterance' index counts USER utterances only.
    """
    u = -1
    for i, t in enumerate(turns):
        if t.get("speaker") == "U":
            u += 1
            if u == utterance_idx:
                return i
    return None


def _next_turn_index(turns: List[Dict[str, str]], start: int, speaker: str) -> Optional[int]:
    for j in range(start, len(turns)):
        if turns[j].get("speaker") == speaker:
            return j
    return None


def _split_intent_id(intent_id: str) -> Tuple[str, int, int]:
    """
    Curated intent id format: <conversation_id>_<utterance_id>_<index>
    Example: CoSRec-Curated_1_3_0
    """
    conv_id, u_str, i_str = str(intent_id).rsplit("_", 2)
    return conv_id, int(u_str), int(i_str)


def _is_asin(doc_id: str) -> bool:
    s = str(doc_id).strip()
    return bool(_ASIN_RE.match(s))


def _is_msmarco(doc_id: str) -> bool:
    s = str(doc_id).strip().lower()
    return "msmarco" in s


@dataclass
class CoSRecConversation:
    dataset: "CoSRecDataset"
    partition: str
    conversation_id: str
    turns: List[Dict[str, str]]


@dataclass
class CoSRecCuratedEpisode:
    """
    Curated-only, item-grounded recommendation episode.
    For recommendation topics, CoSRec uses topic_id like:
      <intent_id>#<user_index>
    Example qrels lines you showed:
      CoSRec-Curated_1_3_0#0  0  B004OA2B22  2
    """
    dataset: "CoSRecDataset"
    conversation_id: str
    topic_id: str
    base_intent_id: str
    user_index: int
    utterance_idx: int
    intent_type: str
    query_variants: List[str]
    qrels: List[Tuple[str, int]]  # ASIN-only qrels for this episode

    user_turn_idx: int
    system_turn_idx: int
    next_user_turn_idx: int

    user_text: str
    system_text: str
    next_user_text: str

    emotion_cache: Any = None
    bias_cache: Any = None

    @property
    def emotion(self) -> Dict[str, Any]:
        if self.emotion_cache is None:
            raise RuntimeError("No emotion_cache attached.")
        return self.emotion_cache.get(self)

    @property
    def bias(self) -> Dict[str, Any]:
        if self.bias_cache is None:
            raise RuntimeError("No bias_cache attached.")
        return self.bias_cache.get(self)


class CoSRecDataset:
    """
    CoSRec dataset loader. Also loads qrels and splits them into:
      - ASIN qrels (product recommendation)
      - MS MARCO qrels (retrieval/search)

    This addresses qrels lines like:
      CoSRec-Curated_1_2_0      0   msmarco_v2.1_doc_...    2
      CoSRec-Curated_1_3_0#0    0   B004OA2B22              2
    """

    REPO_URL = "https://github.com/CAMEO-22/CoSRec"

    def __init__(self, cache_root: Path | str = "./cache", quiet_download: bool = True):
        self.cache_root = Path(cache_root)
        self.root = self.cache_root / "datasets" / "cosrec"
        self.repo_dir = self.root / "repo"
        self.dataset_dir = self.repo_dir / "dataset"
        self.quiet = bool(quiet_download)
        self._ensure_repo()

        self._curated_intent_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._curated_conversations_index: Optional[Dict[str, CoSRecConversation]] = None

        # qrels (all + split)
        self._qrels_all: Optional[Dict[str, List[Tuple[str, int]]]] = None
        self._qrels_asin: Optional[Dict[str, List[Tuple[str, int]]]] = None
        self._qrels_msmarco: Optional[Dict[str, List[Tuple[str, int]]]] = None
        self._asin_doc_ids: Optional[Set[str]] = None
        self._msmarco_doc_ids: Optional[Set[str]] = None

        self._load_qrels()  # load once on init

    # ---------- paths ----------
    def conversations_path(self, partition: str) -> Path:
        return self.dataset_dir / partition / "conversations.jsonl"

    def curated_intents_path(self) -> Path:
        return self.dataset_dir / "curated" / "intents.jsonl"

    def curated_qrels_path(self) -> Path:
        return self.dataset_dir / "curated" / "qrels.qrels"

    # ---------- repo ----------
    def _ensure_repo(self) -> None:
        if self.repo_dir.exists() and self.dataset_dir.exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", self.REPO_URL, str(self.repo_dir)]
        if self.quiet:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.check_call(cmd)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Expected dataset folder at: {self.dataset_dir}")

    # ---------- qrels ----------
    def _load_qrels(self) -> None:
        if self._qrels_all is not None:
            return

        q_all: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        q_asin: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        q_msm: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        asins: Set[str] = set()
        msm: Set[str] = set()

        with open(self.curated_qrels_path(), "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split()
                if len(parts) < 4:
                    continue
                topic_id = str(parts[0])
                doc_id = str(parts[2])
                try:
                    rel = int(parts[3])
                except Exception:
                    continue

                q_all[topic_id].append((doc_id, rel))
                if _is_asin(doc_id):
                    q_asin[topic_id].append((doc_id, rel))
                    asins.add(doc_id)
                elif _is_msmarco(doc_id):
                    q_msm[topic_id].append((doc_id, rel))
                    msm.add(doc_id)

        self._qrels_all = dict(q_all)
        self._qrels_asin = dict(q_asin)
        self._qrels_msmarco = dict(q_msm)
        self._asin_doc_ids = asins
        self._msmarco_doc_ids = msm

    def qrels_all(self) -> Dict[str, List[Tuple[str, int]]]:
        self._load_qrels()
        return dict(self._qrels_all or {})

    def qrels_asin(self) -> Dict[str, List[Tuple[str, int]]]:
        self._load_qrels()
        return dict(self._qrels_asin or {})

    def qrels_msmarco(self) -> Dict[str, List[Tuple[str, int]]]:
        self._load_qrels()
        return dict(self._qrels_msmarco or {})

    def asin_doc_ids(self) -> Set[str]:
        self._load_qrels()
        return set(self._asin_doc_ids or set())

    def msmarco_doc_ids(self) -> Set[str]:
        self._load_qrels()
        return set(self._msmarco_doc_ids or set())

    # ---------- conversations ----------
    def iter_conversations(self, partition: str = "raw") -> Iterator[CoSRecConversation]:
        for conv_id, conv_text in _read_jsonl_single_key(self.conversations_path(partition)):
            turns = _parse_conversation_text(conv_text)
            yield CoSRecConversation(self, partition, conv_id, turns)

    def _index_curated_conversations(self) -> Dict[str, CoSRecConversation]:
        if self._curated_conversations_index is not None:
            return self._curated_conversations_index
        idx: Dict[str, CoSRecConversation] = {}
        for c in self.iter_conversations("curated"):
            idx[c.conversation_id] = c
        self._curated_conversations_index = idx
        return idx

    # ---------- curated intents ----------
    def _load_curated_intent_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Map base_intent_id -> {type, query_variants, conversation_id, utterance_idx}
        """
        if self._curated_intent_map is not None:
            return self._curated_intent_map

        m: Dict[str, Dict[str, Any]] = {}
        for conv_id, blocks in _read_jsonl_single_key(self.curated_intents_path()):
            if not isinstance(blocks, list):
                continue
            for blk in blocks:
                if not isinstance(blk, dict):
                    continue
                uidx = int(blk.get("utterance", -1))
                intents = blk.get("intents") or []
                if not isinstance(intents, list):
                    continue
                for it in intents:
                    if not isinstance(it, dict):
                        continue
                    iid = str(it.get("id", ""))
                    typ = str(it.get("type", ""))
                    qv = it.get("query_variants") or []
                    if not isinstance(qv, list):
                        qv = []
                    if iid:
                        m[iid] = {
                            "type": typ,
                            "query_variants": [str(x) for x in qv],
                            "conversation_id": str(conv_id),
                            "utterance_idx": uidx,
                        }

        self._curated_intent_map = m
        return m

    # ---------- curated episodes (ASIN ONLY, ITEM REQUIRED) ----------
    def iter_curated_item_episodes(
        self,
        intent_type: str = "recommendation",
        min_relevance: int = 1,
        emotion=None,
        bias=None,
    ) -> Iterator[CoSRecCuratedEpisode]:
        """
        Curated-only, item-grounded episodes for recommendation intents.

        - Uses only qrels entries whose doc_id is an ASIN (10-char alnum).
        - Requires topic_id contains '#' (personalized rec topic)
        - Requires at least one ASIN with rel >= min_relevance
        - Requires next user message (so emotion is defined)
        """
        intent_map = self._load_curated_intent_map()
        qrels_asin = self.qrels_asin()
        conv_index = self._index_curated_conversations()

        for topic_id, judgs in qrels_asin.items():
            if "#" not in topic_id:
                continue  # recommendation topics include '#'
            base_intent_id, user_index_str = topic_id.split("#", 1)
            try:
                user_index = int(user_index_str)
            except Exception:
                continue

            meta = intent_map.get(base_intent_id)
            if not meta:
                continue
            if str(meta.get("type", "")) != intent_type:
                continue

            good = [(d, r) for (d, r) in judgs if int(r) >= int(min_relevance)]
            if not good:
                continue  # no items

            conv_id, utterance_idx, _ = _split_intent_id(base_intent_id)
            conv_id = str(meta.get("conversation_id", conv_id))
            utterance_idx = int(meta.get("utterance_idx", utterance_idx))

            conv = conv_index.get(conv_id)
            if conv is None:
                continue

            user_turn_idx = _user_utterance_index_to_turn_index(conv.turns, utterance_idx)
            if user_turn_idx is None:
                continue

            system_turn_idx = user_turn_idx + 1
            if system_turn_idx >= len(conv.turns) or conv.turns[system_turn_idx].get("speaker") != "S":
                continue

            next_user_turn_idx = _next_turn_index(conv.turns, system_turn_idx + 1, "U")
            if next_user_turn_idx is None:
                continue

            yield CoSRecCuratedEpisode(
                dataset=self,
                conversation_id=conv_id,
                topic_id=topic_id,
                base_intent_id=base_intent_id,
                user_index=user_index,
                utterance_idx=utterance_idx,
                intent_type=str(meta.get("type", "")),
                query_variants=list(meta.get("query_variants", [])),
                qrels=good,
                user_turn_idx=user_turn_idx,
                system_turn_idx=system_turn_idx,
                next_user_turn_idx=next_user_turn_idx,
                user_text=conv.turns[user_turn_idx].get("text", ""),
                system_text=conv.turns[system_turn_idx].get("text", ""),
                next_user_text=conv.turns[next_user_turn_idx].get("text", ""),
                emotion_cache=emotion,
                bias_cache=bias,
            )
