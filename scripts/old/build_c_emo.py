import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from dataset import CoSRecDataset
from emotion.go_emotions_cosrec import GoEmotionsCoSRec

CACHE = Path("./cache")
ds = CoSRecDataset(cache_root=CACHE)
emo = GoEmotionsCoSRec(cache_root=CACHE, device="mps", top_k=5)

# build cached episode emotions (only item-grounded curated recommendation episodes)
emo.build(ds, intent_type="recommendation", min_relevance=1, progress_path=CACHE/"cosrec_emotion_progress.json")

# iterate episodes (each has items via qrels, and .emotion available)
for ep in ds.iter_rec_episodes(min_relevance=1, emotion=emo):
    if ep.intent_type != "recommendation":
        continue
    print(ep.topic_id, ep.emotion["emotion"][:2])
    break
