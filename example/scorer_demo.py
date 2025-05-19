from xtest_bert_score import score
from xtest_bert_score.scorer import BERTScorer

bert_scorer = BERTScorer(
    model_type="/data/llm-models/bert-base-uncased"
)
cands = ["The cat sat on the mat."]
refs = ["A cat was sitting on the mat."]
(P, R, F), hashname = bert_scorer.score(cands, refs, return_hash=True)
print(
    f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}"
)
