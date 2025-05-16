from xtest_bert_score import score

with open("hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("refs.txt") as f:
    refs = [line.strip() for line in f]

(P, R, F), hashname = score(cands, refs,model_type="/data/llm-models/bert-base-uncased", lang="en", return_hash=True)
print(
    f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}"
)
