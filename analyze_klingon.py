"""WHY-analysis for the 2026-05-29 report: regenerate all 300 Klingon test preds
(free mode, both directions × both archs), then compute
 (1) failure-mode taxonomy (empty / degenerate-repetition / length ratio)
 (2) per-sentence paired chrF (AR vs Diffusion) + correlation with input length.
Outputs JSON to outputs/analysis_klingon.json.
"""
import os, json, types, torch, sacrebleu, statistics
os.environ.setdefault("HF_HOME", "/content/local_fast/hf_cache")
import sys
REPO = "/content/drive/MyDrive/Xenolinguistics"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/eval")
from eval_khalani_icl import clean_pred
from transformers import AutoTokenizer, AutoModelForCausalLM

LF = "/content/local_fast/outputs"
RUNS = [
    ("en2tlh", "ar",        f"{LF}/klingon_qwen_fullft/final",        "/content/local_fast/data/klingon/test.jsonl",       "klingon", "Klingon"),
    ("en2tlh", "diffusion", f"{LF}/klingon_fastdllm_fullft/final",     "/content/local_fast/data/klingon/test.jsonl",       "klingon", "Klingon"),
    ("tlh2en", "ar",        f"{LF}/klingon_qwen_tlh2en_fullft/final",  "/content/local_fast/data/klingon/test_tlh2en.jsonl","en",      "English"),
    ("tlh2en", "diffusion", f"{LF}/klingon_fastdllm_tlh2en_fullft/final","/content/local_fast/data/klingon/test_tlh2en.jsonl","en",    "English"),
]
N = 300; MNT = 64

def is_degenerate(p):
    toks = p.split()
    if len(toks) < 6:
        return False
    ttr = len(set(toks)) / len(toks)
    maxfreq = max(toks.count(w) for w in set(toks)) / len(toks)
    return ttr < 0.5 or maxfreq > 0.35

def gen_all(model_type, path, testf, tgt, lang):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if model_type == "ar":
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    else:
        import fastdllm_generation as gf
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
        model.mdm_sample = types.MethodType(gf.Fast_dLLM_QwenForCausalLM.batch_sample, model)
        mask_id = tok.encode("|<MASK>|", add_special_tokens=False)[0]
    rows = [json.loads(l) for l in open(testf)][:N]
    preds, refs, srclens = [], [], []
    for ex in rows:
        msgs = [{"role": "user", "content": ex["instruction"]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if model_type == "ar":
            inp = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MNT, do_sample=False, pad_token_id=tok.eos_token_id)
            raw = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            sl = torch.tensor([ids.shape[1]], device=model.device)
            out = model.mdm_sample(ids, tokenizer=tok, block_size=32, small_block_size=32, max_new_tokens=MNT,
                                   mask_id=mask_id, min_len=ids.shape[1], seq_len=sl, use_block_cache=True, threshold=0.9)
            raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        preds.append(clean_pred(raw, lang)); refs.append(ex[tgt])
        srclens.append(len(ex["instruction"].split()))
    del model; torch.cuda.empty_cache()
    return preds, refs, srclens

results = {}
store = {}
for direction, mt, path, testf, tgt, lang in RUNS:
    print(f"=== {direction} {mt} ===", flush=True)
    preds, refs, srclens = gen_all(mt, path, testf, tgt, lang)
    schrf = [sacrebleu.sentence_chrf(p, [r]).score for p, r in zip(preds, refs)]
    empty = sum(1 for p in preds if not p.strip())
    degen = sum(1 for p in preds if is_degenerate(p))
    plen = [len(p.split()) for p in preds]; rlen = [len(r.split()) for r in refs]
    results[f"{direction}_{mt}"] = {
        "n": len(preds), "empty": empty, "degenerate": degen,
        "degenerate_pct": round(100*degen/len(preds), 1),
        "mean_sent_chrF": round(statistics.mean(schrf), 2),
        "mean_pred_len": round(statistics.mean(plen), 1), "mean_ref_len": round(statistics.mean(rlen), 1),
        "len_ratio": round(statistics.mean(plen)/statistics.mean(rlen), 2),
    }
    store[f"{direction}_{mt}"] = {"schrf": schrf, "srclens": srclens, "plen": plen}
    print(" ", results[f"{direction}_{mt}"], flush=True)

# paired AR vs Diff per direction
for d in ["en2tlh", "tlh2en"]:
    a, df = store[f"{d}_ar"], store[f"{d}_diffusion"]
    diff = [x - y for x, y in zip(df["schrf"], a["schrf"])]  # Diff - AR
    ar_win = sum(1 for x in diff if x < -1); df_win = sum(1 for x in diff if x > 1); tie = len(diff) - ar_win - df_win
    # correlation of (Diff-AR) with input length
    import math
    sl = a["srclens"]; mx, my = statistics.mean(sl), statistics.mean(diff)
    cov = sum((s-mx)*(dd-my) for s, dd in zip(sl, diff)); vx = sum((s-mx)**2 for s in sl); vy = sum((dd-my)**2 for dd in diff)
    r = cov/math.sqrt(vx*vy) if vx*vy else 0
    results[f"paired_{d}"] = {"AR_win": ar_win, "Diff_win": df_win, "tie": tie,
                              "mean_chrF_gap_DiffminusAR": round(statistics.mean(diff), 2),
                              "corr_gap_vs_srclen": round(r, 3)}
    print(f"paired {d}:", results[f"paired_{d}"], flush=True)

json.dump(results, open("outputs/analysis_klingon.json", "w"), indent=2, ensure_ascii=False)
print("SAVED outputs/analysis_klingon.json")
print("ANALYSIS_DONE")
