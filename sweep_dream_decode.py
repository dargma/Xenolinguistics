"""
Dream 10k adapter — decoding parameter sweep
기존 학습된 adapter를 재사용하고 디코딩 설정만 바꿔서 평가
"""
import json, torch, sacrebleu, time, sys
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
ADAPTER_PATH = "outputs/dream_official_10k/adapter_best"
TEST_FILE = "data/test_1k.jsonl"
N_EVAL = 100

# Sweep configurations
CONFIGS = [
    # baseline (현재)
    {"name": "baseline",        "steps": 128, "temp": 0.2, "top_p": 0.95, "alg": "entropy"},
    # steps 변경
    {"name": "steps256",        "steps": 256, "temp": 0.2, "top_p": 0.95, "alg": "entropy"},
    {"name": "steps64",         "steps": 64,  "temp": 0.2, "top_p": 0.95, "alg": "entropy"},
    # temperature 변경
    {"name": "temp04",          "steps": 128, "temp": 0.4, "top_p": 0.95, "alg": "entropy"},
    {"name": "temp06",          "steps": 128, "temp": 0.6, "top_p": 0.95, "alg": "entropy"},
    {"name": "temp02_greedy",   "steps": 128, "temp": 0.0, "top_p": 1.0,  "alg": "entropy"},
    # steps + temp 조합
    {"name": "steps256_temp04", "steps": 256, "temp": 0.4, "top_p": 0.95, "alg": "entropy"},
    # alg 변경
    {"name": "maskgit_plus",    "steps": 128, "temp": 0.2, "top_p": 0.95, "alg": "maskgit_plus"},
    {"name": "origin_alg",      "steps": 128, "temp": 0.2, "top_p": 0.95, "alg": "origin"},
    # max_new_tokens 변경
    {"name": "tokens128",       "steps": 128, "temp": 0.2, "top_p": 0.95, "alg": "entropy", "max_new_tokens": 128},
]

test_data = [json.loads(l) for l in open(TEST_FILE)][:N_EVAL]

# Load model once
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
base = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
MASK_ID = tokenizer.mask_token_id
print(f"Model loaded. mask_id={MASK_ID}\n")

results = []

for cfg in CONFIGS:
    name = cfg["name"]
    max_new = cfg.get("max_new_tokens", 64)
    print(f"=== {name} | steps={cfg['steps']} temp={cfg['temp']} alg={cfg['alg']} max_new={max_new} ===")

    predictions, references = [], []
    examples = []
    start = time.time()

    for i, ex in enumerate(test_data):
        messages = [{"role": "user", "content": ex["instruction"]}]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        input_ids = prompt_ids["input_ids"].to(model.device)
        attention_mask = prompt_ids["attention_mask"].to(model.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            try:
                gen_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new,
                    output_history=False,
                    steps=cfg["steps"],
                    temperature=cfg["temp"] if cfg["temp"] > 0 else 0.0001,
                    top_p=cfg["top_p"],
                    alg=cfg["alg"],
                    alg_temp=0.0,
                )
                output = model.diffusion_generate(**gen_kwargs)
                gen_ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
            except Exception as e:
                if i == 0:
                    print(f"  diffusion_generate error: {e}")
                # fallback
                n_masks = max_new
                full_ids = torch.cat([input_ids, torch.full((1, n_masks), MASK_ID, dtype=torch.long, device=model.device)], dim=1)
                full_attn = torch.ones_like(full_ids)
                attn_4d = torch.logical_and(full_attn.unsqueeze(1).unsqueeze(-2), full_attn.unsqueeze(1).unsqueeze(-1))
                for step in range(cfg["steps"] // 4):  # scale fallback steps
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(input_ids=full_ids, attention_mask=attn_4d, use_cache=False).logits
                    shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
                    pred_ids = shift_logits.argmax(dim=-1)
                    mask_pos = (full_ids == MASK_ID)
                    full_ids = torch.where(mask_pos, pred_ids, full_ids)
                gen_ids = full_ids[0]

        gen_text = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True).strip()
        for stop in ["<|im_end|>", "<|endoftext|>", "\n\n\n"]:
            if stop in gen_text:
                gen_text = gen_text[:gen_text.index(stop)].strip()

        predictions.append(gen_text)
        references.append(ex["fi"])

        if i < 5:
            examples.append({"en": ex["en"], "ref": ex["fi"], "pred": gen_text})

    elapsed = time.time() - start
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    result = {
        "name": name,
        "steps": cfg["steps"],
        "temp": cfg["temp"],
        "alg": cfg["alg"],
        "max_new_tokens": max_new,
        "chrF": round(chrf.score, 2),
        "BLEU": round(bleu.score, 2),
        "time_sec": round(elapsed, 1),
        "examples": examples,
    }
    results.append(result)
    print(f"  chrF={chrf.score:.2f}  BLEU={bleu.score:.2f}  time={elapsed:.0f}s")
    for ex in examples[:3]:
        print(f"    {ex['en'][:40]:40} → {ex['pred'][:40]}")
    print()

# Summary
print("\n" + "=" * 70)
print(f"{'Config':<20} {'Steps':>5} {'Temp':>5} {'Alg':<15} {'chrF':>6} {'BLEU':>6} {'Time':>6}")
print("-" * 70)
for r in sorted(results, key=lambda x: -x["chrF"]):
    print(f"{r['name']:<20} {r['steps']:>5} {r['temp']:>5.1f} {r['alg']:<15} {r['chrF']:>6.2f} {r['BLEU']:>6.2f} {r['time_sec']:>5.0f}s")
print("=" * 70)

json.dump(results, open("outputs/dream_official_10k/decode_sweep.json", "w"), indent=2, ensure_ascii=False)
print("\nSaved → outputs/dream_official_10k/decode_sweep.json")
