"""
Dream 10k — training hyperparameter sweep
디코딩 sweep 이후 실행. 학습 설정을 바꿔서 재학습 + 평가.
"""
import subprocess, json, time, os

# Sweep configurations (train_dream_official.py의 CLI args)
CONFIGS = [
    {
        "name": "lr5e5",
        "args": "--lr 5e-5 --epochs 3",
        "note": "LR 5x 증가",
    },
    {
        "name": "lr1e4",
        "args": "--lr 1e-4 --epochs 3",
        "note": "LR 10x 증가 (Qwen과 동일)",
    },
    {
        "name": "token_reweight",
        "args": "--lr 1e-5 --epochs 3 --token_reweighting",
        "note": "Token reweighting 활성화",
    },
    {
        "name": "lr5e5_tokenrw",
        "args": "--lr 5e-5 --epochs 3 --token_reweighting",
        "note": "LR 5x + token reweighting",
    },
    {
        "name": "epoch5",
        "args": "--lr 1e-5 --epochs 5",
        "note": "Epoch 5로 증가",
    },
    {
        "name": "lr5e5_epoch5",
        "args": "--lr 5e-5 --epochs 5",
        "note": "LR 5x + epoch 5",
    },
]

TRAIN_FILE = "data/train_10k.jsonl"
VAL_FILE = "data/val_10k.jsonl"
TEST_FILE = "data/test_1k.jsonl"
BASE_CMD = "python3 train_dream_official.py"
EVAL_CMD = "python3 eval_dream_official.py"

results = []

for cfg in CONFIGS:
    name = cfg["name"]
    out_dir = f"outputs/dream_sweep/{name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[TRAIN] {name}: {cfg['note']}")
    print(f"{'='*60}")

    # Train
    train_cmd = (
        f"{BASE_CMD} "
        f"--train_file {TRAIN_FILE} --val_file {VAL_FILE} "
        f"--output_dir {out_dir} --log_every 100 "
        f"{cfg['args']}"
    )
    print(f"CMD: {train_cmd}")
    t0 = time.time()
    ret = subprocess.run(train_cmd, shell=True, capture_output=True, text=True)
    train_time = time.time() - t0

    if ret.returncode != 0:
        print(f"TRAIN FAILED: {ret.stderr[-500:]}")
        results.append({"name": name, "status": "train_fail", "error": ret.stderr[-200:]})
        continue

    print(f"Train done in {train_time/60:.1f}min")
    # Get training meta
    meta_path = f"{out_dir}/training_meta.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print(f"  best_val_loss={meta.get('best_val_loss', 'N/A')}")

    # Eval using the best adapter
    print(f"\n[EVAL] {name}")
    adapter_path = f"{out_dir}/adapter_best"
    if not os.path.exists(adapter_path):
        adapter_path = f"{out_dir}/adapter_final"

    # Inline eval to avoid model reload
    eval_code = f"""
import json, torch, sacrebleu
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

test_data = [json.loads(l) for l in open("{TEST_FILE}")][:100]
tokenizer = AutoTokenizer.from_pretrained("{adapter_path}", trust_remote_code=True)
base = AutoModel.from_pretrained("Dream-org/Dream-v0-Instruct-7B",
    torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "{adapter_path}")
model.eval()

predictions, references, examples = [], [], []
for i, ex in enumerate(test_data):
    messages = [{{"role": "user", "content": ex["instruction"]}}]
    prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
    input_ids = prompt_ids["input_ids"].to(model.device)
    attn = prompt_ids["attention_mask"].to(model.device)
    plen = input_ids.shape[1]
    with torch.no_grad():
        out = model.diffusion_generate(input_ids, attention_mask=attn,
            max_new_tokens=64, steps=128, temperature=0.2, top_p=0.95,
            alg="entropy", alg_temp=0.0, output_history=False)
        gids = out.sequences[0] if hasattr(out, "sequences") else out[0]
    gen = tokenizer.decode(gids[plen:], skip_special_tokens=True).strip()
    for s in ["<|im_end|>", "<|endoftext|>", "\\n\\n\\n"]:
        if s in gen: gen = gen[:gen.index(s)].strip()
    predictions.append(gen)
    references.append(ex["fi"])
    if i < 5:
        examples.append({{"en": ex["en"], "ref": ex["fi"], "pred": gen}})

chrf = sacrebleu.corpus_chrf(predictions, [references])
bleu = sacrebleu.corpus_bleu(predictions, [references])
result = {{"chrF": round(chrf.score,2), "BLEU": round(bleu.score,2), "examples": examples}}
json.dump(result, open("{out_dir}/eval_results.json", "w"), indent=2, ensure_ascii=False)
print(f"chrF={{chrf.score:.2f}} BLEU={{bleu.score:.2f}}")
"""
    eval_ret = subprocess.run(["python3", "-c", eval_code], capture_output=True, text=True)
    eval_time = time.time() - t0 - train_time

    if eval_ret.returncode != 0:
        print(f"EVAL FAILED: {eval_ret.stderr[-500:]}")
        results.append({"name": name, "status": "eval_fail", "train_time_min": round(train_time/60,1)})
        continue

    # Parse result
    eval_out = eval_ret.stdout.strip().split("\n")[-1]
    print(f"  {eval_out}")

    eval_data = json.load(open(f"{out_dir}/eval_results.json"))
    results.append({
        "name": name,
        "note": cfg["note"],
        "status": "done",
        "train_time_min": round(train_time/60, 1),
        "chrF": eval_data["chrF"],
        "BLEU": eval_data["BLEU"],
    })

# Summary
print("\n" + "=" * 70)
print(f"{'Config':<20} {'Note':<25} {'chrF':>6} {'BLEU':>6} {'Time':>7}")
print("-" * 70)
for r in sorted(results, key=lambda x: -x.get("chrF", 0)):
    print(f"{r['name']:<20} {r.get('note',''):<25} {r.get('chrF','FAIL'):>6} {r.get('BLEU',''):>6} {r.get('train_time_min',''):>6}m")
print("=" * 70)

# Add baseline for comparison
results.append({"name": "baseline_1e5", "note": "현재 기본값", "chrF": 18.56, "BLEU": 9.58, "train_time_min": 29.2})

json.dump(results, open("outputs/dream_sweep/train_sweep_results.json", "w"), indent=2, ensure_ascii=False)
print("\nSaved → outputs/dream_sweep/train_sweep_results.json")
