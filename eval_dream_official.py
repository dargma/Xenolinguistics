"""
Dream-7B Official SFT — Evaluation
공식 diffusion_generate 사용, free_length 모드
"""

import json
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

DATA_TAG = "1k"
MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
ADAPTER_PATH = "outputs/dream_official/adapter_best"

test_data = [json.loads(l) for l in open(f"data/test_{DATA_TAG}.jsonl")]

# Load model
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
base = AutoModel.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

# DreamGenerationConfig
from transformers import GenerationConfig
try:
    from transformers_modules import DreamGenerationConfig
except:
    pass

# Use model's diffusion_generate
MASK_ID = tokenizer.mask_token_id
print(f"mask_token_id: {MASK_ID}")
print(f"Evaluating {len(test_data)} samples...")

predictions = []
references = []
examples = []

for i, ex in enumerate(test_data[:100]):
    # Build prompt with chat template
    messages = [{"role": "user", "content": ex["instruction"]}]
    prompt_str = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
    input_ids = prompt_ids["input_ids"].to(model.device)
    attention_mask = prompt_ids["attention_mask"].to(model.device)

    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        try:
            output = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                output_history=False,
                steps=128,
                temperature=0.2,
                top_p=0.95,
                alg="entropy",
                alg_temp=0.0,
            )
            if hasattr(output, "sequences"):
                gen_ids = output.sequences[0]
            else:
                gen_ids = output[0]
        except Exception as e:
            if i == 0:
                print(f"diffusion_generate failed: {e}")
                print("Falling back to manual denoising...")

            # Fallback: manual iterative denoising
            n_masks = 64
            full_ids = torch.cat([
                input_ids,
                torch.full((1, n_masks), MASK_ID, dtype=torch.long, device=model.device)
            ], dim=1)
            full_attn = torch.ones_like(full_ids)

            # 4D bidirectional attention
            attn_4d = torch.logical_and(
                full_attn.unsqueeze(1).unsqueeze(-2),
                full_attn.unsqueeze(1).unsqueeze(-1),
            )

            for step in range(20):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(
                        input_ids=full_ids,
                        attention_mask=attn_4d,
                        use_cache=False,
                    ).logits

                # Dream logit shift: cat([logits[:,0:1], logits[:,:-1]])
                shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
                pred_ids = shift_logits.argmax(dim=-1)

                # Only update masked positions
                mask_pos = (full_ids == MASK_ID)
                full_ids = torch.where(mask_pos, pred_ids, full_ids)

            gen_ids = full_ids[0]

    # Decode only generated part
    gen_text = tokenizer.decode(
        gen_ids[prompt_len:], skip_special_tokens=True
    ).strip()

    # Clean up: remove anything after <|im_end|> or similar
    for stop in ["<|im_end|>", "<|endoftext|>", "\n\n\n"]:
        if stop in gen_text:
            gen_text = gen_text[:gen_text.index(stop)].strip()

    predictions.append(gen_text)
    references.append(ex["fi"])

    if i < 10:
        examples.append({
            "en": ex["en"],
            "fi_ref": ex["fi"],
            "fi_pred": gen_text,
        })
        print(f"[{i}] EN:   {ex['en']}")
        print(f"    REF:  {ex['fi']}")
        print(f"    PRED: {gen_text}")
        print()

# Metrics
chrf = sacrebleu.corpus_chrf(predictions, [references])
bleu = sacrebleu.corpus_bleu(predictions, [references])

print(f"\n{'='*50}")
print(f"[Dream Official SFT] chrF={chrf.score:.2f}, BLEU={bleu.score:.2f}")
print(f"{'='*50}")

result = {
    "model": "Dream-7B + LoRA (official SFT port)",
    "adapter": ADAPTER_PATH,
    "data_tag": DATA_TAG,
    "n_eval": len(predictions),
    "chrF": round(chrf.score, 2),
    "BLEU": round(bleu.score, 2),
    "examples": examples,
}
json.dump(result, open("outputs/dream_official/eval_results.json", "w"),
          indent=2, ensure_ascii=False)
print(f"Results saved → outputs/dream_official/eval_results.json")
