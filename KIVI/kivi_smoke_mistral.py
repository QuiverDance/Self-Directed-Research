# kivi_smoke_mistral.py
# Goal: Verify KIVI loads and runs a short generation on Mistral-7B (2-bit KV path).
# Notes:
# - Auto-detects model_type and picks the right KIVI class.
# - Prints KIVI config (k_bits, v_bits, group_size, residual_length).
# - Keep prompt short to avoid OOM on single GPU.

import os
import time
import torch
from transformers import AutoConfig, AutoTokenizer

# --- user settings ---
MODEL_DIR = os.path.expanduser("~/models/Mistral-7B-Instruct-v0.2")
K_BITS = 2              # KIVI supports 2/4-bit; 2-bit for smoke test
V_BITS = 2
GROUP_SIZE = 32         # per-group elements (typical: 32)
RESIDUAL_LENGTH = 32    # keep last N tokens in FP16/BF16

def pick_kivi_class(cfg):
    mt = getattr(cfg, "model_type", None)
    if mt == "mistral":
        from models.mistral_kivi import MistralForCausalLM_KIVI as KIVIModel
        return KIVIModel, "mistral"
    elif mt == "llama":
        from models.llama_kivi import LlamaForCausalLM_KIVI as KIVIModel
        return KIVIModel, "llama"
    else:
        raise RuntimeError(f"Unsupported model_type for KIVI: {mt}")

def main():
    print(f"[KIVI-SMOKE] loading config from: {MODEL_DIR}")
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    KIVIModel, fam = pick_kivi_class(cfg)

    # --- inject KIVI knobs into config (per README example style) ---
    # (KIVI uses config.k_bits / v_bits / group_size / residual_length)
    cfg.k_bits = K_BITS
    cfg.v_bits = V_BITS
    cfg.group_size = GROUP_SIZE
    cfg.residual_length = RESIDUAL_LENGTH

    # dtype auto-pick
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print(f"[KIVI-SMOKE] family={fam}  dtype={dtype}  device_map=auto")
    print(f"[KIVI-SMOKE] k_bits={cfg.k_bits}, v_bits={cfg.v_bits}, "
          f"group_size={cfg.group_size}, residual_length={cfg.residual_length}")

    # --- load KIVI-wrapped model ---
    model = KIVIModel.from_pretrained(
        pretrained_model_name_or_path=MODEL_DIR,
        config=cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompt = "You are a helpful assistant.\n\nQ: 서울은 어느 나라의 수도인가?\nA:"
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )
    t1 = time.time()

    text = tok.decode(out[0], skip_special_tokens=True)
    print("\n=== OUTPUT ===")
    print(text)
    print("================\n")
    print(f"[KIVI-SMOKE] elapsed: {t1 - t0:.2f}s")

if __name__ == "__main__":
    main()

