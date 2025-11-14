# run_benchmark.py
#
# A self-contained orchestrator script to run a full suite of standard LLM benchmarks
# using the KIVI model framework. This script includes all necessary logic for data loading,
# prompting, and scoring for all specified benchmarks, and collects performance metrics
# (latency, throughput, memory).

import os
import sys
import json
import time
import argparse
import gc
import random
import re
import string
import subprocess
import math
import tempfile
import re as _re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

try:
    import numpy as np
except Exception:
    np = None

# --- KIVI Model Imports ---
try:
    from models.mistral_kivi import MistralForCausalLM_KIVI
    from models.llama_kivi import LlamaForCausalLM_KIVI
except ImportError:
    print("FATAL: Could not import KIVI models. Ensure 'models' dir is in PYTHONPATH.")
    sys.exit(1)

try:
    from utils.process_args import load_quant_config
except ImportError:
    print("Warning: Could not import quant config loader. Falling back to 'kvtuner_only' mode.")
    def load_quant_config(path):
        return "kvtuner_only", None, None, None

# ==========================================================================================
# Global Benchmark Configuration
# ==========================================================================================
SEED = 2025
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
BENCH_ALL_CONFIG = {
    "mmlu":        {"num_samples": 14042, "kshot": 1, "batch_size": 16, "max_new_tokens": 1},
    "gsm8k":       {"num_samples": 1319, "kshot": 1, "batch_size": 16, "max_new_tokens": 256},
    "humaneval":   {"num_samples": 164, "batch_size": 16, "max_new_tokens": 512},
    "line_retrieval": {
        "num_samples": 200, "batch_size": 2, "max_new_tokens": 64,
        "lr_num_lines": 1000, "lr_min_words": 5, "lr_max_words": 9, "lr_target_mode": "random",
        "lr_max_prompt_tokens": 10000,  # hard cap for prompt tokens (problem text) regardless of longer model context
    },
    "longbench_qasper":  {"num_samples": 200, "batch_size": 1, "max_new_tokens": 32},
    "longbench_hotpotqa":{"num_samples": 200, "batch_size": 1, "max_new_tokens": 32},
    "longbench_2wikimqa":{"num_samples": 200, "batch_size": 1, "max_new_tokens": 32},
    "longbench_musique": {"num_samples": 200, "batch_size": 1, "max_new_tokens": 32},
    "needle": {
        "num_samples": 100, "batch_size": 1, "max_new_tokens": 8,
        "nh_target_tokens": 12000, "nh_depth_mode": "random", "nh_vocab_mode": "random", "nh_depth": 0.5,
    },
}

# ==========================================================================================
# Performance & Utility Helpers
# ==========================================================================================
# clear CUDA memory
def _cleanup_cuda():
    import gc, torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def _is_cuda(device) -> bool:
    return torch.cuda.is_available() and getattr(device, "type", "") == "cuda"

def fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]; k=1024.0
    if n <= 0: return "0 B"
    i = int(math.floor(math.log(n, k))) if n > 0 else 0
    return f"{n / (k**i):.2f} {units[i]}"

def _infer_input_device(model):
    # Prefer the device of embedding weights if present; else any CUDA param; else first param device; else CPU
    try:
        for name, p in model.named_parameters(recurse=True):
            if "embed" in name and p.device.type == "cuda":
                return p.device
        for p in model.parameters():
            if p.device.type == "cuda":
                return p.device
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

def warmup_and_measure_base(model, tokenizer):
    """
    Safe warmup for both sharded (device_map='auto') and single-device setups.
    - Keep inputs on CPU; let HF dispatch handle device placement.
    - Call CUDA sync/empty_cache only if CUDA is available.
    """
    tiny_input = None
    try:
        if torch.cuda.is_available():
            # Do not pass a specific device; sync all current CUDA work safely.
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        with torch.inference_mode():
            tiny_input = tokenizer("warmup", return_tensors="pt")
            # Move to a safe device (embedding device if available)
            dev = _infer_input_device(model)
            if hasattr(tiny_input, "to"):
                tiny_input = tiny_input.to(dev)
            else:
                tiny_input = {k: v.to(dev) for k, v in tiny_input.items()}
            # Use pad_token_id if available; otherwise eos
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            _ = model.generate(**tiny_input, max_new_tokens=2, pad_token_id=pad_id)
        # Report current allocated CUDA memory if CUDA exists; else 0
        return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    except Exception as e:
        print(f"[WARN] warmup skipped (safe): {repr(e)}")
        return 0
    finally:
        # Make sure cleanup is robust even if tiny_input was never assigned
        if tiny_input is not None:
            del tiny_input
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# ==========================================================================================
# KIVI Model Loading Utility
# ==========================================================================================
def load_kivi_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any]:
    model_dir = os.path.expanduser(args.model)
    dtype = torch.float16
    device_map = "auto"

    cfg = AutoConfig.from_pretrained(model_dir)
    mt = getattr(cfg, "model_type", None)
    # Unified quant config loader: mode + KVTuner + ZipCache
    mode, layer_quant_map, zipcache_cfg, group_choices = load_quant_config(
        getattr(args, "quant_config_path", None)
    )
    is_baseline_mode = (mode == "baseline")

    run_label = ""
    if is_baseline_mode:
        run_label = f"[MODE] Baseline (FP16): Loading original {mt.capitalize()}ForCausalLM."
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=cfg,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            attn_implementation="flash_attention_2",
        ).eval()
    else:
        KIVIModel = {"mistral": MistralForCausalLM_KIVI, "llama": LlamaForCausalLM_KIVI}.get(mt)
        if KIVIModel is None:
            raise RuntimeError(f"Unsupported model_type for KIVI: {mt}")
        
        cfg.k_bits, cfg.v_bits = int(args.k_bits), int(args.v_bits)
        cfg.group_size, cfg.residual_length = int(args.group_size), int(args.residual_length)

        use_flash_str = str(getattr(args, "use_flash", "true")).strip().lower()
        cfg.use_flash = use_flash_str in ("1", "true", "yes", "y")
        # if cfg.use_flash: cfg._flash_attn_2_enabled = True

        # Expose quant mode and configs to the model config for later use
        cfg.quant_mode = mode
        cfg.layer_quant_map = layer_quant_map
        cfg.zipcache_config = zipcache_cfg

        if mode == "kvtuner_only":
            if isinstance(layer_quant_map, dict):
                run_label = "[MODE] Quantized (KIVI) - KVTuner Layer-wise"
                if group_choices:
                    run_label += f"\n[KVTuner] Group choices: {group_choices}"
            else:
                run_label = (
                    "[MODE] Quantized (KIVI) - Uniform\n"
                    f"[KIVI] Uniform Config: k_bits={cfg.k_bits}, v_bits={cfg.v_bits}"
                )

        elif mode in ("zipcache_only", "hybrid"):
            if zipcache_cfg is None:
                raise ValueError(
                    "[MODE] ZipCache/hybrid mode requires a 'zipcache' section in the quant config."
                )

            # ZipCache is only implemented for the FlashAttention path.
            if not cfg.use_flash:
                print("[WARN] ZipCache/hybrid mode requires FlashAttention; overriding cfg.use_flash=True.")
                cfg.use_flash = True

            if mode == "zipcache_only":
                run_label = "[MODE] Quantized (KIVI) - ZipCache only"
            else:
                run_label = "[MODE] Quantized (KIVI) - KVTuner + ZipCache (hybrid)"
                if isinstance(layer_quant_map, dict) and group_choices:
                    run_label += f"\n[KVTuner] Group choices: {group_choices}"
        else:
            raise ValueError(f"[MODE] Unknown quantization mode: {mode}")

        model = KIVIModel.from_pretrained(
            model_dir,
            config=cfg,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
            layer_quant_map=layer_quant_map,
        ).eval()
    
    print(run_label)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "left"
    
    print(f"[SETUP] Loaded model={os.path.basename(model_dir)} | dtype={dtype} | device_map={device_map}")
    return model, tokenizer

# ==========================================================================================
# Generic Generation Loop with Detailed Metrics
# ==========================================================================================
def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.LongTensor:
    """Temperature + top-k/top-p sampling on final-logits (per batch)."""
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if top_k and top_k > 0:
        top_k = min(top_k, probs.size(-1))
        vals, idx = torch.topk(probs, top_k, dim=-1)
        probs_filtered = torch.zeros_like(probs).scatter_(-1, idx, vals)
        probs = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)
    if top_p and top_p < 1.0:
        # nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        mask = cdf > top_p
        # keep at least one
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        # multinomial on truncated
        next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_ids = sorted_idx.gather(-1, next_in_sorted)
        return next_ids
    # default multinomial
    return torch.multinomial(probs, num_samples=1)

def generate_and_measure(model: Any, tokenizer: Any, prompts: List[str], batch_size: int, max_new_tokens: int, **gen_kwargs) -> Tuple[List[str], List[Dict]]:
    """Generation loop that delegates to HF generate() and collects timing/memory stats.

    This function uses model.generate() directly (no custom prefill path) so that behavior
    exactly matches the model's generation implementation (including KIVI paths).
    """
    device = next(model.parameters()).device
    all_outputs: List[str] = []
    per_batch_stats: List[Dict] = []

    stop_sequences: List[str] = gen_kwargs.pop("stop_sequences", []) or []
    do_sample: bool = bool(gen_kwargs.pop("do_sample", False))
    temperature: float = float(gen_kwargs.pop("temperature", 0.0) or 0.0)
    top_k: int = int(gen_kwargs.pop("top_k", 0) or 0)
    top_p: float = float(gen_kwargs.pop("top_p", 1.0) or 1.0)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        if not batch_prompts:
            continue

        # Respect model context budget (same as previous low-memory path)
        budget = _model_ctx_budget(model, tokenizer, safety_margin=64)

        # Tokenize on the correct device
        t_start = time.time()
        if _is_cuda(device):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=budget,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # >>> ADD: top-level debug around generate() <<<
        print(
            f"[DEBUG][GEN] batch={i // batch_size} "
            f"n_prompts={len(batch_prompts)} "
            f"input_ids={tuple(input_ids.shape)} "
            f"max_new_tokens={max_new_tokens}",
            flush=True,
        )

        # Single call to model.generate(): let the model handle cache, masks, positions, etc.
        with torch.inference_mode():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 0.0,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        # >>> ADD: confirm we returned from generate() <<<
        first_shape = tuple(gen_out[0].shape) if len(gen_out) > 0 else None
        print(
            f"[DEBUG][GEN] batch={i // batch_size} "
            f"generate() done, first_out={first_shape}",
            flush=True,
        )

        if _is_cuda(device):
            torch.cuda.synchronize(device)
            peak_mem = torch.cuda.max_memory_allocated(device)
        else:
            peak_mem = 0

        total_duration = time.time() - t_start

        # Decode only the newly generated tokens and apply optional stop sequences.
        batch_texts: List[str] = []
        num_new_tokens = 0
        for in_ids, out_ids in zip(input_ids, gen_out):
            # Only tokens after the prompt are counted as "new"
            prompt_len = in_ids.size(0)
            new_ids = out_ids[prompt_len:]
            num_new_tokens += int(new_ids.size(0))

            text = tokenizer.decode(new_ids, skip_special_tokens=True)

            if stop_sequences:
                cut = len(text)
                for ss in stop_sequences:
                    if not ss:
                        continue
                    idx = text.find(ss)
                    if idx != -1:
                        cut = min(cut, idx)
                text = text[:cut]

            batch_texts.append(text)

        all_outputs.extend(batch_texts)

        per_batch_stats.append({
            "batch_index": i // batch_size,
            "num_prompts": len(batch_prompts),
            # We can no longer split prefill vs decode; treat full generate() call as TTFT.
            "ttft_sec": round(total_duration, 4),
            "decode_duration_sec": 0.0,
            "new_tokens": num_new_tokens,
            "throughput_tok_per_sec": round(
                num_new_tokens / max(total_duration, 1e-6), 2
            ),
            "peak_inference_mem_bytes": peak_mem,
        })

        print(f"  ... Generated {len(all_outputs)} / {len(prompts)}", end="\r")

    print()
    return all_outputs, per_batch_stats

# ==========================================================================================
# Benchmark Implementations
# ==========================================================================================

# ================== MMLU =================
DEFAULT_MMLU_LOCAL_DIR = Path(os.path.expanduser("~/Self-Directed-Research/KIVI/datasets/mmlu"))
MMLU_SUBJECTS=["abstract_algebra","anatomy","astronomy","business_ethics","clinical_knowledge","college_biology","college_chemistry","college_computer_science","college_mathematics","college_medicine","college_physics","computer_security","conceptual_physics","econometrics","electrical_engineering","elementary_mathematics","formal_logic","global_facts","high_school_biology","high_school_chemistry","high_school_computer_science","high_school_european_history","high_school_geography","high_school_government_and_politics","high_school_macroeconomics","high_school_mathematics","high_school_microeconomics","high_school_physics","high_school_psychology","high_school_statistics","high_school_us_history","high_school_world_history","human_aging","human_sexuality","international_law","jurisprudence","logical_fallacies","machine_learning","management","marketing","medical_genetics","miscellaneous","moral_disputes","moral_scenarios","nutrition","philosophy","prehistory","professional_accounting","professional_law","professional_medicine","professional_psychology","public_relations","security_studies","sociology","us_foreign_policy","virology","world_religions"]
CHOICES=["A","B","C","D"]

@dataclass
class MMLUExample:
    subject: str; question: str; choices: List[str]; answer: str

def _mmlu_local_loader(subject: str, split: str):
    """Load a local MMLU split from DEFAULT_MMLU_LOCAL_DIR using plain JSONL.

    Expected layout:
      DEFAULT_MMLU_LOCAL_DIR/{subject}/test.jsonl
      DEFAULT_MMLU_LOCAL_DIR/{subject}/dev.jsonl
    Returns:
      A list of dicts, each with keys like "question", "choices", "answer".
    """
    base = DEFAULT_MMLU_LOCAL_DIR
    path = base / subject / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"[MMLU] Local file not found: {path}")

    records = []
    # Simple JSONL reader: one JSON object per line
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def run_mmlu(model, tokenizer, cfg):
    print("[MMLU] Preparing dataset...")
    rng = random.Random(SEED)
    # Always load MMLU from local JSONL files prepared beforehand.
    ds_loader = _mmlu_local_loader

    def _to_letter(ans):
        # Normalize possible formats: int index(0-3) or letter string
        try:
            if isinstance(ans, int):
                return CHOICES[ans]
            if isinstance(ans, str):
                a = ans.strip().upper()
                return a[:1] if a[:1] in CHOICES else a[:1]
        except Exception:
            pass
        return "A"

    pool = [MMLUExample(s, r["question"], r["choices"], _to_letter(r["answer"]))
            for s in MMLU_SUBJECTS
            for r in ds_loader(s, "test")]
    rng.shuffle(pool)
    eval_set = pool[:cfg['num_samples']]
    
    def _sample_demos(subj: str, k: int) -> List[MMLUExample]:
        """Return up to k demos from the subject's DEV split."""
        if k <= 0:
            return []
        dev = ds_loader(subj, "dev")
        if len(dev) == 0:
            return []
        take = min(k, len(dev))
        idxs = rng.sample(range(len(dev)), take)
        out: List[MMLUExample] = []
        for i in idxs:
            r = dev[i]
            out.append(MMLUExample(subj, r["question"], r["choices"], _to_letter(r["answer"])))
        return out

    def _format_prompt(e: MMLUExample) -> str:
        """Build instruction + optional k-shot demos + question."""
        subject = e.subject.replace("_", " ")
        header = (
            f"The following are multiple choice questions (with answers) about {subject}.\n"
            "Choose the correct answer from A to D. Answer with a single letter.\n\n"
        )
        demo_text = ""
        k = int(cfg.get("kshot", 0))
        if k > 0:
            demos = _sample_demos(e.subject, k)
            for d in demos:
                choices_demo = "\n".join(f"{c}. {d.choices[i]}" for i, c in enumerate(CHOICES))
                demo_text += f"Q: {d.question}\n{choices_demo}\nAnswer: {d.answer}\n\n"
        choices = "\n".join(f"{c}. {e.choices[i]}" for i, c in enumerate(CHOICES))
        return header + demo_text + f"Q: {e.question}\n{choices}\nAnswer:"

    prompts = [_format_prompt(e) for e in eval_set]
    golds = [e.answer for e in eval_set]
    
    shot_tag = int(cfg.get("kshot", 0))
    print(f"[MMLU] Generating for {len(prompts)} samples (1-token decoding) [{shot_tag}-shot]...")
    # Generate exactly 1 token to both decide the choice and measure throughput
    outputs, per_batch_stats = generate_and_measure(
        model, tokenizer, prompts,
        cfg['batch_size'], max_new_tokens=1, temperature=0.0
    )
    preds = [( (o or "").strip()[:1].upper() ) for o in outputs]
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    acc = correct / len(eval_set) if eval_set else 0
    
    return {"bench": "mmlu", "score": acc, "score_display": f"acc={acc*100:.2f}% ({correct}/{len(eval_set)})"}, per_batch_stats

# ================== GSM8K ==================
@dataclass
class GSM8KExample:
    question: str; answer: str

def run_gsm8k(model, tokenizer, cfg):
    print("[GSM8K] Preparing dataset...")
    rng = random.Random(SEED)
    
    # Corrected: Removed trust_remote_code=True
    demo_r = rng.choice(list(load_dataset("gsm8k", "main", split="train")))
    demo = GSM8KExample(demo_r["question"], demo_r["answer"])
    
    pool = list(load_dataset("gsm8k", "main", split="test"))
    rng.shuffle(pool)
    eval_set = [GSM8KExample(r["question"], r["answer"]) for r in pool[:cfg['num_samples']]]
    
    prompts = [f"Problem:\n{demo.question}\n\nSolution:\n{demo.answer}\n\n" + f"Problem:\n{e.question}\n\nSolution:\n" if cfg['kshot'] >= 1 else f"Problem:\n{e.question}\n\nSolution:\n" for e in eval_set]
    golds = [e.answer for e in eval_set]
    
    print(f"[GSM8K] Generating for {len(prompts)} samples...")
    outputs, per_batch_stats = generate_and_measure(model, tokenizer, prompts, cfg['batch_size'], cfg['max_new_tokens'], temperature=0.0)
    
    def normalize_answer(text):
        # Match last '#### <num>'; fallback to last number
        text = (text or "").replace(",", "")
        matches = _re.findall(r"####\s*([\-]?\d+(?:\.\d+)?)", text)
        if matches: return matches[-1]
        matches = _re.findall(r"([\-]?\d+(?:\.\d+)?)", text)
        return matches[-1] if matches else ""

    correct = sum(1 for o, g in zip(outputs, golds) if normalize_answer(o) and normalize_answer(o) == normalize_answer(g))
    acc = correct / len(eval_set) if eval_set else 0
    
    return {"bench": "gsm8k", "score": acc, "score_display": f"acc={acc*100:.2f}% ({correct}/{len(eval_set)})"}, per_batch_stats

# ================== HumanEval ==================
def run_humaneval(model, tokenizer, cfg):
    print("[HumanEval] Preparing dataset...")
    # Corrected: Removed trust_remote_code=True
    tasks = list(load_dataset("openai_humaneval", split="test"))
    rng = random.Random(SEED)
    rng.shuffle(tasks)
    tasks = tasks[:cfg['num_samples']]
    prompts = [t['prompt'] for t in tasks]
    
    print(f"[HumanEval] Generating for {len(prompts)} samples...")
    # No stop sequences: we want the full function body
    outputs, per_batch_stats = generate_and_measure(model, tokenizer, prompts, cfg['batch_size'], cfg['max_new_tokens'], temperature=0.0, stop_sequences=None)

    # Strip code fences like ```python ... ```
    def _strip_code_fences(s: str) -> str:
        """Remove leading/trailing Markdown code fences."""
        if s is None:
            return ""
        out = _re.sub(r"^\s*```(?:python)?\s*", "", s, flags=_re.IGNORECASE)
        out = _re.sub(r"\s*```\s*$", "", out)
        return out

    def _sanitize_humaneval_completion(text: str, entry_point: str) -> str:
        """
        Make the completion executable by:
          1) removing code fences,
          2) commenting out natural-language lines before/after code,
          3) ensuring the file starts from code (so imports/helpers are kept).
        This mirrors the behavior that tends to happen implicitly in vLLM path.
        """
        import re
        t = _strip_code_fences(text or "")
        lines = t.splitlines()

        code_starts = ("def ", "class ", "import ", "from ", "@")
        started = False
        out_lines = []
        for ln in lines:
            s = ln.lstrip()
            # Start of "code-like" content
            if not started and any(s.startswith(k) for k in code_starts):
                started = True
                out_lines.append(ln)
                continue
            if not started:
                # Comment out preambles like "Here is the function..."
                out_lines.append("# " + ln if ln.strip() else "")
            else:
                # After code begins, comment out plain-language paragraphs that appear at column 0
                if (not ln.startswith((" ", "\t"))) and not any(s.startswith(k) for k in code_starts) \
                   and s and not s.startswith(("#", "'''", '"""')):
                    out_lines.append("# " + ln)
                else:
                    out_lines.append(ln)

        body = "\n".join(out_lines).strip()
        # If the declared entry_point exists in body, keep as-is; otherwise try to trim from the function start.
        if f"def {entry_point}" in body:
            return body
        m = re.search(rf"def\s+{re.escape(entry_point)}\s*\(", t)
        if m:
            return t[m.start():].strip()
        return body

    passed = 0
    for task, comp in zip(tasks, outputs):
        body = _sanitize_humaneval_completion(comp, task['entry_point'])
        program = "# -*- coding: utf-8 -*-\n" + f"{task['prompt']}{body}\n\n{task['test']}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(program); path = f.name
        try:
            # Run with clean PYTHONPATH and shorter timeout as in reference
            res = subprocess.run([sys.executable, path],
                                 capture_output=True, text=True, timeout=10,
                                 env={**os.environ, "PYTHONPATH": ""})
            if res.returncode == 0: passed += 1
        finally:
            os.unlink(path)
            
    pass_at_1 = passed / len(tasks) if tasks else 0
    return {"bench": "humaneval", "score": pass_at_1, "score_display": f"pass@1={pass_at_1*100:.2f}% ({passed}/{len(tasks)})"}, per_batch_stats

# ================== Line Retrieval ==================
# ==== Helpers for token-aware budgeting (line-retrieval & needle) ====
def _model_ctx_budget(model, tokenizer, safety_margin=16):
    max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not max_ctx:
        max_ctx = getattr(tokenizer, "model_max_length", 16384) or 16384
    return max(128, int(max_ctx) - safety_margin)

_COMMON_WORDS_LR = [
    "the","of","and","to","in","a","is","it","you","that","he","was","for","on","are","as","with",
    "his","they","I","at","be","this","have","from","or","one","had","by","word","but","not","what",
    "all","were","we","when","your","can","said","there","use","an","each","which","do","how","their",
    "if","will","up","other","about","out","many","then","them","these","so","some","her","would","make",
    "like","him","into","time","has","look","two","more","write","go","see","number","no","way","could",
]

def lr_make_line(rng: random.Random, min_words: int, max_words: int) -> str:
    """Generate one synthetic line with random 'words' of 3~8 letters."""
    letters = string.ascii_lowercase
    return " ".join(
        ["".join(rng.choices(letters, k=rng.randint(3, 8)))
         for _ in range(rng.randint(min_words, max_words))]
    )

def lr_choose_target_index(n: int, mode: str, rng: random.Random) -> int:
    """Choose target line index (1-based) depending on sampling mode."""
    if mode == "head":   return 1
    if mode == "tail":   return n
    if mode == "middle": return max(1, min(n, n // 2))
    return rng.randint(1, n)

def lr_count_tokens(tokenizer, text: str) -> int:
    """Count tokens without adding special tokens."""
    return len(tokenizer(text, add_special_tokens=False).input_ids)

def _lr_fmt_block(lines: list[str], enumerate_ln: bool) -> str:
    """Format lines into a display block, optionally enumerated (1-based)."""
    if enumerate_ln:
        return "\n".join(f"{i+1}: {s}" for i, s in enumerate(lines))
    return "\n".join(lines)

def lr_build_prompt(
    lines: list[str], target_idx: int, *, enumerate_ln: bool,
    answer_tag: str, fewshot_demo: int
) -> str:
    """Build instruction + (optional) 1-shot demo + body enclosed by markers."""
    header = (
        "You are given a list of lines between <BEGIN> and <END>.\n"
        f"Return the exact content of line #{target_idx} verbatim.\n"
        f"- Output MUST start with {answer_tag} and then ONLY the line content.\n"
        "- Do NOT add quotes, prefixes, suffixes, or explanations.\n"
    )
    demo = ""
    if fewshot_demo:
        demo_lines = ["alpha beta", "gamma", "delta zeta"]
        demo_target = 2
        demo = (
            "\nExample:\n<BEGIN>\n"
            + _lr_fmt_block(demo_lines, enumerate_ln=True)
            + "\n<END>\nAnswer:\n"
            + f"{answer_tag} {demo_lines[demo_target-1]}\n\n"
        )
    body = "<BEGIN>\n" + _lr_fmt_block(lines, enumerate_ln) + "\n<END>\nAnswer:\n"
    return header + demo + body

def lr_normalize(s: str) -> str:
    """Normalize whitespace for tolerant matching."""
    return re.sub(r"\s+", " ", s.strip())

def lr_extract_answer(text: str, answer_tag: str) -> str | None:
    """Extract the model's answer after [ANS] tag, fallback to first line."""
    if not text:
        return None
    idx = text.find(answer_tag)
    if idx < 0:
        first = text.splitlines()[0] if text else ""
        return first.strip() or None
    sub = text[idx + len(answer_tag):]
    return sub.splitlines()[0].strip() if sub else None

def run_line_retrieval(model, tokenizer, cfg):
    print("[LineRetrieval] Preparing dataset...")

    rng = random.Random(SEED)
    num_samples  = int(cfg.get("num_samples", 10))
    num_lines    = int(cfg.get("lr_num_lines", 2000))
    min_words    = int(cfg.get("lr_min_words", 4))
    max_words    = int(cfg.get("lr_max_words", 10))
    vocab_mode   = cfg.get("lr_vocab_mode", "random")  # "random" | "common"
    target_mode  = cfg.get("lr_target_mode", "random") # "head"|"middle"|"tail"|"random"
    batch_size   = int(cfg.get("batch_size", 1))
    max_new_tok  = int(cfg.get("max_new_tokens", 64))
    answer_tag   = cfg.get("lr_answer_tag", "[ANS]")
    enumerate_ln = bool(cfg.get("lr_enumerate_lines", True))
    stop_on_nl   = bool(cfg.get("lr_stop_on_newline", True))
    strict_exact = bool(cfg.get("lr_strict_exact", False))
    fewshot_demo = int(cfg.get("lr_fewshot_demo", 1))

    # === Token budget for the *prompt text only* ===
    # We must (1) cap the constructed problem within 10k tokens,
    # and (2) still respect the model's max context (minus a safety margin).
    # Final prompt budget = min(lr_max_prompt_tokens, model_budget).
    lr_max_prompt_tokens = int(cfg.get("lr_max_prompt_tokens", 10000))
    ctx_margin_tokens = int(cfg.get("ctx_margin_tokens", 128))
    model_max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None) or 32768
    model_budget  = max(256, int(model_max_ctx) - ctx_margin_tokens)
    budget = max(256, min(lr_max_prompt_tokens, model_budget))

    # Build samples with budget enforcement
    samples = []
    for s in range(num_samples):
        # Build full corpus & select target
        lines = [lr_make_line(rng, min_words, max_words) for _ in range(num_lines)]
        target_idx = lr_choose_target_index(num_lines, target_mode, rng)
        gold = lines[target_idx - 1]

        # Fit to token budget (cap to <= 10k and model limit)
        attempt, cur_num = 0, num_lines
        prompt = lr_build_prompt(
            lines, target_idx,
            enumerate_ln=enumerate_ln, answer_tag=answer_tag, fewshot_demo=fewshot_demo
        )
        tok_len = lr_count_tokens(tokenizer, prompt)
        while tok_len > budget and attempt < 12 and cur_num > 1:
            shrink_ratio = budget / float(tok_len)
            new_num = max(1, int(math.floor(cur_num * shrink_ratio * 0.95)))
            if new_num == cur_num:
                new_num = max(1, cur_num - 1)
            lines       = lines[:new_num]
            target_idx  = min(target_idx, new_num)
            gold        = lines[target_idx - 1]
            prompt      = lr_build_prompt(
                lines, target_idx,
                enumerate_ln=enumerate_ln, answer_tag=answer_tag, fewshot_demo=fewshot_demo
            )
            tok_len     = lr_count_tokens(tokenizer, prompt)
            cur_num     = new_num
            attempt    += 1

        samples.append((prompt, gold))

    prompts = [p for p, _ in samples]
    golds   = [g for _, g in samples]

    print(f"[LineRetrieval] Generating for {len(prompts)} samples...")
    # Decode: stop at first newline to avoid extra text
    stop_sequences = ["\n"] if stop_on_nl else None
    max_new_tok = min(max_new_tok, max(8, max_words * 4))  # small line is enough
    outputs, per_batch_stats = generate_and_measure(
        model, tokenizer, prompts, batch_size, max_new_tok,
        temperature=0.0, top_p=1.0, stop_sequences=stop_sequences
    )

    # Post-process & score
    preds = [o.strip() for o in outputs]
    preds = []
    for out in outputs:
        ans = lr_extract_answer(out, answer_tag) or ""
        preds.append(ans)
    
    correct = 0
    for pred, gold in zip(preds, golds):
       ok = (pred == gold) if strict_exact else (lr_normalize(pred) == lr_normalize(gold))
       correct += int(ok)

    acc = correct / max(1, len(golds))
    print(f"[LINE_RETRIEVAL] acc={acc*100:.2f}% ({correct}/{len(golds)})")

    return {"bench": "line_retrieval", "score": acc, "score_display": f"acc={acc*100:.2f}% ({correct}/{len(golds)})"}, per_batch_stats

# ================== LongBench ==================
_LB_ALLOWED = {"qasper", "hotpotqa", "2wikimqa", "musique"}

_LB_ARTICLES = {"a", "an", "the"}
_LB_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

def _lb_normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, drop English articles, and squeeze spaces."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(f"[{re.escape(_LB_PUNCT)}]", " ", s)
    s = " ".join(w for w in s.split() if w not in _LB_ARTICLES)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _lb_f1_em(pred: str, gold_list: List[str]) -> Tuple[float, float]:
    """Return (max_f1, max_em) exactly as in longbench.py."""
    pn = _lb_normalize_text(pred)
    if pn == "" and any(_lb_normalize_text(g) == "" for g in gold_list):
        return 1.0, 1.0
    best_f1, best_em = 0.0, 0.0
    for g in gold_list:
        gn = _lb_normalize_text(g)
        if gn == pn:
            best_em = max(best_em, 1.0)
        ptoks, gtoks = pn.split(), gn.split()
        if len(ptoks) == 0 and len(gtoks) == 0:
            f1 = 1.0
        elif len(ptoks) == 0 or len(gtoks) == 0:
            f1 = 0.0
        else:
            # multiset overlap
            count_p, count_g = {}, {}
            for t in ptoks: count_p[t] = count_p.get(t, 0) + 1
            for t in gtoks: count_g[t] = count_g.get(t, 0) + 1
            common = 0
            for t in count_p:
                if t in count_g:
                    common += min(count_p[t], count_g[t])
            if common == 0:
                f1 = 0.0
            else:
                prec = common / len(ptoks)
                rec  = common / len(gtoks)
                f1   = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)
    return best_f1, best_em

def _lb_safe_tail_truncate(tokenizer, text: str, budget: int) -> str:
    """Keep the tail tokens within budget (preserve the question tail)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    keep = ids[-budget:]
    return tokenizer.decode(keep, skip_special_tokens=True)

_LB_INSTRUCTIONS = {
    # Match longbench.py INSTRUCTIONS (exact wording matters)
    "qasper": (
        "You are given a scientific article and a question.\n"
        "Answer ONLY with exact phrase(s) copied verbatim from the article.\n"
        "Do NOT add, paraphrase, or infer any new words.\n"
        "If the question cannot be answered from the article, write: unanswerable.\n"
        "If it is yes/no, answer: yes, no, or unanswerable.\n"
        "Output ONLY the answer text, without quotes or extra words.\n"
    ),
    "hotpotqa": (
        "You are given a long passage and a multi-hop question.\n"
        "Answer ONLY with the minimal text span from the passage.\n"
        "If it is yes/no, answer: yes or no. If unanswerable, write: unanswerable.\n"
        "Output ONLY the answer text.\n"
    ),
    "2wikimqa": (
        "You are given a long passage derived from multiple Wikipedia pages and a question.\n"
        "Answer ONLY with the minimal text span from the passage (no extra words).\n"
        "If it is yes/no, answer: yes or no. If unanswerable, write: unanswerable.\n"
        "Output ONLY the answer text.\n"
    ),
    "musique": (
        "You are given a long passage and a question that may require multi-hop reasoning.\n"
        "Answer ONLY with a short span copied from the passage.\n"
        "If unanswerable, write: unanswerable.\n"
        "Output ONLY the answer text.\n"
    ),
}

def _lb_resolve_file(dataset_name: str, base_dir: Optional[str]) -> str:
    """
    Resolve a concrete JSONL path for LongBench v1 subsets (qasper/hotpotqa/2wikimqa/musique).
    Prioritize the extracted structure shown by the user's zip (data/data/*.jsonl).
    """
    if dataset_name not in _LB_ALLOWED:
        raise ValueError(f"[LongBench] Unsupported subset '{dataset_name}'. Allowed: {_LB_ALLOWED}")

    fname = f"{dataset_name}.jsonl"
    candidates: List[Path] = []

    # 1) If user provided --data_dir, search under it.
    if base_dir:
        b = Path(base_dir)
        candidates += [
            b / "data" / "data" / fname,  # unzip -d data  => data/data/<name>.jsonl
            b / "data" / fname,           # sometimes flat data/<name>.jsonl
            b / fname,                    # rare: directly under base
        ]

    # 2) Relative to script dir and CWD (robust fallback)
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates += [
        here / "LongBench" / "data" / "data" / fname,
        here / "data" / "data" / fname,
        here / "data" / fname,
        cwd / "LongBench" / "data" / "data" / fname,
        cwd / "data" / "data" / fname,
        cwd / "data" / fname,
    ]

    for p in candidates:
        if p.exists():
            return str(p)

    tried = "\n  - " + "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"[LongBench] Could not locate JSONL for subset '{dataset_name}'. "
        f"Place files as shown by your unzip log, e.g. data/data/{fname}.\nTried:{tried}"
    )

def _lb_load_json_dataset(dataset_name: str, data_dir: Optional[str]):
    """Load entire LongBench subset locally; do NOT slice here (shuffle first, then slice)."""
    from datasets import load_dataset
    local_file = _lb_resolve_file(dataset_name, data_dir)
    ds = load_dataset("json", data_files=local_file, split="train")
    return list(ds)

def run_longbench(model, tokenizer, cfg, dataset_name: str):
    print(f"[LongBench:{dataset_name}] Preparing dataset...")
    # 1) Load all, then shuffle & slice to num_samples
    data_dir = cfg.get("lb_data_dir") or cfg.get("data_dir")
    pool = _lb_load_json_dataset(dataset_name, data_dir)
    rnd = random.Random(cfg.get("seed", SEED))
    rnd.shuffle(pool)
    ds = pool[: cfg.get("num_samples") or len(pool)]

    # 2) Compute tokenizer-aware context budget:
    #    budget = max_model_len - ctx_margin_tokens - max_new_tokens  (default margin=128)
    ctx_margin_tokens = cfg.get("ctx_margin_tokens", 128)
    max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not max_ctx:
        max_ctx = getattr(tokenizer, "model_max_length", 16384) or 16384
    budget = max(256, int(max_ctx) - int(ctx_margin_tokens) - int(cfg['max_new_tokens']))

    def _ctx_budget_for(sample_question: str, instr: str) -> int:
        """Give almost all available tokens to the document by subtracting the non-doc overhead."""
        overhead_ids = tokenizer.encode(
            f"{instr}\nDocument:\n\n\nQuestion:\n{sample_question}\n\nAnswer:",
            add_special_tokens=False
        )
        return max(64, budget - len(overhead_ids))

    # 3) Build prompts with tail-preserving truncation of ONLY the document
    prompts, golds = [], []
    instr = _LB_INSTRUCTIONS[dataset_name]
    for r in ds:
        # Robust field extraction across LB JSON variants
        ctx = r.get("context") or r.get("passage") or r.get("document") or ""
        q   = r.get("input") or r.get("question") or ""
        ans = r.get("answers") or r.get("answer") or []
        if isinstance(ans, str): ans = [ans]
        if ans and isinstance(ans[0], list):
            # Flatten possible [[...], ...] style
            flat = []
            for sub in ans:
                flat.extend(sub if isinstance(sub, list) else [sub])
            ans = flat

        raw_prompt = f"{instr}\nDocument:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"
        tok_len = len(tokenizer.encode(raw_prompt, add_special_tokens=False))
        if tok_len > budget:
            # Dynamically compute how many tokens we can give to the document
            ctx_budget = _ctx_budget_for(q, instr)
            ctx_trunc  = _lb_safe_tail_truncate(tokenizer, ctx, ctx_budget)
            raw_prompt = f"{instr}\nDocument:\n{ctx_trunc}\n\nQuestion:\n{q}\n\nAnswer:"
        prompts.append(raw_prompt); golds.append(ans)

    print(f"[LongBench:{dataset_name}] Generating for {len(prompts)} samples.")
    outputs, per_batch_stats = generate_and_measure(
        model, tokenizer, prompts,
        cfg['batch_size'], cfg['max_new_tokens'],
        temperature=0.0, stop_sequences=None
    )

    # 4) Score
    f1s, ems = [], []
    for o, gs in zip(outputs, golds):
        f1, em = _lb_f1_em(o or "", gs or [])
        f1s.append(f1); ems.append(em)
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    avg_em = sum(ems) / len(ems) if ems else 0.0
    return {"bench": f"longbench_{dataset_name}", "score": avg_f1, "score_display": f"F1={avg_f1*100:.2f}%, EM={avg_em*100:.2f}%"}, per_batch_stats

# ================== Needle-in-a-Haystack ==================
def _nh_rand_word_common(rng):
    return rng.choice(_COMMON_WORDS_LR)

_NEEDLE_TOKEN_RE = re.compile(r"([A-Za-z0-9\-]{2,})")

def _nh_normalize_value(s: str) -> str:
    if s is None: return ""
    return s.strip().strip("\"'`.,;: ")

def _nh_extract_token(text: str) -> str:
    """If phrase is returned, extract a code-like token (e.g., N123456)."""
    if not text: return ""
    t = text.strip()
    m = _NEEDLE_TOKEN_RE.search(t)
    return m.group(1) if m else t.split()[0]

def run_needle(model, tokenizer, cfg):
    print("[Needle] Preparing dataset.")
    num_samples = int(cfg.get('num_samples', 10))
    batch_size  = int(cfg.get('batch_size', 1))
    max_new_tok = int(cfg.get('max_new_tokens', 8))
    target_tokens = int(cfg.get('nh_target_tokens', 12000))
    depth        = float(cfg.get('nh_depth', 0.5))
    vocab_mode   = cfg.get('nh_vocab_mode', 'random')  # "random" | "common"

    # Context budget for the prompt
    safety_margin = int(cfg.get('ctx_margin_tokens', 64))
    model_max = getattr(getattr(model, "config", None), "max_position_embeddings", None) or 32768
    budget = max(256, model_max - safety_margin - max_new_tok)

    def _make_tokens_to_budget(target_tokens: int, rng_i: random.Random) -> str:
        parts, sent_words = [], rng_i.randint(7, 14)
        while True:
            words = ([_nh_rand_word_common(rng_i) for _ in range(sent_words)]
                     if vocab_mode == "common" else
                     [_nh_rand_word_common(rng_i) for _ in range(sent_words)])
            parts.append(" ".join(words) + ". ")
            txt = "".join(parts)
            if len(tokenizer.encode(txt)) >= target_tokens:
                return txt

    def _insert_needle_by_depth(text: str, needle_sentence: str, frac: float) -> str:
        ids = tokenizer.encode(text)
        k = max(0, min(len(ids), int(round(frac * len(ids)))))
        head = tokenizer.decode(ids[:k], skip_special_tokens=True)
        tail = tokenizer.decode(ids[k:],  skip_special_tokens=True)
        return head + needle_sentence + " " + tail

    # Build prompts/golds
    prompts, golds = [], []
    for i in range(num_samples):
        rng_i = random.Random(SEED + 100003 * i)
        value = "N" + str(rng_i.randint(100000, 999999))
        needle_sentence = f"NEEDLE: {value}."
        target = min(target_tokens, budget - 64)
        if target <= 64: target = max(64, budget // 2)
        base = _make_tokens_to_budget(target, rng_i)
        doc  = _insert_needle_by_depth(base, needle_sentence, frac=max(0.0, min(1.0, depth)))
        prompt = (
            "You will read a long document. Hidden within it is a sentence of the form 'NEEDLE: <value>'.\n"
            "Return ONLY the exact <value>. Do not add any extra words or punctuation. Output only the value.\n"
            f"\nDocument:\n{doc}\n\nAnswer:"
        )
        prompts.append(prompt); golds.append(value)

    print(f"[Needle] Generating for {len(prompts)} samples...")
    outputs, per_batch_stats = generate_and_measure(
        model, tokenizer, prompts,
        batch_size, max_new_tok,
        temperature=0.0, stop_sequences=None
    )

    # Robust scoring (normalize & token-extract), then exact match.
    preds = []
    correct = 0
    for o, gold in zip(outputs, golds):
        pred_text = (o or "").strip()
        pred_val  = _nh_normalize_value(pred_text)
        if len(pred_val.split()) > 1:
            pred_val = _nh_extract_token(pred_val)
        preds.append(pred_val)
        if pred_val == gold:
            correct += 1
    acc = correct / len(golds) if golds else 0.0
    print(f"[NEEDLE] acc={acc*100:.2f}% ({correct}/{len(golds)})")

    return {"bench": "needle", "score": acc, "score_display": f"acc={acc*100:.2f}% ({correct}/{len(golds)})"}, per_batch_stats


# ==========================================================================================
# Main Orchestrator
# ==========================================================================================
def main():
    _cleanup_cuda()

    parser = argparse.ArgumentParser(description="KIVI Benchmark Orchestrator")
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--quant-config-path", type=str, default=None, help="Path to KV Quantization JSON config.")
    parser.add_argument("--k-bits", type=int, default=4); parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32); parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument("--bench", type=str, default="mmlu", help="Benchmark to run.")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--num_samples", type=int, default=None); parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--tag", type=str, default=""); parser.add_argument("--use_flash", type=str, default="true")
    parser.add_argument("--data_dir", "--lb_data_dir", "--data-dir",
                        dest="data_dir", type=str, default=None,
                        help="Path to LongBench raw JSON/JSONL files (e.g., .../LongBench/data).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    
    BENCH_RUNNERS = {
        "mmlu": run_mmlu, "gsm8k": run_gsm8k, "humaneval": run_humaneval,
        "line_retrieval": run_line_retrieval, "needle": run_needle,
        "longbench_qasper": lambda m,t,c: run_longbench(m,t,c, "qasper"),
        "longbench_hotpotqa": lambda m,t,c: run_longbench(m,t,c, "hotpotqa"),
        "longbench_2wikimqa": lambda m,t,c: run_longbench(m,t,c, "2wikimqa"),
        "longbench_musique": lambda m,t,c: run_longbench(m,t,c, "musique"),
    }

    def run_one(bench_name, bench_cfg):
        print(f"\n{'='*25} Running benchmark: {bench_name.upper()} {'='*25}")
        
        if args.num_samples is not None: bench_cfg['num_samples'] = args.num_samples
        if args.batch_size is not None: bench_cfg['batch_size'] = args.batch_size
        if args.data_dir is not None: bench_cfg["lb_data_dir"] = args.data_dir
        
        model, tokenizer = None, None
        result: Dict[str, Any] = {}
        metrics_raw: Dict[str, Any] = {}
        metrics_formatted: Dict[str, Any] = {}
        per_batch_metrics_list: List[Dict[str, Any]] = []
        t_start = time.time()
        
        try:
            model, tokenizer = load_kivi_model_and_tokenizer(args)
            base_mem = warmup_and_measure_base(model, tokenizer)
            
            runner_fn = BENCH_RUNNERS[bench_name]
            result, per_batch_metrics_list = runner_fn(model, tokenizer, bench_cfg)

            total_tokens = sum(b['new_tokens'] for b in per_batch_metrics_list)
            total_gen_time = sum(b['ttft_sec'] + b['decode_duration_sec'] for b in per_batch_metrics_list)
            avg_ttft_ms = (sum(b['ttft_sec'] for b in per_batch_metrics_list) / len(per_batch_metrics_list)) * 1000 if per_batch_metrics_list else 0
            max_peak_mem = max(b['peak_inference_mem_bytes'] for b in per_batch_metrics_list) if per_batch_metrics_list else 0
            
            metrics_raw = {
                "avg_ttft_ms": avg_ttft_ms,
                "throughput_tok_per_sec": total_tokens / total_gen_time if total_gen_time > 0 else 0,
                "peak_inference_mem_bytes": max(0, max_peak_mem - base_mem),
            }
            metrics_formatted = {
                "avg_ttft": f"{metrics_raw['avg_ttft_ms']:.2f} ms",
                "throughput": f"{metrics_raw['throughput_tok_per_sec']:.2f} tok/s",
                "peak_inference_memory": fmt_bytes(metrics_raw['peak_inference_mem_bytes']),
            }

        except Exception as e:
            print(f"ERROR running benchmark {bench_name}: {e}")
            import traceback; traceback.print_exc()
        finally:
            if model is not None or tokenizer is not None:
                del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        total_time = time.time() - t_start
        score_str = result.get("score_display", "N/A")

        print(f"[{bench_name.upper()}] Finished in {total_time:.2f}s | Score: {score_str}")
        if metrics_raw and metrics_formatted:
            print(f"  [METRICS] Avg TTFT: {metrics_formatted['avg_ttft']} | Throughput: {metrics_formatted['throughput']} | Peak Memory: {metrics_formatted['peak_inference_memory']}")

        for batch_stat in per_batch_metrics_list: batch_stat['peak_inference_memory'] = fmt_bytes(max(0, batch_stat['peak_inference_mem_bytes'] - base_mem))

        result.update({
            'config': bench_cfg,
            'model_path': args.model,
            'total_runtime_sec': total_time,
            'metrics_aggregate': {**metrics_raw, **metrics_formatted},
            'per_batch_metrics': per_batch_metrics_list
        })
        filename = f"{bench_name}_{args.tag}.json" if args.tag else f"{bench_name}.json"
        out_path = results_dir / filename
        with open(out_path, "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[{bench_name.upper()}] Results saved to: {out_path}")
        
        return bench_name, {"score": score_str, "runtime": total_time, "throughput": metrics_raw.get('throughput_tok_per_sec', 0), "avg_ttft_ms": metrics_raw.get('avg_ttft_ms', 0), "path": str(out_path)}

    try:
        torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
        if np is not None: np.random.seed(SEED)
        random.seed(SEED)
    except Exception:
        pass

    summary = {}
    total_start = time.time()
    benches_to_run = BENCH_ALL_CONFIG.items() if args.bench == "all" else [(args.bench, BENCH_ALL_CONFIG.get(args.bench, {}))]
    
    for name, config in benches_to_run:
        if name not in BENCH_RUNNERS: print(f"Warning: Benchmark '{name}' not found, skipping."); continue
        for key, default in [('num_samples',100), ('kshot',0), ('batch_size',4), ('max_new_tokens',256)]: config.setdefault(key, default)
        b_name, meta = run_one(name, config)
        summary[b_name] = meta
        
    total_dur = time.time() - total_start
    print(f"\n{'='*30} Overall Summary {'='*30}")
    for k, v in summary.items(): print(f"- {k.upper():<20}: {v['score']} | Avg TTFT: {v['avg_ttft_ms']:.2f} ms | Throughput: {v['throughput']:.2f} tok/s (took {v['runtime']:.2f}s) -> {v['path']}")
    print(f"Total wall time: {total_dur:.2f}s")
    print("="*77)

if __name__ == "__main__":
    main()