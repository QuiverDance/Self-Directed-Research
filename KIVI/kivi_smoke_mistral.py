# kivi_bench_multi.py
# Multi-request bench on KIVI (Mistral-7B) with:
# - Throughput aggregation: Σtokens / Σdecode_time
# - Memory metric = (GPU allocated bytes) - (model base allocated)
# - Per-request peak over model base + leak check after cleanup
#
# NOTE: single-GPU assumption for memory metrics. KIVI kernels expect FP16.

import os, time, math, gc, argparse, torch
from transformers import AutoConfig, AutoTokenizer

def warmup_and_measure_base(model, tok, device):
    """Run a tiny warmup so that persistent runtime buffers are created.
    Return 'model_base_alloc_warm' measured after cleanup.
    """
    # base right after load
    torch.cuda.synchronize(device)
    base0 = torch.cuda.memory_allocated(device)

    # tiny warmup
    tiny = tok("Hi", return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**tiny, use_cache=True, return_dict=True)
        # one decode step
        nxt = tiny["input_ids"][:, -1:]
        out = model(input_ids=nxt, use_cache=True, past_key_values=out.past_key_values, return_dict=True)

    # cleanup temporaries
    del out, tiny, nxt
    torch.cuda.synchronize(device)
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # base after warmup (includes persistent buffers)
    torch.cuda.synchronize(device)
    base1 = torch.cuda.memory_allocated(device)
    return max(base0, base1)

def fmt_bytes(n: int) -> str:
    units = ["B","KiB","MiB","GiB","TiB"]
    if n <= 0: return "0 B"
    k = 1024.0
    i = int(math.floor(math.log(n, k)))
    return f"{n / (k**i):.2f} {units[i]}"

def get_model_dims(cfg):
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    n_heads  = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    n_kv     = getattr(cfg, "num_key_value_heads", n_heads)
    hidden   = getattr(cfg, "hidden_size", None)
    head_dim = hidden // n_heads
    return n_layers, n_heads, n_kv, head_dim

def fp16_kv_bytes_per_token(n_layers, n_kv, head_dim, bytes_per_elem=2):
    return 2 * n_layers * n_kv * head_dim * bytes_per_elem  # 2 = (K,V)

def quant_kv_bytes_per_token_lower(n_layers, n_kv, head_dim, k_bits, v_bits):
    kb = n_layers * n_kv * head_dim * (k_bits / 8.0)
    vb = n_layers * n_kv * head_dim * (v_bits / 8.0)
    return int(kb + vb)

def quant_kv_bytes_per_token_with_meta(n_layers, n_kv, head_dim, k_bits, v_bits, group_size, meta_bpp=2):
    # Rough upper bound with meta (scale+zp as FP16 → 2B each → 4B per group entry)
    data = quant_kv_bytes_per_token_lower(n_layers, n_kv, head_dim, k_bits, v_bits)
    per_group_bytes = 2 * meta_bpp
    meta_k = int(n_layers * n_kv * head_dim * per_group_bytes / group_size)               # amortized over time
    meta_v = int(n_layers * n_kv * math.ceil(head_dim / group_size) * per_group_bytes)    # per token
    return data + meta_k + meta_v

def global_throughput(tokens_list, time_list):
    T, S = sum(tokens_list), sum(time_list)
    return (T / S) if S > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="~/models/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--num-reqs", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prompts-file", type=str, default="")
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--k-bits", type=int, default=2)
    ap.add_argument("--v-bits", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--residual-length", type=int, default=32)
    ap.add_argument("--leak-tol-bytes", type=int, default=1_000_000)  # ~1MB tolerance
    args = ap.parse_args()

    model_dir = os.path.expanduser(args.model)

    # Load cfg & KIVI wrapper
    cfg = AutoConfig.from_pretrained(model_dir)
    mt = getattr(cfg, "model_type", None)
    if mt == "mistral":
        from models.mistral_kivi import MistralForCausalLM_KIVI as KIVIModel
    elif mt == "llama":
        from models.llama_kivi import LlamaForCausalLM_KIVI as KIVIModel
    else:
        raise RuntimeError(f"Unsupported model_type: {mt}")

    cfg.k_bits, cfg.v_bits = int(args.k_bits), int(args.v_bits)
    cfg.group_size, cfg.residual_length = int(args.group_size), int(args.residual_length)

    # FP16 to match KIVI CUDA kernels
    dtype = torch.float16
    print(f"[SETUP] model={model_dir} dtype={dtype} device_map={args.device_map}")
    print(f"[KIVI] k_bits={cfg.k_bits} v_bits={cfg.v_bits} group_size={cfg.group_size} residual_length={cfg.residual_length}")

    model = KIVIModel.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        config=cfg,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    ).eval()

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        # Baseline (weights resident): define "model size" via allocated bytes
        model_base_alloc = torch.cuda.memory_allocated(device)
        model_base_reserved = torch.cuda.memory_reserved(device)
    else:
        model_base_alloc = model_base_reserved = 0

    # Model dims + theoretical KV
    nL, nH, nKV, dH = get_model_dims(cfg)
    fp_per_tok   = fp16_kv_bytes_per_token(nL, nKV, dH, 2)
    q_per_tok_lo = quant_kv_bytes_per_token_lower(nL, nKV, dH, cfg.k_bits, cfg.v_bits)
    q_per_tok_me = quant_kv_bytes_per_token_with_meta(nL, nKV, dH, cfg.k_bits, cfg.v_bits, cfg.group_size)

    print(f"[GPU] model_base_alloc={fmt_bytes(model_base_alloc)} | model_base_reserved={fmt_bytes(model_base_reserved)}")
    print(f"[MODEL] L={nL} H={nH} KV={nKV} d={dH} | KV/token FP16={fmt_bytes(fp_per_tok)} "
          f"| Q(no-meta)={fmt_bytes(q_per_tok_lo)} | Q(+meta≈)={fmt_bytes(q_per_tok_me)}")

    # Build prompts
    prompts = []
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        base = "You are a helpful assistant.\n\nQ: What is attention?\nA:"
        prompts = [base for _ in range(args.num_reqs)]
    elif len(prompts) < args.num_reqs:
        prompts = (prompts * ((args.num_reqs + len(prompts)-1)//len(prompts)))[:args.num_reqs]
    else:
        prompts = prompts[:args.num_reqs]

    per_req = []
    dec_times, dec_tokens, peaks_over_model = [], [], []

    for i, prompt in enumerate(prompts, 1):
        # Sanity: check pre-request ex-model usage (should be ~0)
        if torch.cuda.is_available():
            model_base_alloc = warmup_and_measure_base(model, tok, device)
            model_base_reserved = torch.cuda.memory_reserved(device)  # 참고용
        else:
            model_base_alloc = model_base_reserved = 0
        print(f"[GPU] model_base_alloc(after warmup)={fmt_bytes(model_base_alloc)} | model_base_reserved={fmt_bytes(model_base_reserved)}")
        inputs = tok(prompt, return_tensors="pt").to(device)
        in_len = inputs["input_ids"].shape[1]

        # Reset peak, but we'll subtract model_base from the peak later
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # ---- PREFILL ----
        t0 = time.time()
        with torch.inference_mode():
            out = model(**inputs, use_cache=True, return_dict=True)
        t1 = time.time()
        past = out.past_key_values

        # ---- DECODE ----
        dec_t, gen = 0.0, 0
        next_ids = inputs["input_ids"][:, -1:]
        for _ in range(args.max_new_tokens):
            st = time.time()
            with torch.inference_mode():
                out = model(input_ids=next_ids, use_cache=True, past_key_values=past, return_dict=True)
            ed = time.time()
            dec_t += (ed - st)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            gen += 1
            past = out.past_key_values
            next_ids = next_token

        # ---- Memory snapshots (ex-model) ----
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated(device)
            end_alloc  = torch.cuda.memory_allocated(device)
            # Our metric: subtract model_base_alloc
            peak_over_model = max(0, peak_alloc - model_base_alloc)
            end_over_model  = max(0, end_alloc  - model_base_alloc)
        else:
            peak_over_model = end_over_model = 0

        # ---- Cleanup to release KV etc. ----
        del out, inputs, past, next_ids
        torch.cuda.synchronize(device) if torch.cuda.is_available() else None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            post_cleanup_alloc = torch.cuda.memory_allocated(device)
            post_over_model    = max(0, post_cleanup_alloc - model_base_alloc)
        else:
            post_over_model = 0

        # Decode TPS
        tokps = gen / max(dec_t, 1e-9)

        # Theoretical totals (for reference only)
        total_tok = in_len + gen
        fp_total  = total_tok * fp_per_tok
        q_lo_tot  = total_tok * q_per_tok_lo
        q_me_tot  = total_tok * q_per_tok_me

        dec_times.append(dec_t)
        dec_tokens.append(gen)
        peaks_over_model.append(peak_over_model)

        per_req.append({
            "req": i, "in": in_len, "gen": gen,
            "prefill_s": (t1 - t0), "decode_s": dec_t, "tokps": tokps,
            "peak_over_model": peak_over_model,
            "end_over_model": end_over_model,
            "post_over_model": post_over_model,
            "kv_fp16_total": fp_total, "kv_q_lo_total": q_lo_tot, "kv_q_me_total": q_me_tot,
        })

        leak_flag = "OK"
        if post_over_model > args.leak_tol_bytes:
            leak_flag = f"LEAK? (+{fmt_bytes(post_over_model)})"

        print(f"[REQ#{i}] in={in_len} gen={gen} | prefill={(t1-t0):.2f}s  "
              f"decode={dec_t:.2f}s ({tokps:.2f} tok/s) | "
              f"peak_ex_model={fmt_bytes(peak_over_model)}  end_ex_model={fmt_bytes(end_over_model)}  "
              f"after_cleanup={leak_flag}")

    # Aggregate summary
    agg_tps = global_throughput(dec_tokens, dec_times)
    max_peak_over_model = max(peaks_over_model) if peaks_over_model else 0
    sum_fp  = sum(x["kv_fp16_total"]  for x in per_req)
    sum_qlo = sum(x["kv_q_lo_total"]  for x in per_req)
    sum_qme = sum(x["kv_q_me_total"]  for x in per_req)

    print("\n===== SUMMARY =====")
    print(f"requests={len(per_req)}  total_generated={sum(dec_tokens)}  total_decode_time={sum(dec_times):.2f}s")
    print(f"[Throughput] aggregate_tokens_per_sec = {agg_tps:.2f}  (Σtokens / Σtime)")
    print(f"[GPU] model_base_alloc={fmt_bytes(model_base_alloc)} | max_peak_ex_model={fmt_bytes(max_peak_over_model)}")
    print(f"[KV Totals | Sum-of-runs] FP16={fmt_bytes(sum_fp)} | Q(no-meta)={fmt_bytes(sum_qlo)} | Q(+meta≈)={fmt_bytes(sum_qme)}")
    if sum_qme > 0:
        print(f"[KV Compression | Sum-of-runs] FP16 / Q(+meta≈) ≈ {sum_fp / sum_qme:.2f}×")
    print("===================\n")

    # Compact table
    print("req  in  gen  prefill_s  decode_s  tok/s   peak_ex_model   end_ex_model   after_cleanup")
    for r in per_req:
        status = "OK" if r["post_over_model"] <= args.leak_tol_bytes else f"+{fmt_bytes(r['post_over_model'])}"
        print(f"{r['req']:>3}  {r['in']:>3}  {r['gen']:>3}  "
              f"{r['prefill_s']:.2f}     {r['decode_s']:.2f}   {r['tokps']:.2f}  "
              f"{fmt_bytes(r['peak_over_model']):>13}  {fmt_bytes(r['end_over_model']):>13}  {status}")

if __name__ == "__main__":
    main()
