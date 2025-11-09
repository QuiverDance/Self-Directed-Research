# benchmark_kivi.py
# A benchmark script for KIVI (Mistral-7B) to measure key performance metrics.
# - Measures Runtime, Time-To-First-Token (TTFT), Throughput (Tokens/s), and Peak Inference Memory Footprint.
# - Runs a fixed set of 3 diverse prompts.
# - Prints the generated text for each request to verify model output.

import os
import time
import math
import gc
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer

# KVTuner: Import the function to process the layer-wise config.
from utils.process_args import load_and_process_kvtuner_config

def warmup_and_measure_base(model, tok, device):
    """
    Run a tiny warmup so that persistent runtime buffers are created.
    Return 'model_base_alloc_warm' measured after cleanup.
    """
    # Base right after load
    torch.cuda.synchronize(device)
    base0 = torch.cuda.memory_allocated(device)

    # Tiny warmup
    tiny = tok("Hi", return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**tiny, use_cache=True, return_dict=True)
        # One decode step
        nxt = tiny["input_ids"][:, -1:]
        out = model(input_ids=nxt, use_cache=True, past_key_values=out.past_key_values, return_dict=True)

    # Cleanup temporaries
    del out, tiny, nxt
    gc.collect()
    torch.cuda.empty_cache()

    # Base after warmup (includes persistent buffers)
    torch.cuda.synchronize(device)
    base1 = torch.cuda.memory_allocated(device)
    return max(base0, base1)

def fmt_bytes(n: int) -> str:
    """Formats a number of bytes into a human-readable string (e.g., MiB, GiB)."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    if n <= 0: return "0 B"
    k = 1024.0
    i = int(math.floor(math.log(n, k)))
    return f"{n / (k**i):.2f} {units[i]}"

def main():
    # --- Total Benchmark Timer Start ---
    total_start_time = time.time()
    
    ap = argparse.ArgumentParser(description="KIVI Benchmark Script")
    ap.add_argument("--model", type=str, default="~/models/Mistral-7B-Instruct-v0.2", help="Path to the model directory.")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate for each prompt.")
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--k-bits", type=int, default=2, help="Default bits for Key cache quantization.")
    ap.add_argument("--v-bits", type=int, default=2, help="Default bits for Value cache quantization.")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--residual-length", type=int, default=32)
    ap.add_argument("--layer-quant-config-path", type=str, default=None, help="Path to the KVTuner layer-wise quantization config JSON file.")
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

    # KVTuner: Load config, which now returns a map/marker and the group choices.
    layer_quant_map, group_choices = None, None
    if args.layer_quant_config_path:
        layer_quant_map, group_choices = load_and_process_kvtuner_config(args.layer_quant_config_path)

    cfg.k_bits, cfg.v_bits = int(args.k_bits), int(args.v_bits)
    cfg.group_size, cfg.residual_length = int(args.group_size), int(args.residual_length)

    dtype = torch.float16
    print(f"[SETUP] model={model_dir} dtype={dtype} device_map={args.device_map}")
    
    # KVTuner: Determine and print the run mode based on the loaded config.
    run_label = ""
    model = None

    is_baseline_mode = (layer_quant_map == "disabled")
    if is_baseline_mode:
        run_label = "Baseline (FP16)"
        print(f"[MODE] {run_label}: Loading original transformers MistralForCausalLM.")
        
        # Import and load the original, non-quantized model from transformers library.
        from transformers import MistralForCausalLM
        model = MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            config=cfg,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=args.device_map,
        ).eval()

    else: # This block handles both uniform and layer-wise quantization.
        # Import the KIVI model wrapper.
        from models.mistral_kivi import MistralForCausalLM_KIVI as KIVIModel
        
        if isinstance(layer_quant_map, dict):
            run_label = "Quantized (KIVI) - Layer-wise"
            print(f"[MODE] {run_label}")
            print(f"[KVTuner] Group choices: {group_choices}")
        else:
            run_label = "Quantized (KIVI) - Uniform"
            print(f"[MODE] {run_label}")
            print(f"[KIVI] Uniform Config: k_bits={cfg.k_bits}, v_bits={cfg.v_bits}")

        # Load the KIVI model, passing the layer_quant_map if it exists.
        model = KIVIModel.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            config=cfg,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=args.device_map,
            layer_quant_map=layer_quant_map,
        ).eval()

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"

    device = next(model.parameters()).device
    model_base_alloc = warmup_and_measure_base(model, tok, device)
    print(f"[GPU] Model base memory (after warmup): {fmt_bytes(model_base_alloc)}")

    # --- Define 3 different prompts for benchmarking ---
    prompts = [
        "What is the main difference between deep learning and machine learning?",
        "Write a Python function that calculates the factorial of a number using recursion.",
        "Tell me a short, futuristic story about a friendship between a human and an AI."
    ]

    # --- Lists to store results for summary ---
    all_results = []
    total_generated_tokens = 0
    total_decoding_time = 0.0

    print("\n" + "="*20 + " Starting Benchmark " + "="*20)

    for i, prompt in enumerate(prompts, 1):
        # --- Per-Request Timer Start ---
        t_request_start = time.time()
        
        print(f"\n--- Request #{i} ---")
        print(f"[PROMPT] {prompt}")

        inputs = tok(prompt, return_tensors="pt").to(device)
        
        torch.cuda.reset_peak_memory_stats(device)
        
        t_start_generation = time.time()
        with torch.inference_mode():
            out = model(**inputs, use_cache=True, return_dict=True)
            past = out.past_key_values
            
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            out = model(input_ids=next_ids, use_cache=True, past_key_values=past, return_dict=True)
            past = out.past_key_values

        t_first_token = time.time()
        
        generated_ids = [next_ids.item()]
        t_decode_start = time.time()
        
        for _ in range(1, args.max_new_tokens):
            next_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            if next_ids.item() == tok.eos_token_id:
                break
            
            with torch.inference_mode():
                out = model(input_ids=next_ids, use_cache=True, past_key_values=past, return_dict=True)
                past = out.past_key_values

            generated_ids.append(next_ids.item())

        t_decode_end = time.time()
        
        # --- Collect Metrics ---
        ttft = t_first_token - t_start_generation
        num_generated = len(generated_ids)
        # Total time from first token generation start to last token generation end
        total_generation_time = (t_decode_end - t_decode_start) + (t_first_token - t_start_generation)
        throughput = num_generated / total_generation_time if total_generation_time > 0 else 0
        
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_inference_memory = max(0, peak_alloc - model_base_alloc)
        
        output_text = tok.decode(generated_ids, skip_special_tokens=True)
        print(f"[RESPONSE] {output_text}")
        
        # --- Per-Request Timer End and Runtime Calculation ---
        t_request_end = time.time()
        request_runtime = t_request_end - t_request_start
        
        print("[METRICS]")
        print(f"  - Request Runtime:           {request_runtime:.2f} s")
        print(f"  - System Latency (TTFT):     {ttft * 1000:.2f} ms")
        print(f"  - System Throughput:         {throughput:.2f} tokens/s")
        print(f"  - Peak Inference Memory Footprint: {fmt_bytes(peak_inference_memory)}")

        all_results.append({
            "ttft": ttft * 1000, # Store TTFT in milliseconds
            "peak_inference_memory": peak_inference_memory, # Store peak memory
            "num_generated": num_generated,
            "total_generation_time": total_generation_time,
        })
        total_generated_tokens += num_generated
        total_decoding_time += total_generation_time

        del out, inputs, past, next_ids
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Aggregate Summary ----
    # --- Total Benchmark Timer End ---
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time

    print("\n" + "="*20 + f" Benchmark Summary ({run_label}) " + "="*20)
    
    # Calculate average TTFT (Time To First Token)
    avg_ttft = sum(r['ttft'] for r in all_results) / len(all_results)
    # Correctly calculate aggregate throughput
    aggregate_throughput = total_generated_tokens / total_decoding_time if total_decoding_time > 0 else 0
    # Find the maximum peak memory usage across all requests
    max_peak_inference_memory = max(r['peak_inference_memory'] for r in all_results)

    print(f"Total Benchmark Runtime: {total_runtime:.2f} s")
    print(f"Total requests: {len(prompts)}")
    print(f"Total tokens generated: {total_generated_tokens}")
    print("\n[AGGREGATE & PEAK METRICS]")
    print(f"  - Average System Latency (TTFT): {avg_ttft:.2f} ms")
    print(f"  - Aggregate System Throughput:   {aggregate_throughput:.2f} tokens/s (Total Generated Tokens / Total Generation Time)")
    print(f"  - Max Peak Inference Memory Footprint: {fmt_bytes(max_peak_inference_memory)}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()