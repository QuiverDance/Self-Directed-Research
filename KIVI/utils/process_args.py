# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Output model local path, do not set manually"}
    )
    k_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV_cache quantization bits."},
    )
    v_bits: Optional[int] = field(
        default=2,
        metadata={"help": "KV_cache quantization bits."},
    )
    # KVTuner: Add an argument to accept the layer-wise configuration file path.
    layer_quant_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the KVTuner layer-wise quantization config JSON file."},
    )
    k_quant_dim: Optional[str] = field(
        default='token',
        metadata={"help": "KV_cache quantization bits."},
    )
    v_quant_dim: Optional[str] = field(
        default='token',
        metadata={"help": "KV_cache quantization bits."},
    )
    group_size: Optional[int] = field(
        default=128,
        metadata={"help": "KV_cache quantization group size."},
    )
    residual_length: Optional[int] = field(
        default=128,
        metadata={"help": "KV_cache residual length."},
    )
    output_model_filename: Optional[str] = field(
        default="test-output", metadata={"help": "Output model relative manifold path"}
    )
    load_quant: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a quantized model"},
    )
    w_bit: Optional[int] = field(
        default=4,
        metadata={"help": "The model weight bit width."},
    )
    lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use LoRA"},
    )
    lora_mode: Optional[str] = field(
        default="q",
        metadata={"help": "LoRA mode"},
    )
    lora_r: Optional[int] = field(
        default=1,
        metadata={"help": "LoRA r"},
    )
    lora_alpha: Optional[float] = field(
        default=1.,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: Optional[float] = field(
        default=0.,
        metadata={"help": "LoRA dropout"},
    )
    


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default='c4',
        metadata={"help": "The dataset used for fine-tuning the model."},
    )
    eval_tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    tasks: Optional[str] = field(
        default='wikitext',
        metadata={"help": "The dataset used for evaluation."},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The batch size."},
    )
    num_fewshot: Optional[int] = field(
        default=0,
        metadata={"help": "The number of fewshot examples."},
    )
    output_path: Optional[str] = field(
        default='./outputs',
        metadata={"help": "The output path."},
    )
    e: Optional[bool] = field(
        default=False,
        metadata={"help": "Evaluate on LongBench-E."},
    )
    use_our_imp: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use our KV cache quantization implementation."},
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="./outputs")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    num_train_epochs: Optional[int] = field(default=1)
    n_train_samples: Optional[int] = field(default=None)
    n_eval_samples: Optional[int] = field(default=None)
    qat: Optional[bool] = field(default=False)
    exp_name: Optional[str] = field(default="test")

# KVTuner: Add a new function to parse the layer-wise quantization config file.
def load_and_process_kvtuner_config(config_path: str) -> tuple[any, any]:
    """
    Loads and processes a KVTuner-style layer-wise quantization configuration file.

    This function reads a JSON config.
    - If "enable" is true, it creates a layer-to-bits map and returns it with the group choices.
    - If "enable" is false, it returns a special marker "disabled".
    - If the file doesn't exist, it returns (None, None).

    Args:
        config_path: The file path to the JSON configuration.

    Returns:
        A tuple containing:
        - The processed configuration (a dict map, the string "disabled", or None).
        - The list of group choices if applicable, otherwise None.
    """
    if not config_path or not os.path.exists(config_path):
        return None, None

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check if the layer-wise quantization is enabled in the config.
    if not config.get("enable", False):
        print("[KVTuner] Layer-wise quantization is DISABLED in the config file. Model will run in native FP16 mode.")
        return "disabled", None

    # --- Parse the configuration ---
    quant_scheme = config["quant_scheme"]
    candidates_map = config["candidates_index_map"]
    
    layer_groups = config["groups"][quant_scheme]
    group_choices = config["group_choice"][quant_scheme]

    if len(layer_groups) != len(group_choices):
        raise ValueError(
            f"Mismatch between the number of layer groups ({len(layer_groups)}) "
            f"and group choices ({len(group_choices)}) in config file: {config_path}"
        )

    # --- Create the final layer-to-bits mapping ---
    layer_to_bits_map = {}
    for group_of_layers, choice_index in zip(layer_groups, group_choices):
        candidate_key = str(choice_index)
        if candidate_key not in candidates_map:
            raise KeyError(f"Candidate index '{candidate_key}' not found in candidates_index_map in {config_path}.")
        
        bits_config = candidates_map[candidate_key]
        for layer_index in group_of_layers:
            layer_to_bits_map[layer_index] = bits_config.copy()
    
    print(f"[KVTuner] Loaded and processed layer-wise quantization config from: {config_path}")
    return layer_to_bits_map, group_choices

def process_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_args.output_model_local_path = os.path.join(
        training_args.output_dir, "models", str(model_args.output_model_filename)
    )

    return model_args, data_args, training_args
