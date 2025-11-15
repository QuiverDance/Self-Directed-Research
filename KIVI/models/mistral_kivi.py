# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions andpython kivi_smoke_mistral.py

# limitations under the License.
""" PyTorch Mistral model."""
import inspect
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer

from transformers.models.mistral.configuration_mistral import *
from transformers.models.mistral.modeling_mistral import *
# --- KIVI compatibility helpers for HF 4.43+ cache API ---
try:
    from transformers.utils import is_flash_attn_2_available  # may not exist on old versions
except Exception:
    def is_flash_attn_2_available():
        return False

from quant.new_pack import (
    quant_and_pack_vcache,
    unpack_and_dequant_vcache,
)

DEBUG = False

def _kivi_get_past_length(past):
    if past is None:
        return 0

    if isinstance(past, (list, tuple)) and len(past) > 0:
        first = past[0]

        if isinstance(first, dict):
            shape = first.get("shape", None)
            if shape is not None and len(shape) >= 3:
                return int(shape[2])

        if isinstance(first, (list, tuple)) and len(first) > 0:
            last = first[-1]
            if isinstance(last, int):
                return last
            if hasattr(last, "numel") and last.numel() == 1:
                return int(last.item())

    # 2) HF 4.43+ Cache wrapper
    try:
        return past.get_seq_length()
    except Exception:
        return 0
# --- end helpers ---

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

# --------------------------------------------------------------------------------------
# ZipCache / Hybrid helper classes (FlashAttention path)
# --------------------------------------------------------------------------------------

class ZipCacheLayerState:
    """
    Lightweight per-layer state used by ZipCache / hybrid modes.

    This object tracks:
      * the prompt length (prefill),
      * the number of decode steps since the last saliency refresh,
      * optional tensors that mark unimportant token indices for keys/values,
      * and (optionally) per-token bitwidths for K/V.
    """

    def __init__(self, layer_idx: int, config: Optional[Dict[str, Any]] = None) -> None:
        self.layer_idx: int = int(layer_idx)

        # Total number of tokens in the initial prompt (prefill).
        self.prompt_length: int = 0
        # Number of decode steps processed since the last saliency refresh.
        self.decode_counter: int = 0

        # Optional tensor buffers for unimportant token indices.
        # Shape convention: [batch, num_heads, num_unimportant_tokens]
        self.unimportant_ids_k: Optional[torch.Tensor] = None
        self.unimportant_ids_v: Optional[torch.Tensor] = None

        # Optional scratch space to accumulate streaming scores during decode.
        self._streaming_scores: Optional[torch.Tensor] = None

        # Optional per-token bitwidth maps (filled after saliency is known).
        # Shape: [batch, num_heads, seq_len]
        self.bits_k: Optional[torch.Tensor] = None
        self.bits_v: Optional[torch.Tensor] = None

        # Keep a shallow copy of the ZipCache config if present (useful for debugging).
        self.config: Optional[Dict[str, Any]] = dict(config) if config is not None else None

    def reset_streaming(self) -> None:
        """
        Reset streaming-related state.

        This is typically called at the beginning of a new sequence or when the
        ZipCache controller decides to start a fresh streaming window.
        """
        self.decode_counter = 0
        self._streaming_scores = None
        # Per-token bit policy is tied to the current prefix; when streaming is
        # reset, we also drop any existing bit maps so that they will be recomputed.
        self.bits_k = None
        self.bits_v = None


class ZipCacheSaliencyControllerFlash:
    """
    Controller that computes and maintains token saliency statistics for
    ZipCache / hybrid modes in the FlashAttention path.

    This implementation follows the high-level idea of ZipCache:
      * During prefill, we select a small set of "probe" tokens and use their
        attention distributions over the whole prefix to estimate per-token
        saliency with a normalized attention metric.
      * During decode, we accumulate attention rows over time and, every
        `streaming_gap` steps, refresh the unimportant token sets based on
        the aggregated saliency.

    The saliency itself is stored in ZipCacheLayerState via unimportant_ids_*.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(cfg) if cfg is not None else {}

        # Configuration knobs taken from the kv_quant_config.json `zipcache` section.
        self.streaming_gap: int = int(cfg.get("streaming_gap", 100))
        self.prefill_keep_ratio: float = float(cfg.get("prefill_keep_ratio", 0.4))
        self.k_unimportant_ratio: float = float(cfg.get("k_unimportant_ratio", 0.5))
        self.v_unimportant_ratio: float = float(cfg.get("v_unimportant_ratio", 0.5))
        self.probe_ratio_recent: float = float(cfg.get("probe_ratio_recent", 0.05))
        self.probe_ratio_random: float = float(cfg.get("probe_ratio_random", 0.05))

        self._validate_ratios()

    def _validate_ratios(self) -> None:
        """Sanity-check that all ratio-like parameters are within [0, 1]."""
        assert 0.0 <= self.prefill_keep_ratio <= 1.0, "prefill_keep_ratio must be in [0, 1]"
        for name in ("k_unimportant_ratio", "v_unimportant_ratio", "probe_ratio_recent", "probe_ratio_random"):
            value = getattr(self, name)
            assert 0.0 <= value <= 1.0, f"{name} must be in [0, 1]"

    @torch.no_grad()
    def init_prefill_saliency(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        layer_state: "ZipCacheLayerState",
    ) -> None:
        """
        Initialize saliency information during the prefill phase.

        Args:
            query_states: Tensor of shape [batch, num_heads, seq_len, head_dim].
            key_states:   Tensor of shape [batch, num_heads, seq_len, head_dim].
            attention_mask: Optional causal/padding mask; currently only used
                            to ignore padded tokens when computing saliency.
            layer_state: Per-layer ZipCache state that will be updated in-place.
        """
        assert query_states.dim() == 4 and key_states.dim() == 4, (
            "ZipCache expects [batch, num_heads, seq_len, head_dim] tensors for Q/K."
        )
        batch_size, num_q_heads, seq_len, head_dim = query_states.shape
        _, num_kv_heads, seq_len_kv, head_dim_kv = key_states.shape
        assert seq_len_kv == seq_len and head_dim_kv == head_dim, (
            "ZipCache saliency expects query/key states to share [B, T, D] axes"
        )

        if num_q_heads == num_kv_heads:
            # Standard multi-head attention: one KV head per query head.
            num_heads = num_q_heads
            query_for_saliency = query_states
        else:
            assert (
                num_q_heads % num_kv_heads == 0
            ), f"Incompatible num_heads={num_q_heads} and num_kv_heads={num_kv_heads} for ZipCache saliency"
            group = num_q_heads // num_kv_heads
            # [B, H_q, T, D] -> [B, H_kv, group, T, D] -> mean over group → [B, H_kv, T, D]
            query_for_saliency = (
                query_states.reshape(batch_size, num_kv_heads, group, seq_len, head_dim)
                .mean(dim=2)
                .contiguous()
            )
            num_heads = num_kv_heads

        device = query_states.device

        if DEBUG:
            print(
                f"[DEBUG][ZipSaliency] prefill enter "
                f"B={batch_size} H={num_heads} T={seq_len} "
                f"prefill_keep_ratio={self.prefill_keep_ratio}",
                flush=True,
        )

        # Record the prompt length so that decode can distinguish new tokens
        # from the original prefix, and reset any streaming state.
        layer_state.prompt_length = seq_len
        layer_state.reset_streaming()

        # Degenerate cases: no tokens or "keep everything" => nothing to drop.
        if seq_len == 0 or self.prefill_keep_ratio >= 1.0:
            layer_state.unimportant_ids_k = None
            layer_state.unimportant_ids_v = None
            return

        # ------------------------------------------------------------------
        # 1) Select probe token indices (recent + random), as in ZipCache.
        # ------------------------------------------------------------------
        num_recent = int(seq_len * self.probe_ratio_recent)
        num_random = int(seq_len * self.probe_ratio_random)

        # Clamp to valid ranges.
        num_recent = max(min(num_recent, seq_len), 0)
        # Random probes sample from the earlier part that is not covered by "recent".
        num_random = max(min(num_random, max(seq_len - num_recent, 0)), 0)

        probe_indices: List[torch.Tensor] = []

        if num_recent > 0:
            recent_start = max(seq_len - num_recent, 0)
            recent_idx = torch.arange(recent_start, seq_len, device=device)
            probe_indices.append(recent_idx)

        if num_random > 0:
            # Sample random positions from [0, seq_len - num_recent).
            rand_src_len = max(seq_len - num_recent, 0)
            if rand_src_len > 0:
                perm = torch.randperm(rand_src_len, device=device)
                random_idx = perm[:num_random]
                probe_indices.append(random_idx)

        if not probe_indices:
            # Fallback: use the last token as a single probe.
            probe_idx = torch.tensor([seq_len - 1], device=device, dtype=torch.long)
        else:
            # Merge and sort to get a unique set of probe indices.
            probe_idx = torch.unique(torch.cat(probe_indices, dim=0)).sort().values

        # ------------------------------------------------------------------
        # 2) Compute attention of probe queries over the whole prefix: Q_probe @ K^T.
        # ------------------------------------------------------------------
        # query_states: [B, H, T, D] -> pick probe positions on the time dimension.
        q_probe = query_for_saliency.index_select(dim=2, index=probe_idx)  # [B, H, Q, D]
        # key_states: [B, H, T, D]; batched matmul → [B, H, Q, T].
        scores = torch.einsum("bhqd,bhkd->bhqk", q_probe, key_states) / math.sqrt(head_dim)
        if DEBUG:
            print(
                f"[DEBUG][ZipSaliency] scores computed shape={tuple(scores.shape)}",
                flush=True,
        )
        # ------------------------------------------------------------------
        # 3) Apply attention mask (if any) and softmax.
        # ------------------------------------------------------------------
        if attention_mask is not None:
            # Normalize various HF mask shapes (e.g. [B, T], [B, T, T], [B, 1, Q, T])
            # into a 2D key-validity mask [B, T], where True means "token is valid".
            am = attention_mask

            if am.dim() == 2:
                # Typical tokenizer mask: 1 for real tokens, 0 for padding.
                if am.dtype.is_floating_point:
                    key_mask_2d = am > 0
                else:
                    key_mask_2d = am.to(torch.bool)
            else:
                # Additive or boolean masks with extra query/head axes.
                # We interpret the last axis as "key" positions and treat a key as
                # valid if it is attendable for at least one query position.
                if am.dtype.is_floating_point:
                    # In additive masks, 0 usually means "allowed" and negative
                    # (e.g., -inf or -1e9) means "masked".
                    valid = am >= 0
                else:
                    valid = am.to(torch.bool)

                # Collapse all non-batch, non-key axes (query/head dimensions).
                # Example shapes:
                #   [B, T, T]       -> reduce over dim=1 -> [B, T]
                #   [B, 1, Q, T]    -> reduce over dims=1,2 -> [B, T]
                reduce_dims = tuple(range(1, valid.dim() - 1))
                if reduce_dims:
                    key_mask_2d = valid.any(dim=reduce_dims)
                else:
                    key_mask_2d = valid

            # Lift to [B, 1, 1, T] so it broadcasts over [B, H, Q, T].
            mask = key_mask_2d[:, None, None, :]  # [B, 1, 1, T]
            scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax over the last dimension to obtain attention distributions.
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32)  # [B, H, Q, T]

        # ------------------------------------------------------------------
        # 4) Compute ZipCache-style normalized saliency per token.
        # ------------------------------------------------------------------
        saliency = self._compute_normalized_saliency(attn, probe_idx, seq_len)  # [B, T]
        if DEBUG:
            print(
                f"[DEBUG][ZipSaliency] saliency computed shape={tuple(saliency.shape)}",
                flush=True,
            )
        # ------------------------------------------------------------------
        # 5) Select salient vs unimportant tokens based on prefill_keep_ratio.
        # ------------------------------------------------------------------
        unimportant_ids = self._select_unimportant_from_saliency(
            saliency=saliency,
            keep_ratio=self.prefill_keep_ratio,
            batch_size=batch_size,
            num_heads=num_heads,
            device=device,
        )

        if unimportant_ids is None:
            layer_state.unimportant_ids_k = None
            layer_state.unimportant_ids_v = None
            return

        if DEBUG:
            print(
                f"[DEBUG][ZipSaliency] prefill exit "
                f"unimp_k={None if layer_state.unimportant_ids_k is None else tuple(layer_state.unimportant_ids_k.shape)} "
                f"unimp_v={None if layer_state.unimportant_ids_v is None else tuple(layer_state.unimportant_ids_v.shape)}",
                flush=True,
            )

        layer_state.unimportant_ids_k = unimportant_ids
        # For now we treat K and V the same; if needed we can later decouple
        # them using k_unimportant_ratio / v_unimportant_ratio.
        layer_state.unimportant_ids_v = unimportant_ids.clone()

    def _compute_normalized_saliency(
        self,
        attn: torch.Tensor,
        probe_idx: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Approximate the ZipCache normalized attention saliency.

        Args:
            attn:      Tensor of shape [B, H, Q, T] containing attention scores
                       (after softmax) from Q probe tokens to all T tokens.
            probe_idx: Tensor of shape [Q] with the 0-based positions of probe
                       tokens within the sequence.
            seq_len:   Total prefix length T.

        Returns:
            saliency:  Tensor of shape [B, T] with per-token saliency scores.

        The normalization corrects the lower-triangular bias by dividing each
        token's accumulated attention score by the number of probe rows that
        can attend to that token.
        """
        B, H, Q, T = attn.shape
        assert T == seq_len, "Inconsistent sequence lengths when computing saliency"

        # Aggregate attention mass over heads and probe queries → [B, T].
        raw = attn.sum(dim=(1, 2))  # [B, T]

        # Build "visible count" for each token to correct the lower-triangular bias:
        # a token at position j is only visible to probe queries with index >= j.
        device = attn.device
        probe_idx_sorted = torch.sort(probe_idx).values.to(dtype=torch.long)

        visible_count = torch.zeros(T, device=device, dtype=torch.float32)
        for p in probe_idx_sorted.tolist():
            # Token indices 0..p are visible in the probe at position p.
            visible_count[: p + 1] += 1.0

        # Avoid division by zero: if a token is never visible to any probe,
        # we treat its visible_count as 1 to keep the raw score unchanged.
        visible_count = torch.clamp(visible_count, min=1.0)

        # Normalize: divide each token's total score by how many probe rows see it.
        saliency = raw / visible_count.unsqueeze(0)
        return saliency

    def _select_unimportant_from_saliency(
        self,
        saliency: torch.Tensor,
        keep_ratio: float,
        batch_size: int,
        num_heads: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Helper to convert a [B, T] saliency tensor into per-head unimportant indices.

        Args:
            saliency:  [B, T] saliency scores (higher is more important).
            keep_ratio: Fraction of tokens to keep as "salient". The remaining
                        (1 - keep_ratio) fraction will be marked unimportant.
            batch_size: Batch size B.
            num_heads:  Number of attention heads H.
            device:     Target device.

        Returns:
            unimportant_ids: Long tensor [B, H, L_u] with unimportant token indices,
                             or None if there are no unimportant tokens.
        """
        assert saliency.dim() == 2, "Expected [batch, seq_len] saliency tensor"
        B, T = saliency.shape
        assert B == batch_size, "Batch size mismatch in saliency selection"

        # Clamp keep_ratio to [0, 1] to avoid accidental misconfigurations.
        keep_ratio = max(min(float(keep_ratio), 1.0), 0.0)

        keep = int(T * keep_ratio)
        keep = max(min(keep, T), 0)

        if keep == 0:
            # Everything is unimportant.
            salient_mask = torch.zeros_like(saliency, dtype=torch.bool)
        elif keep >= T:
            # Everything is salient; nothing to drop.
            salient_mask = torch.ones_like(saliency, dtype=torch.bool)
        else:
            # Top-k per batch based on saliency.
            _, topk_idx = torch.topk(saliency, k=keep, dim=-1, largest=True, sorted=False)
            salient_mask = torch.zeros_like(saliency, dtype=torch.bool)
            salient_mask.scatter_(dim=-1, index=topk_idx, value=True)

        unimportant_mask = ~salient_mask          # [B, T]
        num_unimportant = unimportant_mask.sum(dim=-1)  # [B]
        max_Lu = int(num_unimportant.max().item())

        if max_Lu == 0:
            # No unimportant tokens in this batch.
            return None

        # ------------------------------------------------------------------
        # Build per-head index tensor [B, H, L_u] for unimportant tokens.
        # ------------------------------------------------------------------
        unimportant_ids = torch.zeros(
            batch_size,
            num_heads,
            max_Lu,
            device=device,
            dtype=torch.long,
        )

        for b in range(batch_size):
            idx = torch.nonzero(unimportant_mask[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                # This batch element has no unimportant tokens; leave zeros as-is.
                continue
            # If this batch has fewer unimportant tokens than the maximum, we pad
            # with the last index. Later stages can call `torch.unique` on these
            # indices if they need a set without duplicates.
            if idx.numel() < max_Lu:
                pad = idx[-1].expand(max_Lu - idx.numel())
                idx = torch.cat([idx, pad], dim=0)
            # Repeat the same unimportant token indices across all heads.
            unimportant_ids[b, :, :] = idx.unsqueeze(0).expand(num_heads, max_Lu)

        return unimportant_ids

    def update_streaming(
        self,
        attn_row: torch.Tensor,
        layer_state: "ZipCacheLayerState",
        num_kv_heads: Optional[int] = None,
    ) -> None:
        """
        Update streaming saliency signals during decode.

        Args:
            attn_row:    Tensor of shape [batch, num_heads, q_len, total_seq_len]
                         containing the attention distribution of the newly
                         generated tokens. In typical decode, q_len == 1.
            layer_state: Per-layer ZipCache state that will be updated in-place.

        Behavior:
          * Accumulate per-token attention mass over time in `layer_state._streaming_scores`.
          * Every `streaming_gap` decode steps, compute an aggregated saliency
            over the window and refresh the unimportant token sets according to
            k_unimportant_ratio / v_unimportant_ratio.
        """
        assert attn_row.dim() == 4, "Expected attention row as [batch, num_heads, q_len, total_seq_len]"
        B, H_attn, Q, T = attn_row.shape
        assert B >= 1 and H_attn >= 1 and Q >= 1 and T >= 1, "Invalid attention row shape"

        # When using grouped-query attention, the number of KV heads (H_kv)
        # can be smaller than the number of query heads in `attn_row`.
        # For ZipCache we only need per-token saliency, and the per-head index
        # tensor `unimportant_ids_*` is later consumed at KV-head granularity.
        num_heads = int(num_kv_heads) if num_kv_heads is not None else H_attn

        # Aggregate over heads and query positions → [B, T].
        contrib = attn_row.sum(dim=1).sum(dim=2)  # [B, T]

        # Initialize or resize the streaming buffer if needed.
        if layer_state._streaming_scores is None or layer_state._streaming_scores.shape != contrib.shape:
            layer_state._streaming_scores = torch.zeros_like(contrib)

        # Accumulate contributions and advance the decode counter.
        layer_state._streaming_scores += contrib
        layer_state.decode_counter += 1

        # Only refresh saliency every `streaming_gap` steps.
        if self.streaming_gap <= 0 or (layer_state.decode_counter % self.streaming_gap) != 0:
            return

        # ------------------------------------------------------------------
        # 1) Build a streaming saliency estimate over the current window.
        # ------------------------------------------------------------------
        saliency_stream = layer_state._streaming_scores / max(layer_state.decode_counter, 1)
        device = attn_row.device

        # ------------------------------------------------------------------
        # 2) Select unimportant tokens for K and V separately, based on the
        #    configured unimportant ratios. For K we keep (1 - k_unimportant_ratio)
        #    fraction as salient, and similarly for V.
        # ------------------------------------------------------------------
        keep_ratio_k = 1.0 - float(self.k_unimportant_ratio)
        keep_ratio_v = 1.0 - float(self.v_unimportant_ratio)

        unimp_k = self._select_unimportant_from_saliency(
            saliency=saliency_stream,
            keep_ratio=keep_ratio_k,
            batch_size=B,
            num_heads=num_heads,
            device=device,
        )
        unimp_v = self._select_unimportant_from_saliency(
            saliency=saliency_stream,
            keep_ratio=keep_ratio_v,
            batch_size=B,
            num_heads=num_heads,
            device=device,
        )

        # If V-specific selection fails (e.g., all salient), fall back to K's set.
        layer_state.unimportant_ids_k = unimp_k
        layer_state.unimportant_ids_v = unimp_v if unimp_v is not None else unimp_k

        # ------------------------------------------------------------------
        # 3) Reset the streaming window so that the next `streaming_gap` steps
        #    start a fresh accumulation.
        # ------------------------------------------------------------------
        layer_state._streaming_scores = None
        layer_state.decode_counter = 0


class ZipCacheKVCompressorFlash:
    """
    Simple policy helper for ZipCache-style mixed-bit KV compression
    in the FlashAttention path.

    NOTE:
      * In this stage we only implement the **bit assignment policy**,
        not the full packed-byte compression format.
      * Actual quant/dequant math is still handled by the existing
        KIVI / KVTuner Triton kernels that are already wired in
        MistralAttention_KIVI / MistralFlashAttention_KIVI.
      * The goal of this class is to:
          - decide, per layer and per token, which bitwidth to use
            for K/V (high vs low),
          - expose that decision in a simple Tensor form that the
            calling code (FlashAttention forward) can use when it
            calls the Triton quant kernels.
    """

    def __init__(
        self,
        quant_mode: str,
        zipcache_cfg: Optional[Dict[str, Any]],
        bit_candidates: Optional[List[int]] = None,
    ):
        # "zipcache_only" or "hybrid"
        self.quant_mode = quant_mode
        self.cfg = dict(zipcache_cfg) if zipcache_cfg is not None else {}

        # Bit candidate set used for "step down" (8 → 4 → 2 → 1).
        # Prefer explicitly passed candidates, fall back to config.
        if bit_candidates is not None:
            self.bit_candidates: List[int] = sorted(
                set(bit_candidates),
                reverse=True,
            )
        else:
            self.bit_candidates: List[int] = sorted(
                set(self.cfg.get("bit_candidates", [8, 4, 2, 1])),
                reverse=True,
            )

        # Default high/low bits when running in pure ZipCache mode.
        self.bits_high_k: int = int(self.cfg.get("bits_high_k", 8))
        self.bits_high_v: int = int(self.cfg.get("bits_high_v", 8))
        self.bits_low_k: int = int(self.cfg.get("bits_low_k", 4))
        self.bits_low_v: int = int(self.cfg.get("bits_low_v", 4))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _step_down_bits(self, base_bits: int) -> int:
        """
        Given a base bitwidth (e.g., 8), return the "next lower" bitwidth
        from self.bit_candidates (e.g., 4). If base_bits is already the
        lowest, return it unchanged.
        """
        # bit_candidates is sorted in descending order, e.g. [8, 4, 2, 1]
        if base_bits not in self.bit_candidates:
            # If the layer's bits are not in the candidate set, fall back
            # to the closest smaller candidate; if none, keep as-is.
            smaller = [b for b in self.bit_candidates if b < base_bits]
            return smaller[0] if smaller else base_bits

        idx = self.bit_candidates.index(base_bits)
        if idx + 1 < len(self.bit_candidates):
            return self.bit_candidates[idx + 1]
        return base_bits

    def _get_base_bits_for_layer(
        self,
        layer_idx: int,
        layer_quant_bits: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """
        Decide the *base* bitwidths (before ZipCache down-grading) for
        K and V at a given layer.

        - hybrid:
            * If layer_quant_bits is provided, use that.
            * Else, ask kvtuner_quantizer.get_bits(layer_idx) when available.
            * Else, fall back to the highest bit in bit_candidates.
        - zipcache_only:
            * Use cfg.bits_high_k / cfg.bits_high_v.
        """
        if self.quant_mode == "hybrid":
            # 1) Prefer explicit layer_quant_bits.
            if layer_quant_bits is not None:
                base_k, base_v = layer_quant_bits
            else:
                # 2) Fallback: use the highest candidate bitwidth.
                base_k = base_v = self.bit_candidates[0]
        else:
            # zipcache_only: fixed high bits from config
            base_k, base_v = self.bits_high_k, self.bits_high_v

        return int(base_k), int(base_v)

    # ------------------------------------------------------------------ #
    # Public API: bit assignment
    # ------------------------------------------------------------------ #
    def assign_bits_for_tokens(
        self,
        layer_idx: int,
        zip_state: "ZipCacheLayerState",
        layer_quant_bits: Optional[Tuple[int, int]],
        seq_len: int,
        num_kv_heads: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.int32,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-token bit assignment for K/V at this layer.

        Args:
            layer_idx:
                Index of this decoder layer.
            zip_state:
                ZipCacheLayerState with unimportant_ids_k/v.
            layer_quant_bits:
                Optional (bits_k, bits_v) from KVTuner's layer_quant_map.
            seq_len:
                Total KV sequence length T at this layer.
            num_kv_heads:
                Number of key/value heads H_kv.
            batch_size:
                Batch size B.
            device:
                Device where the bit tensors should be allocated.
            dtype:
                Integer dtype for the bit tensors (default: int32).

        Returns:
            {
              "bits_k": [B, H_kv, T] int tensor,
              "bits_v": [B, H_kv, T] int tensor,
              "base_bits_k": int,
              "base_bits_v": int,
              "low_bits_k": int,
              "low_bits_v": int,
            }
        """
        base_bits_k, base_bits_v = self._get_base_bits_for_layer(
            layer_idx=layer_idx,
            layer_quant_bits=layer_quant_bits,
        )

        if self.quant_mode == "zipcache_only":
            low_bits_k = self.bits_low_k
            low_bits_v = self.bits_low_v
        else:
            # hybrid: down-grade by one step from the base bits
            low_bits_k = self._step_down_bits(base_bits_k)
            low_bits_v = self._step_down_bits(base_bits_v)

        # Start with all tokens using the base bitwidth.
        bits_k = torch.full(
            (batch_size, num_kv_heads, seq_len),
            fill_value=base_bits_k,
            dtype=dtype,
            device=device,
        )
        bits_v = torch.full(
            (batch_size, num_kv_heads, seq_len),
            fill_value=base_bits_v,
            dtype=dtype,
            device=device,
        )

        # Apply ZipCache's "non-salient" indices to *lower* the bitwidth.
        # zip_state.unimportant_ids_* are expected to have shape [B, H_kv, L_u]
        # with entries in [0, T).
        if zip_state is not None and zip_state.unimportant_ids_k is not None:
            # Clamp indices just in case; invalid indices are ignored.
            idx_k = zip_state.unimportant_ids_k.clamp_(0, seq_len - 1)
            bits_k.scatter_(-1, idx_k, low_bits_k)

        if zip_state is not None and zip_state.unimportant_ids_v is not None:
            idx_v = zip_state.unimportant_ids_v.clamp_(0, seq_len - 1)
            bits_v.scatter_(-1, idx_v, low_bits_v)

        return {
            "bits_k": bits_k,
            "bits_v": bits_v,
            "base_bits_k": base_bits_k,
            "base_bits_v": base_bits_v,
            "low_bits_k": low_bits_k,
            "low_bits_v": low_bits_v,
        }

    # ------------------------------------------------------------------ #
    # Public API: compress / decompress (thin wrappers for now)
    # ------------------------------------------------------------------ #
    def _split_and_quant(
        self,
        tensor: torch.Tensor,
        unimportant_ids: Optional[torch.Tensor],
        bits_high: int,
        bits_low: int,
        group_size: int,
    ) -> Dict[str, Any]:
        """
        Split tokens into important / unimportant by indices, then quantize
        each group with different bitwidths using the existing KIVI kernels.

        tensor:        [B, H_kv, T, D]
        unimportant_ids: [B, H_kv, L_u] or None
        """
        B, H, T, D = tensor.shape
        device = tensor.device

        out: Dict[str, Any] = {}
        out["bits_high"] = int(bits_high)
        out["bits_low"] = int(bits_low)
        out["unimportant_ids"] = (
            unimportant_ids.detach().clone() if unimportant_ids is not None else None
        )
        out["shape"] = (B, H, T, D)
        out["group_size"] = int(group_size)

        # No ZipCache saliency: all tokens use high bits.
        if unimportant_ids is None or unimportant_ids.numel() == 0:
            code_high, scale_high, mn_high = quant_and_pack_vcache(
                tensor, group_size=group_size, bits=bits_high
            )
            out["mode"] = "all_high"
            out["code_high"] = code_high.detach()
            out["scale_high"] = scale_high.detach()
            out["mn_high"] = mn_high.detach()
            out["code_low"] = None
            out["scale_low"] = None
            out["mn_low"] = None
            return out

        # Build important mask and per-(B,H) important indices.
        mask_important = torch.ones(
            (B, H, T), dtype=torch.bool, device=device
        )
        # Clamp indices for safety
        unimportant_ids = unimportant_ids.clamp_(0, T - 1)
        mask_important.scatter_(2, unimportant_ids, False)

        # Nonzero returns (b, h, t), ordered in row-major order.
        important_flat = torch.nonzero(mask_important, as_tuple=False)
        # Shape: [B*H*(T-Lu), 3] -> take t and reshape back to [B, H, T-Lu]
        important_t = important_flat[:, 2].view(B, H, -1)

        # Gather important / unimportant tokens
        idx_imp = important_t.unsqueeze(-1).expand(-1, -1, -1, D)   # [B,H,T_imp,D]
        idx_unimp = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, D)  # [B,H,L_u,D]

        tensor_high = torch.gather(tensor, dim=2, index=idx_imp)
        tensor_low = torch.gather(tensor, dim=2, index=idx_unimp)

        # Quantize high / low parts separately
        code_high, scale_high, mn_high = quant_and_pack_vcache(
            tensor_high, group_size=group_size, bits=bits_high
        )
        code_low, scale_low, mn_low = quant_and_pack_vcache(
            tensor_low, group_size=group_size, bits=bits_low
        )

        out["mode"] = "mixed"
        out["code_high"] = code_high.detach()
        out["scale_high"] = scale_high.detach()
        out["mn_high"] = mn_high.detach()
        out["code_low"] = code_low.detach()
        out["scale_low"] = scale_low.detach()
        out["mn_low"] = mn_low.detach()
        return out

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        zip_state: "ZipCacheLayerState",
        layer_idx: int,
        layer_quant_bits: Optional[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """
        Real KV "compression" for FlashAttention + ZipCache.

        - Decide base / low bits per mode (zipcache_only / hybrid).
        - Split tokens into important / unimportant by saliency.
        - Quantize important tokens with high bits, unimportant with low bits.
        - Store packed blobs + index meta so we can fully reconstruct later.

        key_states, value_states: [B, H_kv, T, D] (FP16/BF16).
        """
        if key_states is None or value_states is None:
            raise ValueError(
                f"[ZipCacheKVCompressorFlash] compress called with None K/V at layer {layer_idx}"
            )

        B, H, T, D = key_states.shape
        device = key_states.device

        # Decide base/low bits for this layer + current saliency state.
        # We reuse the public bit-assignment helper so that base/low bits
        # stay consistent between "policy only" and "real compression".
        bits_info = self.assign_bits_for_tokens(
            layer_idx=layer_idx,
            zip_state=zip_state,
            layer_quant_bits=layer_quant_bits,
            seq_len=T,
            num_kv_heads=H,
            batch_size=B,
            device=device,
        )
        base_bits_k = bits_info["base_bits_k"]
        base_bits_v = bits_info["base_bits_v"]
        low_bits_k = bits_info["low_bits_k"]
        low_bits_v = bits_info["low_bits_v"]

        # Snapshot indices at compression time (zip_state may evolve later).
        unimportant_ids_k = (
            zip_state.unimportant_ids_k.detach().clone()
            if zip_state is not None and zip_state.unimportant_ids_k is not None
            else None
        )
        unimportant_ids_v = (
            zip_state.unimportant_ids_v.detach().clone()
            if zip_state is not None and zip_state.unimportant_ids_v is not None
            else None
        )

        # Use V-style packing (group along last dim), which matches KIVI v-cache kernels.
        group_size = int(self.cfg.get("group_size", 32))

        k_blob = self._split_and_quant(
            key_states, unimportant_ids_k, base_bits_k, low_bits_k, group_size
        )
        v_blob = self._split_and_quant(
            value_states, unimportant_ids_v, base_bits_v, low_bits_v, group_size
        )

        meta = {
            "layer_idx": int(layer_idx),
            "quant_mode": self.quant_mode,
            "layer_quant_bits": layer_quant_bits,
            "prompt_length": int(getattr(zip_state, "prompt_length", 0)),
            "decode_counter": int(getattr(zip_state, "decode_counter", 0)),
        }

        kv_blob = {
            "shape": (B, H, T, D),
            "key": k_blob,
            "value": v_blob,
            "base_bits_k": int(base_bits_k),
            "low_bits_k": int(low_bits_k),
            "base_bits_v": int(base_bits_v),
            "low_bits_v": int(low_bits_v),
            "meta": meta,
        }
        return kv_blob

    def _dequant_and_merge(
        self,
        blob: Dict[str, Any],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Inverse of `_split_and_quant`:

        - Dequantize high / low groups separately.
        - Scatter them back to original token order using stored indices.
        """
        mode = blob["mode"]
        B, H, T, D = blob["shape"]
        group_size = blob["group_size"]
        device = blob["code_high"].device if blob["code_high"] is not None else "cuda"

        bits_high = int(blob["bits_high"])
        bits_low = int(blob["bits_low"])
        unimportant_ids = blob["unimportant_ids"]
        out = torch.empty((B, H, T, D), device=device, dtype=dtype)

        if mode == "all_high":
            # Dequantize entire sequence with high bits.
            data = unpack_and_dequant_vcache(
                blob["code_high"],
                blob["scale_high"],
                blob["mn_high"],
                group_size=group_size,
                bits=bits_high,
            )
            return data.to(dtype=dtype)

        if unimportant_ids is None:
            raise ValueError("[ZipCacheKVCompressorFlash] mixed mode but unimportant_ids is None")

        # Rebuild mask & important indices
        mask_important = torch.ones((B, H, T), dtype=torch.bool, device=device)
        unimportant_ids = unimportant_ids.clamp_(0, T - 1)
        mask_important.scatter_(2, unimportant_ids, False)

        important_flat = torch.nonzero(mask_important, as_tuple=False)
        important_t = important_flat[:, 2].view(B, H, -1)  # [B, H, T_imp]

        # Dequantize groups
        high = unpack_and_dequant_vcache(
            blob["code_high"],
            blob["scale_high"],
            blob["mn_high"],
            group_size=group_size,
            bits=bits_high,
        ).to(dtype=dtype)
        low = unpack_and_dequant_vcache(
            blob["code_low"],
            blob["scale_low"],
            blob["mn_low"],
            group_size=group_size,
            bits=bits_low,
        ).to(dtype=dtype)

        # Scatter high / low back into [B,H,T,D]
        idx_imp = important_t.unsqueeze(-1).expand(-1, -1, -1, D)
        idx_unimp = unimportant_ids.unsqueeze(-1).expand(-1, -1, -1, D)

        out.scatter_(2, idx_imp, high)
        out.scatter_(2, idx_unimp, low)
        return out

    def decompress(
        self,
        kv_blob: Any,
        zip_state: "ZipCacheLayerState",
        layer_idx: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse of `compress`.

        kv_blob: dict as returned by `compress` (or legacy tuple for backward compat).
        """
        if kv_blob is None:
            raise ValueError(
                f"[ZipCacheKVCompressorFlash] decompress called with kv_blob=None at layer {layer_idx}"
            )

        # Backward compatibility: old tuple (key_fp, value_fp, meta)
        if isinstance(kv_blob, tuple) and len(kv_blob) == 3:
            key_fp, value_fp, _meta = kv_blob
            return key_fp.to(dtype=dtype), value_fp.to(dtype=dtype)

        key_blob = kv_blob["key"]
        value_blob = kv_blob["value"]

        key = self._dequant_and_merge(key_blob, dtype=dtype)
        value = self._dequant_and_merge(value_blob, dtype=dtype)
        return key, value

# --------------------------------------------------------------------------------------

_CONFIG_FOR_DOC = "MistralConfig"

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class MistralAttention_KIVI(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig, nbits_key: int, nbits_value: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # KVTuner: Set k_bits and v_bits from the passed arguments, not the global config.
        # This is the core change for applying layer-wise quantization.
        self.k_bits = nbits_key
        self.v_bits = nbits_value

        self.group_size = config.group_size
        self.residual_length = config.residual_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        # Ensure position_ids for window [kv_seq_len-q_len, kv_seq_len)
        if position_ids is None:
            start_pos = kv_seq_len - q_len
            position_ids = torch.arange(start_pos, kv_seq_len, device=value_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]

            if key_states_quant_trans is not None:
                # import ipdb; ipdb.set_trace()
                key_states_quant_trans_repeat = repeat_kv_quant(key_states_quant_trans, self.num_key_value_groups)
                key_scale_trans_repeat = repeat_kv_quant(key_scale_trans, self.num_key_value_groups)
                key_mn_trans_repeat = repeat_kv_quant(key_mn_trans, self.num_key_value_groups)
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states.contiguous(), key_states_quant_trans_repeat, 
                                key_scale_trans_repeat, key_mn_trans_repeat, self.k_bits) # key_states_quant_trans, key_scale_trans, key_mn_trans need to be repeated
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            
            key_states_full_repeat = repeat_kv(key_states_full, self.num_key_value_groups)
            att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3)) # key_states_full need to be repeated
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output = torch.matmul(attn_weights, value_states_full_repeat) # value_states_full need to be repeated
            else:
                value_states_quant_repeat = repeat_kv_quant(value_states_quant, self.num_key_value_groups)
                value_scale_repeat = repeat_kv_quant(value_scale, self.num_key_value_groups)
                value_mn_repeat = repeat_kv_quant(value_mn, self.num_key_value_groups)
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length].contiguous(), value_states_quant_repeat, 
                                                value_scale_repeat, value_mn_repeat, self.v_bits) # value_states_quant, value_scale, value_mn need to be repeated
                
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full_repeat) # value_states_full need to be repeated

            if value_states_full.shape[-2] > self.residual_length:
                assert value_states_full.shape[-2] == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn
        
        else:

            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
        
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, 
                                        key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states) 
        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len) if use_cache else None
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralFlashAttention_KIVI(MistralAttention_KIVI):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(
        self,
        config: MistralConfig,
        nbits_key: int,
        nbits_value: int,
        layer_idx: int,
        quant_mode: Optional[str] = None,
        zipcache_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the FlashAttention-based attention block used by KIVI.

        In addition to the base MistralAttention initialization, this constructor
        keeps track of:
          * the layer index (for debugging and ZipCache policies),
          * the quantization mode (kvtuner_only / zipcache_only / hybrid),
          * the ZipCache configuration dict,
          * and optional ZipCache helper objects used in FlashAttention paths.
        """
        # Keep the parent's initialization (this sets up projections and KV quant parameters).
        super().__init__(config=config, nbits_key=nbits_key, nbits_value=nbits_value)

        # Store basic metadata for this attention block.
        self.layer_idx = layer_idx
        # quant_mode is controlled by the top-level config / loader.
        self.quant_mode = quant_mode or getattr(config, "quant_mode", "kvtuner_only")
        # Normalize to a plain dict for easier use later on.
        self.zipcache_config = dict(zipcache_config) if zipcache_config is not None else None

        # This class should only be used for quantized runs, not baseline FP16.
        assert self.quant_mode in ("kvtuner_only", "zipcache_only", "hybrid"), (
            f"MistralFlashAttention_KIVI should not be instantiated in mode '{self.quant_mode}'. "
            "Baseline FP16 runs must use the original MistralAttention implementation."
        )

        # Initialize ZipCache helpers only when explicitly requested.
        if self.quant_mode in ("zipcache_only", "hybrid"):
            # Per-layer state that tracks prompt length and streaming statistics.
            self.zip_state = ZipCacheLayerState(layer_idx=self.layer_idx, config=self.zipcache_config)
            # Saliency controller for FlashAttention-based ZipCache.
            self.zip_saliency = ZipCacheSaliencyControllerFlash(self.zipcache_config)
            # KV bitwidth policy helper. For now we seed it with the current layer-wise
            # base bitwidths; in later steps we will connect it to the full KVTuner
            # candidate set if available.
            self.zip_compressor = ZipCacheKVCompressorFlash(
                quant_mode=self.quant_mode,
                zipcache_cfg=self.zipcache_config,
                bit_candidates=None,
            )
        else:
            # In plain KVTuner-only mode we do not need any ZipCache helpers.
            self.zip_state = None
            self.zip_saliency = None
            self.zip_compressor = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # Simple heuristic: if there is no past_key_value, we treat this call
        # as the prefill phase; otherwise we are in the decode phase.
        is_prefill = past_key_value is None

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # For ZipCache modes we keep track of the full KV sequence that will be stored back into the cache.
        key_states_all: Optional[torch.Tensor] = None
        value_states_all: Optional[torch.Tensor] = None

        kv_seq_len = key_states.shape[-2]
        if use_cache and past_key_value is not None:
            if self.quant_mode == "kvtuner_only":
                # KVTuner-only cache layout: tuple with previous length in the last slot.
                kv_seq_len += past_key_value[-1]
            else:
                # ZipCache / hybrid: past_key_value is a kv_blob produced by ZipCacheKVCompressorFlash.
                # It stores the history length in the "shape" metadata as (B, H_kv, T_hist, D).
                if isinstance(past_key_value, dict) and "shape" in past_key_value:
                    kv_seq_len += int(past_key_value["shape"][2])
                else:
                    # Backward compatibility: fallback to tuple-style (key_hist, value_hist, meta)
                    kv_seq_len += int(past_key_value[0].shape[-2])

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        # ------------------------------------------------------------------
        # ZipCache / hybrid: during prefill, initialize saliency and compute
        # per-token bitwidths for this layer.
        #
        # This does NOT change the attention computation itself; it only
        # updates ZipCacheLayerState with:
        #   * prompt_length
        #   * unimportant_ids_k/v
        #   * bits_k / bits_v
        # ------------------------------------------------------------------
        if self.quant_mode in ("zipcache_only", "hybrid") and is_prefill:
            # These attributes are created in __init__ when ZipCache is enabled.
            assert getattr(self, "zip_saliency", None) is not None, "zip_saliency must be initialized in ZipCache modes"
            assert getattr(self, "zip_state", None) is not None, "zip_state must be initialized in ZipCache modes"
            assert getattr(self, "zip_compressor", None) is not None, "zip_compressor must be initialized in ZipCache modes"

            if DEBUG:
                print(
                    f"[DEBUG][Zip] prefill start "
                    f"layer={self.layer_idx} "
                    f"q={tuple(query_states.shape)} "
                    f"k={tuple(key_states.shape)}",
                    flush=True,
            )

            # 1) Use probe tokens + normalized attention to estimate token saliency
            #    and fill layer_state.unimportant_ids_k/v.
            self.zip_saliency.init_prefill_saliency(
                query_states=query_states,
                key_states=key_states,
                attention_mask=attention_mask,
                layer_state=self.zip_state,
            )

            if DEBUG:
                print(
                    f"[DEBUG][Zip] prefill done layer={self.layer_idx}",
                    flush=True,
            )

            # 2) Convert saliency into a per-token bitwidth policy.
            #    For zipcache_only mode, this uses bits_high_k/v and bits_low_k/v
            #    from the ZipCache config. For hybrid, it uses the KVTuner base
            #    bits as "high" and the next lower candidate as "low".
            bits_info = self.zip_compressor.assign_bits_for_tokens(
                layer_idx=self.layer_idx,
                zip_state=self.zip_state,
                layer_quant_bits=(self.k_bits, self.v_bits),
                seq_len=kv_seq_len,
                num_kv_heads=self.num_key_value_heads,
                batch_size=bsz,
                device=hidden_states.device,
            )

            bits_k = bits_info["bits_k"]
            bits_v = bits_info["bits_v"]

            if DEBUG:
                print(
                    f"[DEBUG][Zip] bits assigned "
                    f"layer={self.layer_idx} "
                    f"bits_k_shape={tuple(self.zip_state.bits_k.shape) if self.zip_state.bits_k is not None else None} ",
                    flush=True,
                )

            # 3) Store the policy in the layer state so that the KV quantization
            #    path (or later debug tooling) can reuse it.
            self.zip_state.bits_k = bits_k
            self.zip_state.bits_v = bits_v

            # ------------------------------------------------------------------
            # Extra safety: verify that the per-token policy matches the current
            # prefix length and head configuration.
            # ------------------------------------------------------------------
            assert bits_k.shape[0] == bsz and bits_k.shape[1] == self.num_key_value_heads, (
                f"[ZipCache] bits_k batch/head mismatch: got {tuple(bits_k.shape)}, "
                f"expected (B={bsz}, H_kv={self.num_key_value_heads}, T=kv_seq_len)"
            )
            assert bits_k.shape == bits_v.shape, (
                f"[ZipCache] bits_k/bits_v shape mismatch: "
                f"{tuple(bits_k.shape)} vs {tuple(bits_v.shape)}"
            )
            # Prompt length should be aligned with the KV sequence length at prefill.
            assert self.zip_state.prompt_length == kv_seq_len, (
                f"[ZipCache] prompt_length ({self.zip_state.prompt_length}) "
                f"!= kv_seq_len ({kv_seq_len}) at prefill."
            )

        # use_sliding_windows = (
        #     _flash_supports_window_size
        #     and hasattr(self.config, "sliding_window") is not None
        #     and kv_seq_len > self.config.sliding_window
        # )

        # if not _flash_supports_window_size:
        #     logger.warning_once(
        #         "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
        #         " make sure to upgrade flash-attn library."
        #     )

        if use_cache and past_key_value is not None:
            if self.quant_mode == "kvtuner_only":
                # Original KVTuner-only cache layout: tuple of quantized/full K/V + stats + length.
                key_states_quant_trans = past_key_value[0]
                key_states_full = past_key_value[1]
                key_scale_trans = past_key_value[2]
                key_mn_trans = past_key_value[3]
                value_states_quant = past_key_value[4]
                value_states_full = past_key_value[5]
                value_scale = past_key_value[6]
                value_mn = past_key_value[7]
            else:
                # ZipCache-only / hybrid: past_key_value is the kv_blob created by ZipCacheKVCompressorFlash.
                # We reconstruct the full-precision K/V history and feed it into the same
                # downstream attention logic, but we do NOT reuse the KVTuner tuple layout.
                assert getattr(self, "zip_compressor", None) is not None, "zip_compressor must be initialized in ZipCache modes"
                assert getattr(self, "zip_state", None) is not None, "zip_state must be initialized in ZipCache modes"

                key_hist, value_hist = self.zip_compressor.decompress(
                    kv_blob=past_key_value,
                    zip_state=self.zip_state,
                    layer_idx=self.layer_idx,
                    dtype=key_states.dtype,
                )

                # No quantized prefix for ZipCache path; everything is treated as "full".
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
                key_states_full = key_hist

                value_states_quant = None
                value_scale = None
                value_mn = None
                value_states_full = value_hist

                # Full sequence that should be written back into the ZipCache after this step.
                key_states_all = torch.cat([key_hist, key_states], dim=2)
                value_states_all = torch.cat([value_hist, value_states], dim=2)

            # import ipdb; ipdb.set_trace()

            if key_states_quant_trans is not None:
                key_states_quant_trans_repeat = repeat_kv_quant(key_states_quant_trans, self.num_key_value_groups)
                key_scale_trans_repeat = repeat_kv_quant(key_scale_trans, self.num_key_value_groups)
                key_mn_trans_repeat = repeat_kv_quant(key_mn_trans, self.num_key_value_groups)
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states.contiguous(), key_states_quant_trans_repeat, 
                                key_scale_trans_repeat, key_mn_trans_repeat, self.k_bits) # key_states_quant_trans, key_scale_trans, key_mn_trans need to be repeated
            else:
                att_qkquant = None
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            key_states_full_repeat = repeat_kv(key_states_full, self.num_key_value_groups)
            att_qkfull = torch.matmul(query_states, key_states_full_repeat.transpose(2, 3))

            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # ZipCache / hybrid: during decode, feed the attention distribution
            # of the newly generated token into the streaming saliency controller.
            # For prefill (q_len > 1) we only initialize saliency once above.
            if self.quant_mode in ("zipcache_only", "hybrid") and not is_prefill:
                assert getattr(self, "zip_saliency", None) is not None, "zip_saliency must be initialized in ZipCache modes"
                assert getattr(self, "zip_state", None) is not None, "zip_state must be initialized in ZipCache modes"
                
                if DEBUG:
                    print(
                        f"[DEBUG][Zip] streaming update start "
                        f"layer={self.layer_idx} "
                        f"attn_row={tuple(attn_weights.shape)} "
                        f"decode_counter={self.zip_state.decode_counter}",
                        flush=True,
                )

                # attn_weights has shape [batch, num_heads, q_len, total_seq_len].
                # In typical decode, q_len == 1, so this is a single "row" per head.
                self.zip_saliency.update_streaming(
                    attn_row=attn_weights,
                    layer_state=self.zip_state,
                    num_kv_heads=self.num_key_value_heads,
                )

                if DEBUG:
                    print(
                        f"[DEBUG][Zip] streaming update done "
                        f"layer={self.layer_idx} "
                        f"decode_counter={self.zip_state.decode_counter}",
                        flush=True,
                    )
                
            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output = torch.matmul(attn_weights, value_states_full_repeat)
            else:
                value_states_quant_repeat = repeat_kv_quant(value_states_quant, self.num_key_value_groups)
                value_scale_repeat = repeat_kv_quant(value_scale, self.num_key_value_groups)
                value_mn_repeat = repeat_kv_quant(value_mn, self.num_key_value_groups)
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length].contiguous(), value_states_quant_repeat, 
                                                value_scale_repeat, value_mn_repeat, self.v_bits) # value_states_quant, value_scale, value_mn need to be repeated
                
                value_states_full_repeat = repeat_kv(value_states_full, self.num_key_value_groups)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full_repeat) # value_states_full need to be repeated

            attn_output = attn_output.transpose(1, 2).contiguous()

            if self.quant_mode == "kvtuner_only" and value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn


        else:
            # print(f"kivi with flash! {self.k_bits}")

            key_states_repeat = repeat_kv(key_states, self.num_key_value_groups)
            value_states_repeat = repeat_kv(value_states, self.num_key_value_groups)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                # Handle the case where the model is quantized
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states_repeat = key_states_repeat.to(target_dtype)
                value_states_repeat = value_states_repeat.to(target_dtype)

            if DEBUG:
                print(
                    f"[DEBUG][FlashKIVI] layer={self.layer_idx} "
                    f"calling flash_attn q={tuple(query_states.shape)} "
                    f"k_rep={tuple(key_states_repeat.shape)} "
                    f"v_rep={tuple(value_states_repeat.shape)} "
                    f"q_len={q_len} kv_seq_len={kv_seq_len}",
                flush=True,
            )
            
            attn_output = self._flash_attention_forward(
                query_states.transpose(1, 2), key_states_repeat.transpose(1, 2), 
                value_states_repeat.transpose(1, 2), None, q_len, dropout=0.0
            )

            if DEBUG:
                print(
                    f"[DEBUG][FlashKIVI] layer={self.layer_idx} flash_attn done",
                    flush=True,
                )

            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
            
            if self.quant_mode in ("zipcache_only", "hybrid"):
                # For ZipCache prefill, full prefix is exactly the current key/value.
                key_states_all = key_states
                value_states_all = value_states
            
        if use_cache:
            if self.quant_mode == "kvtuner_only":
                # Keep the original KVTuner cache layout.
                past_key_value = (
                    key_states_quant_trans,
                    key_states_full,
                    key_scale_trans,
                    key_mn_trans,
                    value_states_quant,
                    value_states_full,
                    value_scale,
                    value_mn,
                    kv_seq_len,
                )
            else:
                # ZipCache-only / hybrid: store a ZipCache kv_blob instead of the KVTuner tuple.
                # If we already built full KV (key_states_all/value_states_all), use that.
                # Otherwise fall back to the current key/value states.
                ks = key_states_all if key_states_all is not None else key_states
                vs = value_states_all if value_states_all is not None else value_states

                assert getattr(self, "zip_compressor", None) is not None, "zip_compressor must be initialized in ZipCache modes"
                assert getattr(self, "zip_state", None) is not None, "zip_state must be initialized in ZipCache modes"

                past_key_value = self.zip_compressor.compress(
                    key_states=ks,
                    value_states=vs,
                    zip_state=self.zip_state,
                    layer_idx=self.layer_idx,
                    layer_quant_bits=(self.k_bits, self.v_bits),
                )
        else:
            past_key_value = None

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        # if not output_attentions:
        attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                if DEBUG:
                    print(
                        f"[DEBUG][FlashCore] no_pad q={tuple(query_states.shape)} "
                        f"k={tuple(key_states.shape)} v={tuple(value_states.shape)}",
                        flush=True,
                    )
            
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                )

                if DEBUG:
                    print("[DEBUG][FlashCore] flash_attn_func done (no_pad)", flush=True)
            else:
                if DEBUG:
                    print(
                        f"[DEBUG][FlashCore] sliding q={tuple(query_states.shape)} "
                        f"k={tuple(key_states.shape)} v={tuple(value_states.shape)}",
                    flush=True,
                )
                
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )
                if DEBUG:
                    print("[DEBUG][FlashCore] flash_attn_func done (sliding)", flush=True)

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MistralDecoderLayer_KIVI(nn.Module):
    def __init__(self, config: MistralConfig, nbits_key: int, nbits_value: int, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Remember the layer index for debugging / ZipCache policies.
        self.layer_idx = layer_idx

        # Read quantization mode and ZipCache config from the shared config.
        quant_mode = getattr(config, "quant_mode", "kvtuner_only")
        zipcache_config = getattr(config, "zipcache_config", None)

        # KVTuner: Pass the layer-specific nbits to the attention module.
        if not getattr(config, "use_flash", False):
            # Non-flash path keeps the original KIVI attention implementation.
            self.self_attn = MistralAttention_KIVI(
                config=config,
                nbits_key=nbits_key,
                nbits_value=nbits_value,
            )
        else:
            # FlashAttention path optionally uses ZipCache / hybrid modes.
            self.self_attn = MistralFlashAttention_KIVI(
                config=config,
                nbits_key=nbits_key,
                nbits_value=nbits_value,
                layer_idx=layer_idx,
                quant_mode=quant_mode,
                zipcache_config=zipcache_config,
            )

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralModel_KIVI(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig, layer_quant_map: Optional[Dict[int, Dict[str, int]]] = None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # KVTuner: Modify layer creation to pass layer-specific quantization bits.
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            # Determine the nbits for this specific layer.
            if layer_quant_map and layer_idx in layer_quant_map:
                # Use the specific bits from the map if available.
                nbits_key = layer_quant_map[layer_idx]['nbits_key']
                nbits_value = layer_quant_map[layer_idx]['nbits_value']
            else:
                # Fallback to global config values if no layer-specific config is found.
                nbits_key = config.k_bits
                nbits_value = config.v_bits
            
            # Pass the specific nbits to the decoder layer constructor.
            self.layers.append(
                MistralDecoderLayer_KIVI(
                    config,
                    nbits_key=nbits_key,
                    nbits_value=nbits_value,
                    layer_idx=layer_idx,
                )
            )

        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = _kivi_get_past_length(past_key_values)
        if past_key_values_length > 0:
            seq_length_with_past += past_key_values_length
        else:
            past_key_values = None

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and hasattr(self.config, "_flash_attn_2_enabled")
            and self.config._flash_attn_2_enabled
            and past_key_values is not None
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # >>> ADD: per-layer debug before attention <<<
            if DEBUG:
                print(
                    f"[DEBUG][KIVI-DEC] layer={idx} "
                    f"hidden_in={tuple(hidden_states.shape)} "
                    f"past_is_none={past_key_value is None}",
                    flush=True,
                )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            # >>> ADD: per-layer debug after attention <<<
            if DEBUG:
                print(
                    f"[DEBUG][KIVI-DEC] layer={idx} done "
                    f"hidden_out={tuple(hidden_states.shape)}",
                    flush=True,
                )
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM_KIVI(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    # KVTuner: Modify the constructor to accept **kwargs and pass the map down.
    def __init__(self, config, **kwargs):
        super().__init__(config)
        # KVTuner: Pop the layer_quant_map from kwargs to pass it to the core model.
        # This allows the argument to be passed through from_pretrained().
        layer_quant_map = kwargs.pop("layer_quant_map", None)
        self.model = MistralModel_KIVI(config, layer_quant_map=layer_quant_map)
        
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ############### DEBUG INFO ###############
        # >>> ADD: high-level model forward debug <<<
        bsz = None
        seqlen = None
        if input_ids is not None:
            bsz, seqlen = input_ids.shape
        elif inputs_embeds is not None:
            bsz, seqlen, _ = inputs_embeds.shape

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first = past_key_values[0]
            # KVTuner-only: tuple of tensors, ZipCache: dict with "shape"
            if isinstance(first, dict) and "shape" in first:
                past_len = int(first["shape"][2])
            elif isinstance(first, (tuple, list)) and len(first) > 0:
                # K-mode: key tensor is [B, H_kv, T, D]
                past_len = int(first[0].shape[-2])

        if DEBUG:
            print(
                f"[DEBUG][KIVI-CLM] bsz={bsz} seqlen={seqlen} "
                f"past_len={past_len} use_cache={use_cache}",
                flush=True,
            )
        #################################

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        past_length = _kivi_get_past_length(past_key_values)
        if past_length > 0:
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past