#!/usr/bin/env python3
"""
Export ONNX subgraphs for Qwen3-TTS VoiceDesign checkpoints.

This script is intentionally stage-based (not full end-to-end generate), so it
matches a C++ runtime split:
  - talker_decode.onnx
  - code_predictor_decode.onnx
  - speech_tokenizer_decode.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import (
    apply_rotary_pos_emb,
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
)


def _require_onnx() -> None:
    try:
        import onnx  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Python package `onnx` is not installed. "
            "Install it first, then rerun this exporter."
        ) from exc
    try:
        import onnxscript  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Python package `onnxscript` is not installed. "
            "Your torch.onnx exporter requires it."
        ) from exc


class TalkerDecodeExport(nn.Module):
    """
    Full-context talker decode without HF masking_utils path.

    Inputs:
      - prefill_embeds: [B, P, D]
      - codec_ids: [B, T, Q]
      - trailing_text: [B, T, D]
    Outputs:
      - logits: [B, 1, vocab_size]
      - last_hidden: [B, 1, D]
    """

    def __init__(self, talker: nn.Module):
        super().__init__()
        self.talker = talker
        self.talker.model.config._attn_implementation = "eager"

    def forward(
        self,
        prefill_embeds: torch.Tensor,
        codec_ids: torch.Tensor,
        trailing_text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        first_embed = self.talker.get_input_embeddings()(codec_ids[:, :, :1])
        codec_embeds = [first_embed]
        # Remaining codebooks use code_predictor codec embeddings.
        for i in range(1, codec_ids.shape[2]):
            codec_embeds.append(
                self.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i : i + 1])
            )

        step_embed = torch.cat(codec_embeds, dim=2).sum(2) + trailing_text
        hidden_states = torch.cat([prefill_embeds, step_embed], dim=1)

        bsz = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        pos_1d = torch.arange(0, seq_len, device=hidden_states.device, dtype=torch.long).view(1, seq_len).repeat(bsz, 1)
        pos_3d = pos_1d.unsqueeze(0).expand(3, -1, -1)

        # Additive causal mask expected by eager_attention_forward: [B, 1, Q, K].
        # Use finite negative value for better ONNXRuntime numerical stability.
        mask = torch.triu(
            torch.full((seq_len, seq_len), -1e4, device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len).repeat(bsz, 1, 1, 1)

        position_embeddings = self.talker.model.rotary_emb(hidden_states, pos_3d)
        for layer in self.talker.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=pos_1d,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )[0]
        hidden_states = self.talker.model.norm(hidden_states)
        hidden = hidden_states[:, -1:, :]
        logits = self.talker.codec_head(hidden)
        return logits, hidden[:, -1:, :]


class VoiceDesignPrefillBuilderExport(nn.Module):
    """
    Build prefill embeddings for VoiceDesign (single sample, non-streaming, no speaker id).

    Inputs:
      - input_ids: [1, L_text]
      - instruct_ids: [1, L_inst]
      - codec_language_token_id: [1] (use -1 for Auto)
    Outputs:
      - prefill_embeds: [1, S, D]
      - tts_pad_embed: [1, 1, D]
    """

    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.m = full_model

    def forward(
        self,
        input_ids: torch.Tensor,
        instruct_ids: torch.Tensor,
        codec_language_token_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        talker = self.m.talker
        cfg = self.m.config
        tc = cfg.talker_config

        # Optional instruct prompt.
        instruct_embed = talker.text_projection(talker.get_text_embeddings()(instruct_ids))

        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor(
                    [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            )
        ).chunk(3, dim=1)

        lang_tok = codec_language_token_id[0].to(dtype=input_ids.dtype)
        auto_vec = torch.tensor(
            [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id, tc.codec_pad_id],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        lang_vec = torch.stack(
            [
                torch.tensor(tc.codec_think_id, device=input_ids.device, dtype=input_ids.dtype),
                torch.tensor(tc.codec_think_bos_id, device=input_ids.device, dtype=input_ids.dtype),
                lang_tok,
                torch.tensor(tc.codec_think_eos_id, device=input_ids.device, dtype=input_ids.dtype),
            ],
            dim=0,
        )
        is_auto = (codec_language_token_id[0] < 0)
        codec_prefill_1d = torch.where(is_auto, auto_vec, lang_vec)
        codec_prefill_2d = codec_prefill_1d.unsqueeze(0)

        codec_input_embedding_0 = talker.get_input_embeddings()(codec_prefill_2d)
        codec_input_embedding_1 = talker.get_input_embeddings()(
            torch.tensor([[tc.codec_pad_id, tc.codec_bos_id]], device=input_ids.device, dtype=input_ids.dtype)
        )
        codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)

        # <|im_start|>assistant\n
        role_embed = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))
        talker_input_embed = torch.cat(
            [
                role_embed,
                torch.cat(
                    (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed), dim=1
                )
                + codec_input_embedding[:, :-1],
            ],
            dim=1,
        )

        # non_streaming_mode=True path from original generate()
        talker_input_embed = torch.cat(
            [
                talker_input_embed,
                talker.text_projection(talker.get_text_embeddings()(input_ids[:, 3:4])) + codec_input_embedding[:, -1:],
            ],
            dim=1,
        )
        talker_input_embed = talker_input_embed[:, :-1]

        text_core = talker.text_projection(talker.get_text_embeddings()(input_ids[:, 3:-5]))
        pad_len = input_ids[:, 3:-5].shape[1] + 1
        pad_ids = torch.full((1, pad_len), tc.codec_pad_id, device=input_ids.device, dtype=input_ids.dtype)
        bos_ids = torch.tensor([[tc.codec_bos_id]], device=input_ids.device, dtype=input_ids.dtype)

        talker_input_embed = torch.cat(
            [
                talker_input_embed,
                torch.cat((text_core, tts_eos_embed), dim=1) + talker.get_input_embeddings()(pad_ids),
                tts_pad_embed + talker.get_input_embeddings()(bos_ids),
            ],
            dim=1,
        )
        prefill_embeds = torch.cat([instruct_embed, talker_input_embed], dim=1)
        return prefill_embeds, tts_pad_embed


class TalkerPrefillExport(nn.Module):
    """
    Run talker over the prefill embeddings and output first-step state.

    Inputs:
      - prefill_embeds: [1, S, D]
    Outputs:
      - logits: [1, 1, vocab_size]
      - last_hidden: [1, 1, D]
    """

    def __init__(self, talker: nn.Module):
        super().__init__()
        self.talker = talker

    def forward(self, prefill_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.talker.model(
            input_ids=None,
            inputs_embeds=prefill_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.talker.codec_head(hidden)
        return logits, hidden


class _TalkerCacheMixin:
    def _attn_with_optional_past(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if hasattr(attn, "rope_scaling"):
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                attn.rope_scaling["mrope_section"],
                attn.rope_scaling["interleaved"],
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_k is not None and past_v is not None:
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        attn_output, _ = eager_attention_forward(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=attn.scaling,
            dropout=0.0,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, key_states, value_states


class TalkerPrefillCacheExport(nn.Module, _TalkerCacheMixin):
    """
    Talker prefill with explicit KV cache outputs.

    Inputs:
      - prefill_embeds: [B, P, D]
    Outputs:
      - logits: [B, 1, vocab_size]
      - last_hidden: [B, 1, D]
      - present_k: [L, B, KV, P, HD]
      - present_v: [L, B, KV, P, HD]
    """

    def __init__(self, talker: nn.Module):
        super().__init__()
        self.talker = talker
        self.talker.model.config._attn_implementation = "eager"

    def forward(self, prefill_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = prefill_embeds
        bsz, seq_len, _ = hidden_states.shape

        pos_1d = torch.arange(0, seq_len, device=hidden_states.device, dtype=torch.long).view(1, seq_len).repeat(bsz, 1)
        pos_3d = pos_1d.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.talker.model.rotary_emb(hidden_states, pos_3d)

        mask = torch.triu(
            torch.full((seq_len, seq_len), -1e4, device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len).repeat(bsz, 1, 1, 1)

        present_k = []
        present_v = []
        for layer in self.talker.model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, k, v = self._attn_with_optional_past(
                layer.self_attn,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                past_k=None,
                past_v=None,
            )
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

            present_k.append(k)
            present_v.append(v)

        hidden_states = self.talker.model.norm(hidden_states)
        hidden = hidden_states[:, -1:, :]
        logits = self.talker.codec_head(hidden)
        return logits, hidden, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


class TalkerDecodeCacheExport(nn.Module, _TalkerCacheMixin):
    """
    One-step talker decode with explicit KV cache.

    Inputs:
      - codec_ids_step: [B, 1, Q]
      - trailing_text_step: [B, 1, D]
      - past_k: [L, B, KV, S, HD]
      - past_v: [L, B, KV, S, HD]
      - cache_position: [1] (absolute position of current token)
    Outputs:
      - logits: [B, 1, vocab_size]
      - last_hidden: [B, 1, D]
      - present_k: [L, B, KV, S+1, HD]
      - present_v: [L, B, KV, S+1, HD]
    """

    def __init__(self, talker: nn.Module):
        super().__init__()
        self.talker = talker
        self.talker.model.config._attn_implementation = "eager"

    def forward(
        self,
        codec_ids_step: torch.Tensor,
        trailing_text_step: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        first_embed = self.talker.get_input_embeddings()(codec_ids_step[:, :, :1])
        codec_embeds = [first_embed]
        for i in range(1, codec_ids_step.shape[2]):
            codec_embeds.append(
                self.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids_step[:, :, i : i + 1])
            )
        hidden_states = torch.cat(codec_embeds, dim=2).sum(2) + trailing_text_step

        bsz = hidden_states.shape[0]
        pos_1d = cache_position.view(1, 1).repeat(bsz, 1)
        pos_3d = pos_1d.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = self.talker.model.rotary_emb(hidden_states, pos_3d)

        present_k = []
        present_v = []
        for li, layer in enumerate(self.talker.model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, k, v = self._attn_with_optional_past(
                layer.self_attn,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_k=past_k[li],
                past_v=past_v[li],
            )
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

            present_k.append(k)
            present_v.append(v)

        hidden_states = self.talker.model.norm(hidden_states)
        hidden = hidden_states[:, -1:, :]
        logits = self.talker.codec_head(hidden)
        return logits, hidden, torch.stack(present_k, dim=0), torch.stack(present_v, dim=0)


class CodePredictorDecodeExport(nn.Module):
    """
    One-step code predictor logits (all code groups at once), no HF generate loop.

    Inputs:
      - seed_embeds: [B, 2, D_talker]
      - attention_mask: [B, 2]
      - position_ids: [B, 2]
    Outputs:
      - logits: [B, vocab_size]
    """

    def __init__(self, code_predictor: nn.Module):
        super().__init__()
        self.code_predictor = code_predictor

    def forward(
        self,
        seed_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.code_predictor(
            input_ids=None,
            inputs_embeds=seed_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        # In current checkpoint this is a tensor [B, S, V].
        logits = out.logits
        if logits.dim() == 3:
            return logits[:, -1, :]
        if logits.dim() == 2:
            return logits
        raise RuntimeError(f"Unexpected logits rank from code predictor: {tuple(logits.shape)}")


class CodePredictorFixedStepExport(nn.Module):
    """
    Autoregressive one-step code predictor for groups 2..Q.

    Inputs:
      - past_hidden: [B, 1, D_talker]
      - first_code_id: [B, 1]
      - prev_codes: [B, Q-2] (already generated sub-codes; only first `step` are used)
    Outputs:
      - logits: [B, code_predictor_vocab]
    """

    def __init__(self, talker: nn.Module, step: int):
        super().__init__()
        self.step = step
        self.talker = talker
        self.code_predictor = talker.code_predictor

    def forward(
        self,
        past_hidden: torch.Tensor,
        first_code_id: torch.Tensor,
        prev_codes: torch.Tensor,
    ) -> torch.Tensor:
        past_hidden = self.code_predictor.small_to_mtp_projection(past_hidden)
        first_embed = self.talker.get_input_embeddings()(first_code_id)
        first_embed = self.code_predictor.small_to_mtp_projection(first_embed)
        prev_embeds = []
        for i in range(self.step):
            e = self.code_predictor.get_input_embeddings()[i](prev_codes[:, i : i + 1])
            prev_embeds.append(self.code_predictor.small_to_mtp_projection(e))
        all_embeds = torch.cat([past_hidden, first_embed] + prev_embeds, dim=1)
        batch = all_embeds.shape[0]
        seq_len = all_embeds.shape[1]
        attention_mask = torch.ones((batch, seq_len), dtype=torch.long, device=all_embeds.device)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=all_embeds.device).view(1, seq_len).repeat(batch, 1)

        out = self.code_predictor.model(
            input_ids=None,
            inputs_embeds=all_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        hidden = out.last_hidden_state[:, -1, :]
        return self.code_predictor.lm_head[self.step](hidden)


class CodePredictorDynamicStepExport(nn.Module, _TalkerCacheMixin):
    """
    Autoregressive one-step code predictor for groups 2..Q using one shared ONNX graph.

    Inputs:
      - past_hidden: [B, 1, D_talker]
      - first_code_id: [B, 1]
      - prev_codes: [B, Q-2] (already generated sub-codes; only first `step_id` are used)
      - step_id: [1] int64 in [0, Q-2]
    Outputs:
      - logits: [B, code_predictor_vocab]
    """

    def __init__(self, talker: nn.Module):
        super().__init__()
        self.talker = talker
        self.code_predictor = talker.code_predictor
        self.num_steps = talker.config.num_code_groups - 1
        self.code_predictor.model.config._attn_implementation = "eager"

    def forward(
        self,
        past_hidden: torch.Tensor,
        first_code_id: torch.Tensor,
        prev_codes: torch.Tensor,
        step_id: torch.Tensor,
    ) -> torch.Tensor:
        step_scalar = torch.clamp(step_id.view(-1)[0], min=0, max=self.num_steps - 1)

        past_hidden = self.code_predictor.small_to_mtp_projection(past_hidden)
        first_embed = self.talker.get_input_embeddings()(first_code_id)
        first_embed = self.code_predictor.small_to_mtp_projection(first_embed)

        prev_embeds = [
            self.code_predictor.small_to_mtp_projection(
                self.code_predictor.get_input_embeddings()[i](prev_codes[:, i : i + 1])
            )
            for i in range(self.num_steps - 1)
        ]
        all_embeds = torch.cat([past_hidden, first_embed] + prev_embeds, dim=1)
        batch = all_embeds.shape[0]
        seq_len = all_embeds.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=all_embeds.device).view(1, seq_len).repeat(batch, 1)
        position_embeddings = self.code_predictor.model.rotary_emb(all_embeds, position_ids)

        mask = torch.triu(
            torch.full((seq_len, seq_len), -1e4, device=all_embeds.device, dtype=all_embeds.dtype),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len).repeat(batch, 1, 1, 1)

        hidden_states = all_embeds
        for layer in self.code_predictor.model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, _, _ = self._attn_with_optional_past(
                layer.self_attn,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                past_k=None,
                past_v=None,
            )
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

        hidden_states = self.code_predictor.model.norm(hidden_states)
        hidden_index = (step_scalar + 1).to(dtype=torch.long).view(1, 1, 1).expand(batch, 1, hidden_states.shape[-1])
        hidden = torch.gather(hidden_states, dim=1, index=hidden_index).squeeze(1)

        logits_all = []
        for i in range(self.num_steps):
            logits_all.append(self.code_predictor.lm_head[i](hidden))
        logits_all = torch.stack(logits_all, dim=1)  # [B, num_steps, V]
        step_index = step_scalar.to(dtype=torch.long).view(1, 1, 1).expand(batch, 1, logits_all.shape[-1])
        logits = torch.gather(logits_all, dim=1, index=step_index).squeeze(1)
        return logits


class SpeechTokenizerDecodeExport(nn.Module):
    """
    Voice tokenizer decoder (12Hz): codes -> waveform tensor.

    Inputs:
      - audio_codes: [B, T, Q]
    Outputs:
      - audio_values: [B, N]
      - audio_lengths: [B]
    """

    def __init__(self, speech_tokenizer_model: nn.Module, decode_upsample_rate: int):
        super().__init__()
        self.decoder = speech_tokenizer_model.decoder
        self.decode_upsample_rate = decode_upsample_rate

    def forward(self, audio_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.decoder(audio_codes.transpose(1, 2)).squeeze(1)
        lengths = (audio_codes[..., 0] > 0).sum(1) * self.decode_upsample_rate
        return hidden, lengths.to(dtype=torch.int64)


def _export_onnx(
    model: nn.Module,
    args: Tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    opset: int,
    dynamo: bool = True,
) -> None:
    torch.onnx.export(
        model,
        args,
        str(output_path),
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=dynamo,
    )
    # Prevent cross-model collisions like onnx__MatMul_* when multiple models are exported
    # into one directory by packing each model's external tensors into a unique data file.
    import onnx

    model_proto = onnx.load_model(str(output_path), load_external_data=True)
    onnx.save_model(
        model_proto,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.name}.data",
        # Keep tiny scalar/tiny-shape constants in the main .onnx for ORT compatibility.
        size_threshold=1024,
    )


def _cleanup_orphan_external_files(out_dir: Path) -> None:
    import onnx

    referenced = set()
    for onnx_path in out_dir.glob("*.onnx"):
        model_proto = onnx.load_model(str(onnx_path), load_external_data=False)
        for init in model_proto.graph.initializer:
            if init.data_location != onnx.TensorProto.EXTERNAL:
                continue
            location = None
            for kv in init.external_data:
                if kv.key == "location":
                    location = kv.value
                    break
            if location:
                referenced.add(location)

    removed_files = 0
    removed_bytes = 0
    for path in out_dir.iterdir():
        if not path.is_file() or path.name.endswith(".onnx"):
            continue
        if path.name not in referenced:
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1

    if removed_files > 0:
        print(f"[cleanup] removed orphan external files: {removed_files}, bytes={removed_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("onnx_out"))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--minimal-runtime-bundle",
        action="store_true",
        help=(
            "Export only models required by C++ runtime with KV cache and shared dynamic CP: "
            "prefill_builder, talker_prefill_cache, talker_decode_cache, code_predictor_dynamic, speech_tokenizer_decode."
        ),
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(str(model_dir), dtype=torch.float32)
    tts.model.eval()
    tts.model.talker.eval()
    tts.model.talker.code_predictor.eval()
    tts.model.speech_tokenizer.model.eval()

    cfg = tts.model.config.talker_config
    batch = 1
    q = int(cfg.num_code_groups)
    d = int(cfg.hidden_size)
    code_vocab = int(cfg.vocab_size)
    cp_vocab = int(cfg.code_predictor_config.vocab_size)

    summary = {
        "model_dir": str(model_dir),
        "tokenizer_type": tts.model.tokenizer_type,
        "tts_model_type": tts.model.tts_model_type,
        "num_code_groups": q,
        "talker_hidden_size": d,
        "talker_vocab_size": code_vocab,
        "code_predictor_vocab_size": cp_vocab,
        "decode_upsample_rate": int(tts.model.speech_tokenizer.model.get_decode_upsample_rate()),
    }
    print(json.dumps(summary, indent=2))

    if args.inspect_only:
        return

    _require_onnx()

    talker_export = TalkerDecodeExport(tts.model.talker).eval()
    prefill_builder_export = VoiceDesignPrefillBuilderExport(tts.model).eval()
    talker_prefill_export = TalkerPrefillExport(tts.model.talker).eval()
    talker_prefill_cache_export = TalkerPrefillCacheExport(tts.model.talker).eval()
    talker_decode_cache_export = TalkerDecodeCacheExport(tts.model.talker).eval()
    cp_export = CodePredictorDecodeExport(tts.model.talker.code_predictor).eval()
    cp_dynamic_export = CodePredictorDynamicStepExport(tts.model.talker).eval()
    st_export = SpeechTokenizerDecodeExport(
        tts.model.speech_tokenizer.model,
        decode_upsample_rate=int(tts.model.speech_tokenizer.model.get_decode_upsample_rate()),
    ).eval()

    with torch.no_grad():
        t_hist = 4
        p_len = 40
        prefill = torch.zeros((batch, p_len, d), dtype=torch.float32)
        codec_ids = torch.ones((batch, t_hist, q), dtype=torch.long)
        trailing = torch.zeros((batch, t_hist, d), dtype=torch.float32)
        if not args.minimal_runtime_bundle:
            _export_onnx(
                talker_export,
                (prefill, codec_ids, trailing),
                out_dir / "talker_decode.onnx",
                input_names=["prefill_embeds", "codec_ids", "trailing_text"],
                output_names=["logits", "last_hidden"],
                dynamic_axes={
                    "prefill_embeds": {0: "batch", 1: "prefill_len"},
                    "codec_ids": {0: "batch", 1: "steps"},
                    "trailing_text": {0: "batch", 1: "steps"},
                    "logits": {0: "batch"},
                    "last_hidden": {0: "batch"},
                },
                opset=args.opset,
                dynamo=False,
            )

        # Prefill builder (text+instruct+language-token -> prefill embeds)
        dummy_input_ids = torch.tensor([[151644, 77091, 198, 872, 151645, 198, 151644, 77091, 198]], dtype=torch.long)
        dummy_instruct_ids = torch.tensor([[151644, 872, 198, 11782, 151645, 198]], dtype=torch.long)
        dummy_lang = torch.tensor([-1], dtype=torch.long)
        _export_onnx(
            prefill_builder_export,
            (dummy_input_ids, dummy_instruct_ids, dummy_lang),
            out_dir / "prefill_builder.onnx",
            input_names=["input_ids", "instruct_ids", "codec_language_token_id"],
            output_names=["prefill_embeds", "tts_pad_embed"],
            dynamic_axes={
                "input_ids": {1: "text_len"},
                "instruct_ids": {1: "instruct_len"},
                "prefill_embeds": {1: "prefill_len"},
            },
            opset=args.opset,
            dynamo=True,
        )

        # Talker prefill (prefill embeds -> first hidden/logits)
        dummy_prefill = torch.zeros((1, 32, d), dtype=torch.float32)
        if not args.minimal_runtime_bundle:
            _export_onnx(
                talker_prefill_export,
                (dummy_prefill,),
                out_dir / "talker_prefill.onnx",
                input_names=["prefill_embeds"],
                output_names=["logits", "last_hidden"],
                dynamic_axes={
                    "prefill_embeds": {1: "prefill_len"},
                },
                opset=args.opset,
                dynamo=True,
            )
        # Talker prefill with explicit cache outputs
        kv_heads = int(cfg.num_key_value_heads)
        head_dim = int(cfg.hidden_size // cfg.num_attention_heads)
        n_layers = int(cfg.num_hidden_layers)
        _export_onnx(
            talker_prefill_cache_export,
            (dummy_prefill,),
            out_dir / "talker_prefill_cache.onnx",
            input_names=["prefill_embeds"],
            output_names=["logits", "last_hidden", "present_k", "present_v"],
            dynamic_axes={
                "prefill_embeds": {1: "prefill_len"},
                "present_k": {3: "cache_len"},
                "present_v": {3: "cache_len"},
            },
            opset=args.opset,
            dynamo=False,
        )

        # Talker one-step decode with cache input/output
        dummy_codec_step = torch.ones((batch, 1, q), dtype=torch.long)
        dummy_trailing_step = torch.zeros((batch, 1, d), dtype=torch.float32)
        dummy_past_k = torch.zeros((n_layers, batch, kv_heads, t_hist, head_dim), dtype=torch.float32)
        dummy_past_v = torch.zeros((n_layers, batch, kv_heads, t_hist, head_dim), dtype=torch.float32)
        dummy_cache_pos = torch.tensor([t_hist], dtype=torch.long)
        _export_onnx(
            talker_decode_cache_export,
            (dummy_codec_step, dummy_trailing_step, dummy_past_k, dummy_past_v, dummy_cache_pos),
            out_dir / "talker_decode_cache.onnx",
            input_names=["codec_ids_step", "trailing_text_step", "past_k", "past_v", "cache_position"],
            output_names=["logits", "last_hidden", "present_k", "present_v"],
            dynamic_axes={
                "codec_ids_step": {0: "batch"},
                "trailing_text_step": {0: "batch"},
                "past_k": {1: "batch", 3: "cache_len"},
                "past_v": {1: "batch", 3: "cache_len"},
                "logits": {0: "batch"},
                "last_hidden": {0: "batch"},
                "present_k": {1: "batch", 3: "cache_len_plus_1"},
                "present_v": {1: "batch", 3: "cache_len_plus_1"},
            },
            opset=args.opset,
            dynamo=False,
        )

        seed = torch.zeros((batch, 2, d), dtype=torch.float32)
        attn_2 = torch.ones((batch, 2), dtype=torch.long)
        pos_2 = torch.arange(0, 2, dtype=torch.long).view(1, 2).repeat(batch, 1)
        if not args.minimal_runtime_bundle:
            _export_onnx(
                cp_export,
                (seed, attn_2, pos_2),
                out_dir / "code_predictor_decode.onnx",
                input_names=["seed_embeds", "attention_mask", "position_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "seed_embeds": {0: "batch"},
                    "attention_mask": {0: "batch"},
                    "position_ids": {0: "batch"},
                    "logits": {0: "batch"},
                },
                opset=args.opset,
                dynamo=True,
            )

        prev_codes = torch.zeros((batch, q - 2), dtype=torch.long)
        first_code_id = torch.zeros((batch, 1), dtype=torch.long)
        past_hidden = torch.zeros((batch, 1, d), dtype=torch.float32)
        step_id = torch.tensor([0], dtype=torch.long)
        _export_onnx(
            cp_dynamic_export,
            (past_hidden, first_code_id, prev_codes, step_id),
            out_dir / "code_predictor_dynamic.onnx",
            input_names=["past_hidden", "first_code_id", "prev_codes", "step_id"],
            output_names=["logits"],
            dynamic_axes={
                "past_hidden": {0: "batch"},
                "first_code_id": {0: "batch"},
                "prev_codes": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset=args.opset,
            dynamo=False,
        )
        if not args.minimal_runtime_bundle:
            for step in range(q - 1):
                cp_fixed_export = CodePredictorFixedStepExport(tts.model.talker, step=step).eval()
                _export_onnx(
                    cp_fixed_export,
                    (past_hidden, first_code_id, prev_codes),
                    out_dir / f"code_predictor_step_{step:02d}.onnx",
                    input_names=["past_hidden", "first_code_id", "prev_codes"],
                    output_names=["logits"],
                    dynamic_axes={
                        "past_hidden": {0: "batch"},
                        "first_code_id": {0: "batch"},
                        "prev_codes": {0: "batch"},
                        "logits": {0: "batch"},
                    },
                    opset=args.opset,
                    dynamo=True,
                )

        # 12Hz decoder expects [B, T, Q].
        t_steps = 20
        audio_codes = torch.ones((batch, t_steps, q), dtype=torch.long)
        _export_onnx(
            st_export,
            (audio_codes,),
            out_dir / "speech_tokenizer_decode.onnx",
            input_names=["audio_codes"],
            output_names=["audio_values", "audio_lengths"],
            dynamic_axes={
                "audio_codes": {0: "batch", 1: "code_steps"},
                "audio_values": {0: "batch", 1: "audio_samples"},
                "audio_lengths": {0: "batch"},
            },
            opset=args.opset,
            dynamo=True,
        )

    _cleanup_orphan_external_files(out_dir)
    print(f"Exported ONNX files into: {out_dir}")


if __name__ == "__main__":
    main()
