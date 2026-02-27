#!/usr/bin/env python3
"""
Prepare token ids for the C++ VoiceDesign ONNX smoke pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from qwen_tts.core.models import Qwen3TTSConfig


def _build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def _build_instruct_text(instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--instruct", type=str, required=True)
    ap.add_argument("--language", type=str, default="auto")
    ap.add_argument("--out", type=Path, default=Path("voice_design_tokens.json"))
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(str(args.model_dir), fix_mistral_regex=True)
    cfg = Qwen3TTSConfig.from_pretrained(str(args.model_dir))

    in_text = _build_assistant_text(args.text)
    in_inst = _build_instruct_text(args.instruct)

    text_ids = processor(text=in_text, return_tensors="pt", padding=True)["input_ids"][0].to(torch.int64)
    inst_ids = processor(text=in_inst, return_tensors="pt", padding=True)["input_ids"][0].to(torch.int64)

    lang = args.language.lower()
    if lang == "auto":
        codec_language_token_id = -1
    else:
        if lang not in cfg.talker_config.codec_language_id:
            raise ValueError(
                f"Unsupported language '{args.language}'. "
                f"Supported: {sorted(cfg.talker_config.codec_language_id.keys()) + ['auto']}"
            )
        codec_language_token_id = int(cfg.talker_config.codec_language_id[lang])

    payload = {
        "text": args.text,
        "instruct": args.instruct,
        "language": args.language,
        "input_ids": text_ids.tolist(),
        "instruct_ids": inst_ids.tolist(),
        "codec_language_token_id": codec_language_token_id,
    }

    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote token payload: {args.out}")


if __name__ == "__main__":
    main()

