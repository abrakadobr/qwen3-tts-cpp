#!/usr/bin/env python3
"""Compare C++ ONNX-generated codec IDs against deterministic Python reference.

This runs the same stage wrappers used for ONNX export (prefill, talker prefill,
talker decode, per-step code predictor) in greedy mode and compares line-by-line
with a C++ codes text file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen_tts import Qwen3TTSModel
from export_voice_design_onnx import (
    CodePredictorFixedStepExport,
    TalkerDecodeExport,
    TalkerPrefillExport,
    VoiceDesignPrefillBuilderExport,
)


def read_codes_txt(path: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(x) for x in line.split()])
    return rows


def make_py_codes(model_dir: Path, token_json: Path, steps: int) -> list[list[int]]:
    tts = Qwen3TTSModel.from_pretrained(str(model_dir), dtype=torch.float32)
    tts.model.eval()
    talker = tts.model.talker
    q = int(talker.config.num_code_groups)

    pb = VoiceDesignPrefillBuilderExport(tts.model).eval()
    tp = TalkerPrefillExport(talker).eval()
    td = TalkerDecodeExport(talker).eval()
    cp_steps = [CodePredictorFixedStepExport(talker, i).eval() for i in range(q - 1)]

    j = json.loads(token_json.read_text(encoding="utf-8"))
    input_ids = torch.tensor([j["input_ids"]], dtype=torch.long)
    instruct_ids = torch.tensor([j["instruct_ids"]], dtype=torch.long)
    lang = torch.tensor([j["codec_language_token_id"]], dtype=torch.long)

    with torch.no_grad():
        prefill_embeds, tts_pad = pb(input_ids, instruct_ids, lang)
        logits0, hidden0 = tp(prefill_embeds)

        current_first = int(torch.argmax(logits0[0, 0]).item())
        current_hidden = hidden0.clone()
        all_codes: list[list[int]] = []

        for s in range(steps):
            row = [0] * q
            row[0] = current_first
            prev = torch.zeros((1, q - 2), dtype=torch.long)

            for g in range(q - 1):
                logits = cp_steps[g](
                    current_hidden,
                    torch.tensor([[row[0]]], dtype=torch.long),
                    prev,
                )
                pred = int(torch.argmax(logits[0]).item())
                row[g + 1] = pred
                if g < q - 2:
                    prev[0, g] = pred

            all_codes.append(row)
            if s == steps - 1:
                break

            hist = torch.tensor([all_codes], dtype=torch.long)
            trailing = tts_pad.expand(1, hist.shape[1], tts_pad.shape[-1]).contiguous()
            logits_next, hidden_next = td(prefill_embeds, hist, trailing)
            current_first = int(torch.argmax(logits_next[0, 0]).item())
            current_hidden = hidden_next

    return all_codes


def first_mismatch(a: list[list[int]], b: list[list[int]]) -> tuple[int, int] | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            m = min(len(a[i]), len(b[i]))
            for j in range(m):
                if a[i][j] != b[i][j]:
                    return (i, j)
            return (i, m)
    if len(a) != len(b):
        return (n, 0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--token-json", type=Path, required=True)
    ap.add_argument("--cpp-codes", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--dump-py-codes", type=Path, default=None)
    args = ap.parse_args()

    cpp_codes = read_codes_txt(args.cpp_codes)
    py_codes = make_py_codes(args.model_dir, args.token_json, args.steps)

    if args.dump_py_codes is not None:
        lines = [" ".join(str(x) for x in row) for row in py_codes]
        args.dump_py_codes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mismatch = first_mismatch(py_codes, cpp_codes)
    if mismatch is None:
        print(f"MATCH: {len(py_codes)} steps identical")
        return 0

    step_idx, code_idx = mismatch
    if step_idx < len(py_codes) and step_idx < len(cpp_codes) and code_idx < len(py_codes[step_idx]) and code_idx < len(cpp_codes[step_idx]):
        print(
            "MISMATCH: "
            f"step={step_idx + 1} codebook={code_idx + 1} "
            f"py={py_codes[step_idx][code_idx]} cpp={cpp_codes[step_idx][code_idx]}"
        )
    else:
        print(
            "MISMATCH: different lengths "
            f"py_steps={len(py_codes)} cpp_steps={len(cpp_codes)}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
