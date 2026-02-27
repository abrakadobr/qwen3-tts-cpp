#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def _cleanup_orphan_external_files(out_dir: Path) -> None:
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
        print(f"[cleanup] removed orphan files={removed_files}, bytes={removed_bytes}")


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _pick_models(in_dir: Path, selected: Iterable[str] | None) -> list[Path]:
    if selected:
        models: list[Path] = []
        for name in selected:
            p = in_dir / name
            if not p.exists():
                raise FileNotFoundError(f"Model does not exist: {p}")
            if p.suffix != ".onnx":
                raise ValueError(f"Expected .onnx file name, got: {name}")
            models.append(p)
        return models
    return sorted(in_dir.glob("*.onnx"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of .onnx file names to quantize. By default quantizes all *.onnx in --in-dir.",
    )
    parser.add_argument(
        "--weight-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Dynamic quantized weight type.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel quantization for weights when supported.",
    )
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"--in-dir does not exist: {in_dir}")

    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
    models = _pick_models(in_dir, args.models)
    if not models:
        raise RuntimeError(f"No ONNX models found in {in_dir}")

    print(f"[quant] input dir: {in_dir}")
    print(f"[quant] output dir: {out_dir}")
    print(f"[quant] models: {len(models)}")

    for src in models:
        dst = out_dir / src.name
        print(f"[quant] {src.name} -> {dst.name}")
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=args.per_channel,
            weight_type=weight_type,
            use_external_data_format=True,
            extra_options={"EnableSubgraph": True},
        )

    _cleanup_orphan_external_files(out_dir)

    in_bytes = _dir_size_bytes(in_dir)
    out_bytes = _dir_size_bytes(out_dir)
    ratio = (out_bytes / in_bytes) if in_bytes > 0 else 0.0
    print(f"[size] input_bytes={in_bytes}")
    print(f"[size] output_bytes={out_bytes}")
    print(f"[size] ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
