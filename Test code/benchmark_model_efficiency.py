from __future__ import annotations

import csv
import gc
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile
from transformers import SegformerForSemanticSegmentation

from ceca_model import build_ceca_model
from tpsa_model import build_tpsa_model
from combined_model import build_combined_model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "09_model_efficiency"

MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
H = 512
W = 512
BATCH = 1
WARMUP = 50
RUNS = 200

BASELINE = ROOT / "rust1/experiment/checkpoints/formal_training_best"
LZ = ROOT / "rustLz/experiment/checkpoints/formal_training_best"
CECA = ROOT / "rustCECA/experiment/checkpoints/formal_training_best_state.pt"
TPSA = ROOT / "rustTPSA/experiment/checkpoints/formal_training_best_state.pt"
ALL = ROOT / "rustALL/experiment/checkpoints/formal_training_best_state.pt"


class LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits


def load_baseline():
    return SegformerForSemanticSegmentation.from_pretrained(
        BASELINE, local_files_only=True
    )


def load_lz():
    return SegformerForSemanticSegmentation.from_pretrained(
        LZ, local_files_only=True
    )


def load_ceca():
    return build_ceca_model(
        model_name_or_path=MODEL_NAME,
        num_labels=2,
        local_files_only=True,
        state_dict_path=CECA,
    )


def load_tpsa():
    return build_tpsa_model(
        model_name_or_path=MODEL_NAME,
        num_labels=2,
        local_files_only=True,
        state_dict_path=TPSA,
    )


def load_all():
    return build_combined_model(
        model_name_or_path=MODEL_NAME,
        num_labels=2,
        local_files_only=True,
        state_dict_path=ALL,
    )


SPECS = [
    ("Baseline", load_baseline, BASELINE),
    ("Lz-only", load_lz, LZ),
    ("CECA-only", load_ceca, CECA),
    ("TPSA-only", load_tpsa, TPSA),
    ("TPSA+CECA+Lz", load_all, ALL),
]


def mb(value):
    return float(value) / (1024 ** 2)


def path_size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    f = pos - lo
    return values[lo] * (1 - f) + values[hi] * f


def comparable_state_size(model, name):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / ("_temp_" + name.replace("+", "_").replace("-", "_") + ".pt")
    torch.save(model.state_dict(), temp)
    size = mb(temp.stat().st_size)
    temp.unlink()
    return size


def complexity(model, device):
    wrapper = LogitsWrapper(model).to(device).eval()
    x = torch.randn(BATCH, 3, H, W, device=device)
    with torch.inference_mode():
        macs, thop_params = profile(wrapper, inputs=(x,), verbose=False)
    del x, wrapper
    torch.cuda.empty_cache()
    return float(macs), float(thop_params)


def runtime(model, device):
    model = model.to(device).eval()
    x = torch.randn(BATCH, 3, H, W, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(WARMUP):
            with torch.autocast("cuda", dtype=torch.float16):
                y = model(pixel_values=x).logits
        torch.cuda.synchronize(device)

        memory_before = mb(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(RUNS)]

        for i in range(RUNS):
            starts[i].record()
            with torch.autocast("cuda", dtype=torch.float16):
                y = model(pixel_values=x).logits
            ends[i].record()

        torch.cuda.synchronize(device)

    if not torch.isfinite(y).all():
        raise ValueError("Non-finite logits detected.")

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    mean_ms = statistics.fmean(times)
    peak = mb(torch.cuda.max_memory_allocated(device))

    result = {
        "latency_mean_ms": mean_ms,
        "latency_std_ms": statistics.pstdev(times),
        "latency_median_ms": statistics.median(times),
        "latency_p95_ms": percentile(times, 0.95),
        "fps": 1000.0 * BATCH / mean_ms,
        "memory_before_inference_mb": memory_before,
        "peak_memory_mb": peak,
        "extra_inference_memory_mb": max(peak - memory_before, 0.0),
        "output_shape": str(tuple(y.shape)),
    }

    del x, y, starts, ends
    model.to("cpu")
    torch.cuda.empty_cache()
    return result


def benchmark(name, loader, source, device):
    print("\n" + "=" * 70)
    print(f"Benchmarking {name}")
    print("=" * 70)

    model = loader()

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    source_mb = mb(path_size(source))
    state_mb = comparable_state_size(model, name)

    print("Calculating approximate MACs...")
    macs, thop_params = complexity(model, device)

    model.to("cpu")
    torch.cuda.empty_cache()

    print(f"Running {WARMUP} warm-up and {RUNS} timed passes...")
    speed = runtime(model, device)

    row = {
        "model": name,
        "input_size": f"{H}x{W}",
        "batch_size": BATCH,
        "precision": "FP16 autocast",
        "parameters": params,
        "parameters_m": params / 1e6,
        "trainable_parameters": trainable,
        "trainable_parameters_m": trainable / 1e6,
        "thop_parameters": int(round(thop_params)),
        "macs": macs,
        "macs_g": macs / 1e9,
        "approx_flops_g": 2 * macs / 1e9,
        "source_checkpoint_mb": source_mb,
        "comparable_state_dict_mb": state_mb,
        **speed,
    }

    print(f"Parameters: {row['parameters_m']:.3f} M")
    print(f"Approx. MACs: {row['macs_g']:.3f} G")
    print(f"Latency: {row['latency_mean_ms']:.3f} ms")
    print(f"FPS: {row['fps']:.2f}")
    print(f"Peak VRAM: {row['peak_memory_mb']:.2f} MB")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def save_results(rows):
    OUT.mkdir(parents=True, exist_ok=True)

    csv_path = OUT / "model_efficiency_results.csv"
    json_path = OUT / "model_efficiency_results.json"
    txt_path = OUT / "model_efficiency_summary.txt"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "Model Efficiency Benchmark",
        "",
        f"GPU: {torch.cuda.get_device_name(0)}",
        f"PyTorch: {torch.__version__}",
        f"Input: {H}x{W}",
        f"Batch size: {BATCH}",
        "Precision: FP16 autocast",
        f"Warm-up runs: {WARMUP}",
        f"Timed runs: {RUNS}",
        "Latency excludes image loading, preprocessing and postprocessing.",
        "MACs are THOP estimates; custom or element-wise operators may be under-counted.",
        "",
    ]

    for r in rows:
        lines += [
            r["model"],
            f"  Parameters: {r['parameters_m']:.6f} M",
            f"  MACs: {r['macs_g']:.6f} G",
            f"  Approx. FLOPs: {r['approx_flops_g']:.6f} G",
            f"  State dict: {r['comparable_state_dict_mb']:.3f} MB",
            f"  Mean latency: {r['latency_mean_ms']:.3f} ms",
            f"  P95 latency: {r['latency_p95_ms']:.3f} ms",
            f"  FPS: {r['fps']:.3f}",
            f"  Peak VRAM: {r['peak_memory_mb']:.3f} MB",
            "",
        ]

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("All benchmarks completed.")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Summary: {txt_path}")
    print("=" * 70)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    missing = [str(path) for _, _, path in SPECS if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing paths:\n" + "\n".join(missing))

    device = torch.device("cuda")

    print("=" * 70)
    print("SegFormer model efficiency benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: {H}x{W}, batch size: {BATCH}")
    print(f"Warm-up/timed runs: {WARMUP}/{RUNS}")
    print("=" * 70)

    rows = [
        benchmark(name, loader, source, device)
        for name, loader, source in SPECS
    ]

    save_results(rows)


if __name__ == "__main__":
    main()
