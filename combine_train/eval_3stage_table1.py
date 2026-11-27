from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from descatternet_model_3stage import build_descatternet
from glcme_preprocess_3stage import (
    center_paired_crop,
    preprocess_glcme_image,
    read_gray_image,
    sliding_window_boxes,
)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METHOD_NAME = "FIDER-Net full staged bridge"


@dataclass(frozen=True)
class Sample:
    condition: str
    basename: str
    scatter_path: str
    gt_path: str


def list_image_files(folder: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files[p.stem] = p
    return files


def discover_condition_dirs(scattering_root: Path) -> List[Path]:
    direct_images = [p for p in scattering_root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if direct_images:
        return [scattering_root]
    condition_dirs = []
    for p in sorted(scattering_root.iterdir()):
        if p.is_dir():
            imgs = [q for q in p.iterdir() if q.is_file() and q.suffix.lower() in VALID_EXTS]
            if imgs:
                condition_dirs.append(p)
    if not condition_dirs:
        raise FileNotFoundError(f"No image folders were found under: {scattering_root}")
    return condition_dirs


def _sort_name(name: str):
    return int(name) if name.isdigit() else name


def build_samples(basenames: Sequence[str], scattering_root: Path, gt_root: Path) -> List[Sample]:
    gt_map = list_image_files(gt_root)
    if not gt_map:
        raise FileNotFoundError(f"No GT images found in: {gt_root}")

    basename_set = set(basenames)
    samples: List[Sample] = []
    for cond_dir in discover_condition_dirs(scattering_root):
        condition = cond_dir.name
        scatter_map = list_image_files(cond_dir)
        common = sorted(basename_set.intersection(scatter_map.keys()).intersection(gt_map.keys()), key=_sort_name)
        for name in common:
            samples.append(
                Sample(
                    condition=condition,
                    basename=name,
                    scatter_path=str(scatter_map[name]),
                    gt_path=str(gt_map[name]),
                )
            )
    if not samples:
        raise RuntimeError("No matched samples were found. Check split_basenames.json and dataset roots.")
    return samples


def parse_harmonics(obj) -> Tuple[int, ...]:
    if isinstance(obj, (tuple, list)):
        if not obj:
            raise ValueError("gate_harmonics cannot be empty.")
        return tuple(int(v) for v in obj)
    text = str(obj).strip().strip("()[]")
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("gate_harmonics cannot be empty.")
    return tuple(int(v) for v in values)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config(args) -> dict:
    run_dir = Path(args.run_dir)
    config_path = Path(args.config) if args.config else run_dir / "stage3_joint" / "config.json"
    config = load_json(config_path) if config_path.exists() else {}

    if args.scattering_root is not None:
        config["scattering_root"] = args.scattering_root
    if args.clear_root is not None:
        config["clear_root"] = args.clear_root
    if args.crop_size is not None:
        config["crop_size"] = args.crop_size
    if args.tiles_y is not None:
        config["tiles_y"] = args.tiles_y
    if args.tiles_x is not None:
        config["tiles_x"] = args.tiles_x
    if args.clip_limit_ratio is not None:
        config["clip_limit_ratio"] = args.clip_limit_ratio
    if args.base_channels is not None:
        config["base_channels"] = args.base_channels
    if args.dropout is not None:
        config["dropout"] = args.dropout
    return config


def load_joint_models(args, config: dict, device: torch.device):
    checkpoint_path = Path(args.weights_joint) if args.weights_joint else Path(args.run_dir) / "stage3_joint" / "weights" / "best.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_config = dict(ckpt.get("config", {}))
    merged = dict(config)
    merged.update(ckpt_config)

    base_channels = int(merged.get("base_channels", 16))
    dropout = float(merged.get("dropout", 0.1))
    harmonics = parse_harmonics(merged.get("gate_harmonics", "1,2"))
    gate_shared_alpha = bool(merged.get("gate_shared_alpha", False))

    descatter = build_descatternet(
        base_channels=base_channels,
        dropout=dropout,
        batchnorm=True,
        use_spectral_gate=False,
    ).to(device)
    dehoney = build_descatternet(
        base_channels=base_channels,
        dropout=dropout,
        batchnorm=True,
        use_spectral_gate=True,
        gate_a_px_input=float(merged.get("gate_a_px_input", 4.0)),
        gate_bandwidth=float(merged.get("gate_bandwidth", 0.06)),
        gate_harmonics=harmonics,
        gate_alpha_max=float(merged.get("gate_alpha_max", 0.7)),
        gate_init_alpha=float(merged.get("gate_init_alpha", 0.0)),
        gate_per_channel=(not gate_shared_alpha),
    ).to(device)

    descatter.load_state_dict(ckpt["descatter_state_dict"], strict=True)
    dehoney.load_state_dict(ckpt["dehoney_state_dict"], strict=True)
    descatter.eval()
    dehoney.eval()
    return descatter, dehoney, checkpoint_path


def count_trainable_parameters(*models: torch.nn.Module) -> int:
    return sum(p.numel() for model in models for p in model.parameters() if p.requires_grad)


def build_lpips_model(device: torch.device, net: str):
    try:
        import lpips

        return lpips.LPIPS(net=net).to(device).eval(), None
    except Exception as exc:
        return None, str(exc)


def tensor_from_u8(img_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img_u8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)


@torch.no_grad()
def infer_patch(descatter_model, dehoney_model, input_u8: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    x = tensor_from_u8(input_u8, device)
    mid = descatter_model(x)
    final = dehoney_model(mid)
    mid_np = mid.squeeze().detach().cpu().numpy().astype(np.float32)
    final_np = final.squeeze().detach().cpu().numpy().astype(np.float32)
    return np.clip(mid_np, 0.0, 1.0), np.clip(final_np, 0.0, 1.0)


@torch.no_grad()
def run_sliding_inference(descatter_model, dehoney_model, input_u8: np.ndarray, crop_size: int, stride: int, device: torch.device):
    h, w = input_u8.shape
    boxes = sliding_window_boxes((h, w), crop_size=crop_size, stride=stride)
    accum_mid = np.zeros((h, w), dtype=np.float32)
    accum_final = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for y1, x1, y2, x2 in boxes:
        mid_patch, final_patch = infer_patch(descatter_model, dehoney_model, input_u8[y1:y2, x1:x2], device)
        accum_mid[y1:y2, x1:x2] += mid_patch
        accum_final[y1:y2, x1:x2] += final_patch
        count[y1:y2, x1:x2] += 1.0
    return np.clip(accum_mid / np.maximum(count, 1e-8), 0, 1), np.clip(accum_final / np.maximum(count, 1e-8), 0, 1)


def compute_quality(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
    ssim = float(structural_similarity(gt, pred, data_range=1.0))
    return psnr, ssim


@torch.no_grad()
def compute_lpips(lpips_model, pred: np.ndarray, gt: np.ndarray, device: torch.device) -> float:
    if lpips_model is None:
        return float("nan")
    pred_t = torch.from_numpy(np.clip(pred, 0, 1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(np.clip(gt, 0, 1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    pred_rgb = pred_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
    gt_rgb = gt_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
    return float(lpips_model(pred_rgb, gt_rgb).detach().flatten()[0].cpu())


def prepare_eval_pair(sample: Sample, config: dict, inference_mode: str, crop_size: int):
    raw_scatter = read_gray_image(sample.scatter_path)
    gt_u8 = read_gray_image(sample.gt_path)
    glcme_u8 = preprocess_glcme_image(
        raw_scatter,
        num_tiles=(int(config.get("tiles_y", 8)), int(config.get("tiles_x", 8))),
        clip_limit_ratio=float(config.get("clip_limit_ratio", 0.002)),
        return_uint8=True,
    )
    if inference_mode == "center_crop":
        glcme_crop, gt_crop = center_paired_crop(glcme_u8, gt_u8, crop_size)
        return glcme_crop, gt_crop.astype(np.float32) / 255.0
    return glcme_u8, gt_u8.astype(np.float32) / 255.0


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def build_condition_rows(rows: Sequence[dict]) -> List[dict]:
    by_condition: Dict[str, List[dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    condition_rows = []
    for condition, cond_rows in sorted(by_condition.items()):
        condition_rows.append(
            {
                "condition": condition,
                "count": len(cond_rows),
                "psnr": _mean([r["psnr"] for r in cond_rows]),
                "ssim": _mean([r["ssim"] for r in cond_rows]),
                "lpips": _mean([r["lpips"] for r in cond_rows]),
            }
        )
    return condition_rows


def write_metric_details(output_dir: Path, rows: List[dict], condition_rows: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = output_dir / "test_metrics_per_sample.csv"
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_condition_csv = output_dir / "test_metrics_by_condition.csv"
    with open(by_condition_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(condition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(condition_rows)

    print(f"[Metrics] Saved per-sample CSV: {per_sample_csv}", flush=True)
    print(f"[Metrics] Saved by-condition CSV: {by_condition_csv}", flush=True)


@torch.no_grad()
def measure_fps_and_memory(descatter_model, dehoney_model, samples: Sequence[Sample], config: dict, args, device: torch.device):
    if args.skip_fps:
        print("[FPS] Skipped by --skip_fps.", flush=True)
        return float("nan"), float("nan")

    warmup = max(0, int(args.fps_warmup_samples))
    repeats = max(1, int(args.fps_repeats))
    crop_size = int(config.get("crop_size", 1024))
    fps_sample_count = max(1, min(len(samples), int(args.fps_max_samples)))

    print(
        f"[FPS] Start measurement | mode={args.fps_mode} | samples={fps_sample_count} | "
        f"repeats={repeats} | warmup={warmup}",
        flush=True,
    )

    prepared = []
    if args.fps_mode == "network":
        fps_samples = samples[:fps_sample_count]
        print("[FPS] Preparing preprocessed inputs for network-only timing ...", flush=True)
        for idx, sample in enumerate(fps_samples, start=1):
            input_u8, _ = prepare_eval_pair(sample, config, args.inference_mode, crop_size)
            prepared.append(input_u8)
            if idx % max(1, args.log_interval) == 0 or idx == len(fps_samples):
                print(f"[FPS] prepared {idx}/{len(fps_samples)}", flush=True)

        for idx, input_u8 in enumerate(prepared[:warmup], start=1):
            if args.inference_mode == "center_crop":
                infer_patch(descatter_model, dehoney_model, input_u8, device)
            else:
                run_sliding_inference(descatter_model, dehoney_model, input_u8, crop_size, args.stride, device)
            print(f"[FPS] warmup {idx}/{min(warmup, len(prepared))}", flush=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        total = 0
        elapsed = 0.0
        for repeat_idx in range(1, repeats + 1):
            for idx, input_u8 in enumerate(prepared, start=1):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                if args.inference_mode == "center_crop":
                    infer_patch(descatter_model, dehoney_model, input_u8, device)
                else:
                    run_sliding_inference(descatter_model, dehoney_model, input_u8, crop_size, args.stride, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - start
                total += 1
                if idx % max(1, args.log_interval) == 0 or idx == len(prepared):
                    fps_now = total / elapsed if elapsed > 0 else float("nan")
                    print(f"[FPS] repeat {repeat_idx}/{repeats} | {idx}/{len(prepared)} | fps={fps_now:.2f}", flush=True)
    else:
        subset = list(samples[:fps_sample_count])
        for idx, sample in enumerate(subset[:warmup], start=1):
            input_u8, _ = prepare_eval_pair(sample, config, args.inference_mode, crop_size)
            if args.inference_mode == "center_crop":
                infer_patch(descatter_model, dehoney_model, input_u8, device)
            else:
                run_sliding_inference(descatter_model, dehoney_model, input_u8, crop_size, args.stride, device)
            print(f"[FPS] warmup {idx}/{min(warmup, len(subset))}", flush=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        total = 0
        elapsed = 0.0
        for repeat_idx in range(1, repeats + 1):
            for idx, sample in enumerate(subset, start=1):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                input_u8, _ = prepare_eval_pair(sample, config, args.inference_mode, crop_size)
                if args.inference_mode == "center_crop":
                    infer_patch(descatter_model, dehoney_model, input_u8, device)
                else:
                    run_sliding_inference(descatter_model, dehoney_model, input_u8, crop_size, args.stride, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - start
                total += 1
                if idx % max(1, args.log_interval) == 0 or idx == len(subset):
                    fps_now = total / elapsed if elapsed > 0 else float("nan")
                    print(f"[FPS] repeat {repeat_idx}/{repeats} | {idx}/{len(subset)} | fps={fps_now:.2f}", flush=True)

    fps = float(total / elapsed) if elapsed > 0 else float("nan")
    peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2)) if device.type == "cuda" else float("nan")
    print(f"[FPS] Done | fps={fps:.2f} | peak_memory_mb={peak_memory_mb:.1f}", flush=True)
    return fps, peak_memory_mb


def write_table_outputs(run_dir: Path, output_dir: Path, summary: dict, rows: List[dict], condition_rows: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metric_details(output_dir, rows, condition_rows)

    table_csv = output_dir / "table1_fider_net_full.csv"
    fieldnames = ["method", "psnr", "ssim", "lpips", "fps", "params_m", "gpu_memory_mb", "test_samples", "split", "checkpoint", "fps_mode", "inference_mode"]
    with open(table_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in fieldnames})

    table_json = output_dir / "table1_fider_net_full.json"
    table_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def fmt(value, digits):
        try:
            if math.isnan(float(value)):
                return "NaN"
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    md_lines = [
        "| Method | PSNR | SSIM | LPIPS | FPS | Params (M) | GPU Mem (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        (
            f"| {summary['method']} | {fmt(summary['psnr'], 3)} | {fmt(summary['ssim'], 4)} | "
            f"{fmt(summary['lpips'], 4)} | {fmt(summary['fps'], 2)} | {fmt(summary['params_m'], 3)} | "
            f"{fmt(summary['gpu_memory_mb'], 1)} |"
        ),
        "",
    ]
    table_md = output_dir / "table1_fider_net_full.md"
    table_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Also mirror the compact Table 1 row into the run directory for quick discovery.
    if output_dir != run_dir:
        (run_dir / table_csv.name).write_text(table_csv.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / table_md.name).write_text(table_md.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / table_json.name).write_text(table_json.read_text(encoding="utf-8"), encoding="utf-8")

    print("[Metrics] Table 1 row:")
    print("\n".join(md_lines[:3]))
    print(f"[Metrics] Saved summary CSV: {table_csv}")
    print(f"[Metrics] Saved summary Markdown: {table_md}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a full GLCME + FIDER-Net run and export Table 1 metrics.")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--method_name", default=METHOD_NAME, type=str)
    parser.add_argument("--weights_joint", default=None, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--split_json", default=None, type=str)
    parser.add_argument("--split_key", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scattering_root", default=None, type=str)
    parser.add_argument("--clear_root", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--inference_mode", default="center_crop", choices=["center_crop", "sliding"])
    parser.add_argument("--crop_size", default=None, type=int)
    parser.add_argument("--stride", default=224, type=int)
    parser.add_argument("--tiles_y", default=None, type=int)
    parser.add_argument("--tiles_x", default=None, type=int)
    parser.add_argument("--clip_limit_ratio", default=None, type=float)
    parser.add_argument("--base_channels", default=None, type=int)
    parser.add_argument("--dropout", default=None, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--fps_mode", default="network", choices=["network", "full_pipeline"])
    parser.add_argument("--fps_warmup_samples", default=3, type=int)
    parser.add_argument("--fps_repeats", default=1, type=int)
    parser.add_argument("--fps_max_samples", default=1200, type=int)
    parser.add_argument("--skip_fps", action="store_true", help="Skip FPS/memory measurement and only export quality metrics.")
    parser.add_argument("--log_interval", default=20, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "table1_full_eval"
    config = resolve_config(args)

    split_json = Path(args.split_json) if args.split_json else run_dir / "split_basenames.json"
    split_info = load_json(split_json)
    basenames = list(split_info[args.split_key])

    scattering_root = Path(config.get("scattering_root", "datasets/train_scatter"))
    clear_root = Path(config.get("clear_root", "datasets/train_GT"))
    crop_size = int(config.get("crop_size", 1024))

    samples = build_samples(basenames, scattering_root, clear_root)
    device = torch.device(args.device)
    descatter_model, dehoney_model, checkpoint_path = load_joint_models(args, config, device)
    lpips_model, lpips_error = build_lpips_model(device, args.lpips_net)
    if lpips_error:
        print(f"[Metrics] LPIPS unavailable; writing NaN values. Reason: {lpips_error}")

    params = count_trainable_parameters(descatter_model, dehoney_model)
    print(f"[Info] Run dir: {run_dir}")
    print(f"[Info] Checkpoint: {checkpoint_path}")
    print(f"[Info] Split: {args.split_key} | samples: {len(samples)} | scattering_root={scattering_root} | clear_root={clear_root}")
    print(f"[Info] Inference mode: {args.inference_mode} | fps_mode={args.fps_mode} | crop_size={crop_size}")

    rows: List[dict] = []
    for idx, sample in enumerate(samples, start=1):
        input_u8, gt_eval = prepare_eval_pair(sample, config, args.inference_mode, crop_size)
        if args.inference_mode == "center_crop":
            _, final_pred = infer_patch(descatter_model, dehoney_model, input_u8, device)
        else:
            _, final_pred = run_sliding_inference(descatter_model, dehoney_model, input_u8, crop_size, args.stride, device)

        psnr, ssim = compute_quality(final_pred, gt_eval)
        lpips_val = compute_lpips(lpips_model, final_pred, gt_eval, device)
        rows.append(
            {
                "method": args.method_name,
                "split": args.split_key,
                "condition": sample.condition,
                "basename": sample.basename,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_val,
                "scatter_path": sample.scatter_path,
                "gt_path": sample.gt_path,
            }
        )
        if idx % max(1, args.log_interval) == 0 or idx == len(samples):
            print(f"[Eval] {idx}/{len(samples)} | {sample.condition}/{sample.basename} | PSNR={psnr:.3f} SSIM={ssim:.4f} LPIPS={lpips_val:.4f}")

    condition_rows = build_condition_rows(rows)
    write_metric_details(output_dir, rows, condition_rows)

    fps, gpu_memory_mb = measure_fps_and_memory(descatter_model, dehoney_model, samples, config, args, device)
    summary = {
        "method": args.method_name,
        "split": args.split_key,
        "test_samples": len(rows),
        "psnr": _mean([r["psnr"] for r in rows]),
        "ssim": _mean([r["ssim"] for r in rows]),
        "lpips": _mean([r["lpips"] for r in rows]),
        "fps": fps,
        "params_m": float(params / 1_000_000.0),
        "gpu_memory_mb": gpu_memory_mb,
        "checkpoint": str(checkpoint_path),
        "fps_mode": args.fps_mode,
        "inference_mode": args.inference_mode,
        "lpips_note": "" if lpips_error is None else f"LPIPS unavailable: {lpips_error}",
    }
    write_table_outputs(run_dir, output_dir, summary, rows, condition_rows)


if __name__ == "__main__":
    main()
