from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from descatternet_model_3stage import build_descatternet
from glcme_preprocess_3stage import preprocess_glcme_image, read_gray_image, sliding_window_boxes

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    condition: str
    basename: str
    scatter_path: str
    gt_path: str
    fiber_path: Optional[str] = None


def list_image_files(folder: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files[p.stem] = p
    return files



def discover_condition_dirs(scattering_root: Path) -> List[Path]:
    condition_dirs: List[Path] = []
    for p in sorted(scattering_root.iterdir()):
        if p.is_dir():
            imgs = [q for q in p.iterdir() if q.is_file() and q.suffix.lower() in VALID_EXTS]
            if imgs:
                condition_dirs.append(p)
    if not condition_dirs:
        raise FileNotFoundError(f"No condition folders with images were found under: {scattering_root}")
    return condition_dirs



def _sort_name(x: str):
    return int(x) if x.isdigit() else x



def build_samples(scattering_root: Path, gt_root: Path, fiber_root: Optional[Path]) -> List[Sample]:
    gt_map = list_image_files(gt_root)
    if not gt_map:
        raise FileNotFoundError(f"No GT images found in: {gt_root}")

    fiber_map = list_image_files(fiber_root) if fiber_root is not None else {}

    samples: List[Sample] = []
    missing_gt: List[Tuple[str, str]] = []
    missing_fiber: List[Tuple[str, str]] = []

    for cond_dir in discover_condition_dirs(scattering_root):
        condition = cond_dir.name
        scatter_map = list_image_files(cond_dir)
        common = sorted(set(scatter_map.keys()).intersection(gt_map.keys()), key=_sort_name)
        for name in common:
            fiber_path = str(fiber_map[name]) if name in fiber_map else None
            if fiber_root is not None and fiber_path is None:
                missing_fiber.append((condition, name))
            samples.append(
                Sample(
                    condition=condition,
                    basename=name,
                    scatter_path=str(scatter_map[name]),
                    gt_path=str(gt_map[name]),
                    fiber_path=fiber_path,
                )
            )

        for name in sorted(set(scatter_map.keys()) - set(gt_map.keys()), key=_sort_name):
            missing_gt.append((condition, name))

    if not samples:
        raise RuntimeError("No matched scattering/GT pairs were found. Please check whether file names match.")

    if missing_gt:
        print(f"[Warn] {len(missing_gt)} scattering images were skipped because matching GT files were not found.")
        preview = ", ".join([f"{c}/{n}" for c, n in missing_gt[:10]])
        if preview:
            print(f"[Warn] Missing GT examples: {preview}")

    if fiber_root is not None and missing_fiber:
        print(f"[Warn] {len(missing_fiber)} samples do not have matching fiber images. Mid-stage metrics will be omitted for them.")
        preview = ", ".join([f"{c}/{n}" for c, n in missing_fiber[:10]])
        if preview:
            print(f"[Warn] Missing fiber examples: {preview}")

    return samples



def parse_harmonics_flexible(obj) -> Tuple[int, ...]:
    if isinstance(obj, (tuple, list)):
        if len(obj) == 0:
            raise ValueError("gate_harmonics cannot be empty.")
        return tuple(int(v) for v in obj)
    text = str(obj).strip()
    text = text.strip("()[]")
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("gate_harmonics cannot be empty.")
    return tuple(int(v) for v in values)



def tensor_from_patch(patch_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(patch_u8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return x.to(device)



def extract_config(ckpt) -> Dict:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        return dict(ckpt["config"])
    return {}



def resolve_model_params(config: Dict, args) -> Dict:
    return {
        "base_channels": int(config.get("base_channels", args.base_channels)),
        "dropout": float(config.get("dropout", args.dropout)),
        "gate_a_px_input": float(config.get("gate_a_px_input", args.gate_a_px_input)),
        "gate_bandwidth": float(config.get("gate_bandwidth", args.gate_bandwidth)),
        "gate_harmonics": parse_harmonics_flexible(config.get("gate_harmonics", args.gate_harmonics)),
        "gate_alpha_max": float(config.get("gate_alpha_max", args.gate_alpha_max)),
        "gate_init_alpha": float(config.get("gate_init_alpha", args.gate_init_alpha)),
        "gate_per_channel": not bool(config.get("gate_shared_alpha", args.gate_shared_alpha)),
    }



def load_joint_models(args, device: torch.device):
    if args.weights_joint:
        joint_path = Path(args.weights_joint)
        ckpt = torch.load(joint_path, map_location=device)
        if not isinstance(ckpt, dict) or "descatter_state_dict" not in ckpt or "dehoney_state_dict" not in ckpt:
            raise ValueError(f"Joint checkpoint format is invalid: {joint_path}")
        params = resolve_model_params(extract_config(ckpt), args)
        descatter = build_descatternet(
            base_channels=params["base_channels"],
            dropout=params["dropout"],
            batchnorm=True,
            use_spectral_gate=False,
        ).to(device)
        dehoney = build_descatternet(
            base_channels=params["base_channels"],
            dropout=params["dropout"],
            batchnorm=True,
            use_spectral_gate=True,
            gate_a_px_input=params["gate_a_px_input"],
            gate_bandwidth=params["gate_bandwidth"],
            gate_harmonics=params["gate_harmonics"],
            gate_alpha_max=params["gate_alpha_max"],
            gate_init_alpha=params["gate_init_alpha"],
            gate_per_channel=params["gate_per_channel"],
        ).to(device)
        descatter.load_state_dict(ckpt["descatter_state_dict"], strict=True)
        dehoney.load_state_dict(ckpt["dehoney_state_dict"], strict=True)
        descatter.eval()
        dehoney.eval()
        return descatter, dehoney, params

    if not args.descatter_weights or not args.dehoney_weights:
        raise ValueError("You must provide either --weights_joint or both --descatter_weights and --dehoney_weights.")

    desc_ckpt = torch.load(Path(args.descatter_weights), map_location=device)
    deh_ckpt = torch.load(Path(args.dehoney_weights), map_location=device)
    merged_cfg = extract_config(desc_ckpt)
    merged_cfg.update(extract_config(deh_ckpt))
    params = resolve_model_params(merged_cfg, args)

    descatter = build_descatternet(
        base_channels=params["base_channels"],
        dropout=params["dropout"],
        batchnorm=True,
        use_spectral_gate=False,
    ).to(device)
    dehoney = build_descatternet(
        base_channels=params["base_channels"],
        dropout=params["dropout"],
        batchnorm=True,
        use_spectral_gate=True,
        gate_a_px_input=params["gate_a_px_input"],
        gate_bandwidth=params["gate_bandwidth"],
        gate_harmonics=params["gate_harmonics"],
        gate_alpha_max=params["gate_alpha_max"],
        gate_init_alpha=params["gate_init_alpha"],
        gate_per_channel=params["gate_per_channel"],
    ).to(device)

    desc_state = desc_ckpt.get("model_state_dict", desc_ckpt.get("state_dict", desc_ckpt))
    deh_state = deh_ckpt.get("model_state_dict", deh_ckpt.get("state_dict", deh_ckpt))
    descatter.load_state_dict(desc_state, strict=True)
    dehoney.load_state_dict(deh_state, strict=True)
    descatter.eval()
    dehoney.eval()
    return descatter, dehoney, params


@torch.no_grad()
def run_sliding_window_inference(
    descatter_model: torch.nn.Module,
    dehoney_model: torch.nn.Module,
    input_img_u8: np.ndarray,
    crop_size: int,
    stride: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = input_img_u8.shape
    boxes = sliding_window_boxes((h, w), crop_size=crop_size, stride=stride)

    accum_mid = np.zeros((h, w), dtype=np.float32)
    accum_final = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for y1, x1, y2, x2 in boxes:
        patch = input_img_u8[y1:y2, x1:x2]
        x = tensor_from_patch(patch, device)
        mid_patch = descatter_model(x).squeeze().cpu().numpy().astype(np.float32)
        final_patch = dehoney_model(torch.from_numpy(mid_patch).unsqueeze(0).unsqueeze(0).to(device)).squeeze().cpu().numpy().astype(np.float32)
        accum_mid[y1:y2, x1:x2] += mid_patch
        accum_final[y1:y2, x1:x2] += final_patch
        count[y1:y2, x1:x2] += 1.0

    mid_pred = accum_mid / np.maximum(count, 1e-8)
    final_pred = accum_final / np.maximum(count, 1e-8)
    return np.clip(mid_pred, 0.0, 1.0), np.clip(final_pred, 0.0, 1.0)


@torch.no_grad()
def run_center_crop_inference(
    descatter_model: torch.nn.Module,
    dehoney_model: torch.nn.Module,
    input_img_u8: np.ndarray,
    crop_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = input_img_u8.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} exceeds image size {(h, w)}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    patch = input_img_u8[top:top + crop_size, left:left + crop_size]
    x = tensor_from_patch(patch, device)
    mid_pred = descatter_model(x).squeeze().cpu().numpy().astype(np.float32)
    final_pred = dehoney_model(torch.from_numpy(mid_pred).unsqueeze(0).unsqueeze(0).to(device)).squeeze().cpu().numpy().astype(np.float32)
    return np.clip(mid_pred, 0.0, 1.0), np.clip(final_pred, 0.0, 1.0)



def crop_center(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} exceeds image size {(h, w)}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[top:top + crop_size, left:left + crop_size]



def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)
    mse = float(np.mean((pred - gt) ** 2))
    ssim = float(structural_similarity(gt, pred, data_range=1.0))
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
    return mse, ssim, psnr



def save_gray_float(img: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(save_path)



def save_panel(images: Sequence[np.ndarray], save_path: Path) -> None:
    panel = np.concatenate(images, axis=1)
    save_gray_float(panel, save_path)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a trained three-stage FIDER-Net. The script performs online GLCME on scattering inputs, "
                    "runs descatter and dehoney subnets in sequence, and evaluates final output against GT."
    )
    parser.add_argument("--val_scatter_root", required=True, type=str,
                        help="Root folder containing per-condition scattering images.")
    parser.add_argument("--val_gt_root", required=True, type=str,
                        help="Folder containing GT images. File names must match those under each condition.")
    parser.add_argument("--val_fiber_root", default=None, type=str,
                        help="Optional folder containing fiber-only images. If provided, mid-stage metrics are also computed against fiber GLCME.")

    parser.add_argument("--weights_joint", default=None, type=str,
                        help="Path to stage3 joint checkpoint, e.g. stage3_joint/weights/best.pt")
    parser.add_argument("--descatter_weights", default=None, type=str,
                        help="Optional path to descatter checkpoint, e.g. best_descatter.pt")
    parser.add_argument("--dehoney_weights", default=None, type=str,
                        help="Optional path to dehoney checkpoint, e.g. best_dehoney.pt")

    parser.add_argument("--output_dir", default="outputs/validation", type=str)
    parser.add_argument("--inference_mode", default="sliding", choices=["sliding", "center_crop"])
    parser.add_argument("--crop_size", default=1024, type=int)
    parser.add_argument("--stride", default=224, type=int)
    parser.add_argument("--tiles_y", default=8, type=int)
    parser.add_argument("--tiles_x", default=8, type=int)
    parser.add_argument("--clip_limit_ratio", default=0.002, type=float)
    parser.add_argument("--base_channels", default=16, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--gate_a_px_input", default=4.0, type=float)
    parser.add_argument("--gate_bandwidth", default=0.06, type=float)
    parser.add_argument("--gate_harmonics", default="1,2", type=str)
    parser.add_argument("--gate_alpha_max", default=0.7, type=float)
    parser.add_argument("--gate_init_alpha", default=0.0, type=float)
    parser.add_argument("--gate_shared_alpha", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_glcme_inputs", action="store_true")
    parser.add_argument("--save_mid_predictions", action="store_true")
    return parser.parse_args()



def main():
    args = parse_args()
    scatter_root = Path(args.val_scatter_root)
    gt_root = Path(args.val_gt_root)
    fiber_root = Path(args.val_fiber_root) if args.val_fiber_root else None
    output_dir = Path(args.output_dir)

    final_dir = output_dir / "final_predictions"
    mid_dir = output_dir / "mid_predictions"
    panel_dir = output_dir / "comparisons"
    glcme_dir = output_dir / "glcme_inputs"

    final_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mid_predictions:
        mid_dir.mkdir(parents=True, exist_ok=True)
    if args.save_glcme_inputs:
        glcme_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(scatter_root, gt_root, fiber_root)
    device = torch.device(args.device)
    descatter_model, dehoney_model, params = load_joint_models(args, device)

    print(f"[Info] Found {len(samples)} matched scattering/GT pairs across {len(discover_condition_dirs(scatter_root))} conditions.")
    print(f"[Info] Device: {device}")
    print(f"[Info] Inference mode: {args.inference_mode}")
    print(
        f"[Info] Model params | base_channels={params['base_channels']} dropout={params['dropout']} "
        f"gate_harmonics={params['gate_harmonics']} gate_a_px_input={params['gate_a_px_input']}"
    )

    rows: List[dict] = []
    by_condition = defaultdict(lambda: {
        "count": 0,
        "final_mse": 0.0,
        "final_ssim": 0.0,
        "final_psnr": 0.0,
        "mid_count": 0,
        "mid_mse": 0.0,
        "mid_ssim": 0.0,
        "mid_psnr": 0.0,
    })

    for idx, sample in enumerate(samples, start=1):
        raw_scatter_u8 = read_gray_image(sample.scatter_path)
        gt_u8 = read_gray_image(sample.gt_path)
        scatter_glcme_u8 = preprocess_glcme_image(
            raw_scatter_u8,
            num_tiles=(args.tiles_y, args.tiles_x),
            clip_limit_ratio=args.clip_limit_ratio,
            return_uint8=True,
        )

        if args.inference_mode == "center_crop":
            mid_pred, final_pred = run_center_crop_inference(descatter_model, dehoney_model, scatter_glcme_u8, args.crop_size, device)
            raw_vis = crop_center(raw_scatter_u8, args.crop_size).astype(np.float32) / 255.0
            scatter_glcme_vis = crop_center(scatter_glcme_u8, args.crop_size).astype(np.float32) / 255.0
            gt_vis = crop_center(gt_u8, args.crop_size).astype(np.float32) / 255.0
        else:
            mid_pred, final_pred = run_sliding_window_inference(descatter_model, dehoney_model, scatter_glcme_u8, args.crop_size, args.stride, device)
            raw_vis = raw_scatter_u8.astype(np.float32) / 255.0
            scatter_glcme_vis = scatter_glcme_u8.astype(np.float32) / 255.0
            gt_vis = gt_u8.astype(np.float32) / 255.0

        final_mse, final_ssim, final_psnr = compute_metrics(final_pred, gt_vis)

        mid_mse = mid_ssim = mid_psnr = None
        fiber_glcme_vis = None
        if sample.fiber_path is not None:
            fiber_u8 = read_gray_image(sample.fiber_path)
            fiber_glcme_u8 = preprocess_glcme_image(
                fiber_u8,
                num_tiles=(args.tiles_y, args.tiles_x),
                clip_limit_ratio=args.clip_limit_ratio,
                return_uint8=True,
            )
            fiber_glcme_vis = crop_center(fiber_glcme_u8, args.crop_size).astype(np.float32) / 255.0 if args.inference_mode == "center_crop" else fiber_glcme_u8.astype(np.float32) / 255.0
            mid_mse, mid_ssim, mid_psnr = compute_metrics(mid_pred, fiber_glcme_vis)

        stem = f"{sample.condition}_{sample.basename}"
        final_path = final_dir / f"{stem}.png"
        panel_path = panel_dir / f"{stem}_panel.png"
        save_gray_float(final_pred, final_path)
        if args.save_mid_predictions:
            save_gray_float(mid_pred, mid_dir / f"{stem}_mid.png")
        if args.save_glcme_inputs:
            save_gray_float(scatter_glcme_vis, glcme_dir / f"{stem}_scatter_glcme.png")
            if fiber_glcme_vis is not None:
                save_gray_float(fiber_glcme_vis, glcme_dir / f"{stem}_fiber_glcme.png")

        panel_images = [raw_vis, scatter_glcme_vis, mid_pred]
        if fiber_glcme_vis is not None:
            panel_images.append(fiber_glcme_vis)
        panel_images.extend([final_pred, gt_vis])
        save_panel(panel_images, panel_path)

        row = {
            "index": idx,
            "condition": sample.condition,
            "basename": sample.basename,
            "scatter_path": sample.scatter_path,
            "fiber_path": sample.fiber_path or "",
            "gt_path": sample.gt_path,
            "final_mse": final_mse,
            "final_ssim": final_ssim,
            "final_psnr": final_psnr,
            "mid_mse": "" if mid_mse is None else mid_mse,
            "mid_ssim": "" if mid_ssim is None else mid_ssim,
            "mid_psnr": "" if mid_psnr is None else mid_psnr,
            "final_prediction_path": str(final_path),
            "comparison_path": str(panel_path),
        }
        rows.append(row)

        stats = by_condition[sample.condition]
        stats["count"] += 1
        stats["final_mse"] += final_mse
        stats["final_ssim"] += final_ssim
        stats["final_psnr"] += final_psnr
        if mid_mse is not None:
            stats["mid_count"] += 1
            stats["mid_mse"] += mid_mse
            stats["mid_ssim"] += mid_ssim
            stats["mid_psnr"] += mid_psnr

        msg = (
            f"[Val] {idx:03d}/{len(samples):03d} | {sample.condition}/{sample.basename} "
            f"| final: MSE={final_mse:.6f} SSIM={final_ssim:.4f} PSNR={final_psnr:.3f}"
        )
        if mid_mse is not None:
            msg += f" | mid: PSNR={mid_psnr:.3f}"
        print(msg)

    metrics_csv = output_dir / "metrics_per_sample.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = output_dir / "metrics_by_condition.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "count", "final_mse", "final_ssim", "final_psnr", "mid_count", "mid_mse", "mid_ssim", "mid_psnr"],
        )
        writer.writeheader()
        for condition, stats in sorted(by_condition.items()):
            count = max(1, int(stats["count"]))
            mid_count = max(1, int(stats["mid_count"])) if int(stats["mid_count"]) > 0 else 0
            writer.writerow(
                {
                    "condition": condition,
                    "count": int(stats["count"]),
                    "final_mse": float(stats["final_mse"] / count),
                    "final_ssim": float(stats["final_ssim"] / count),
                    "final_psnr": float(stats["final_psnr"] / count),
                    "mid_count": int(stats["mid_count"]),
                    "mid_mse": "" if mid_count == 0 else float(stats["mid_mse"] / mid_count),
                    "mid_ssim": "" if mid_count == 0 else float(stats["mid_ssim"] / mid_count),
                    "mid_psnr": "" if mid_count == 0 else float(stats["mid_psnr"] / mid_count),
                }
            )

    overall_final_mse = float(np.mean([float(r["final_mse"]) for r in rows]))
    overall_final_ssim = float(np.mean([float(r["final_ssim"]) for r in rows]))
    overall_final_psnr = float(np.mean([float(r["final_psnr"]) for r in rows]))

    mid_rows = [r for r in rows if r["mid_mse"] != ""]
    overall_mid_mse = float(np.mean([float(r["mid_mse"]) for r in mid_rows])) if mid_rows else None
    overall_mid_ssim = float(np.mean([float(r["mid_ssim"]) for r in mid_rows])) if mid_rows else None
    overall_mid_psnr = float(np.mean([float(r["mid_psnr"]) for r in mid_rows])) if mid_rows else None

    overall_txt = output_dir / "overall_metrics.txt"
    with open(overall_txt, "w", encoding="utf-8") as f:
        f.write(f"Final MSE  : {overall_final_mse:.6f}\n")
        f.write(f"Final SSIM : {overall_final_ssim:.6f}\n")
        f.write(f"Final PSNR : {overall_final_psnr:.6f} dB\n")
        if overall_mid_mse is not None:
            f.write(f"Mid MSE    : {overall_mid_mse:.6f}\n")
            f.write(f"Mid SSIM   : {overall_mid_ssim:.6f}\n")
            f.write(f"Mid PSNR   : {overall_mid_psnr:.6f} dB\n")

    print("[Done] ===============================")
    print(f"[Done] Final MSE  = {overall_final_mse:.6f}")
    print(f"[Done] Final SSIM = {overall_final_ssim:.4f}")
    print(f"[Done] Final PSNR = {overall_final_psnr:.3f} dB")
    if overall_mid_mse is not None:
        print(f"[Done] Mid   MSE  = {overall_mid_mse:.6f}")
        print(f"[Done] Mid   SSIM = {overall_mid_ssim:.4f}")
        print(f"[Done] Mid   PSNR = {overall_mid_psnr:.3f} dB")
    print(f"[Done] Saved per-sample metrics to: {metrics_csv}")
    print(f"[Done] Saved per-condition metrics to: {summary_csv}")
    print(f"[Done] Saved overall metrics to: {overall_txt}")
    print(f"[Done] Final prediction images: {final_dir}")
    print(f"[Done] Comparison panels: {panel_dir}")


if __name__ == "__main__":
    main()
