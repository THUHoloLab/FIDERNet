from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

from descatternet_model_3stage import build_descatternet
from glcme_preprocess_3stage import preprocess_glcme_image, read_gray_image, sliding_window_boxes

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_harmonics_flexible(obj) -> Tuple[int, ...]:
    if isinstance(obj, (tuple, list)):
        if len(obj) == 0:
            raise ValueError("gate_harmonics cannot be empty.")
        return tuple(int(v) for v in obj)
    text = str(obj).strip().strip("()[]")
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("gate_harmonics cannot be empty.")
    return tuple(int(v) for v in values)


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


def tensor_from_patch(patch_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(patch_u8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return x.to(device)


@torch.no_grad()
def run_sliding_window_inference(
    descatter_model: torch.nn.Module,
    dehoney_model: torch.nn.Module,
    input_img_u8: np.ndarray,
    crop_size: int,
    stride: int,
    device: torch.device,
):
    h, w = input_img_u8.shape
    boxes = sliding_window_boxes((h, w), crop_size=crop_size, stride=stride)

    accum_mid = np.zeros((h, w), dtype=np.float32)
    accum_final = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for y1, x1, y2, x2 in boxes:
        patch = input_img_u8[y1:y2, x1:x2]
        x = tensor_from_patch(patch, device)
        mid_t = descatter_model(x)
        final_t = dehoney_model(mid_t)
        mid_np = mid_t.squeeze().cpu().numpy().astype(np.float32)
        final_np = final_t.squeeze().cpu().numpy().astype(np.float32)
        accum_mid[y1:y2, x1:x2] += mid_np
        accum_final[y1:y2, x1:x2] += final_np
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
):
    h, w = input_img_u8.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"crop_size={crop_size} exceeds image size {(h, w)}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    patch = input_img_u8[top:top + crop_size, left:left + crop_size]
    x = tensor_from_patch(patch, device)
    mid_t = descatter_model(x)
    final_t = dehoney_model(mid_t)
    return (
        np.clip(mid_t.squeeze().cpu().numpy().astype(np.float32), 0.0, 1.0),
        np.clip(final_t.squeeze().cpu().numpy().astype(np.float32), 0.0, 1.0),
    )


def save_gray_float(img: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(save_path)


def save_panel(images, save_path: Path) -> None:
    panel = np.concatenate(images, axis=1)
    save_gray_float(panel, save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo for trained three-stage FIDER-Net. Input one raw fiber-bundle image and output one restored image."
    )
    parser.add_argument("--input_image", required=True, type=str, help="Path to one raw fiber-bundle image.")
    parser.add_argument("--weights_joint", default=None, type=str, help="Path to the trained stage3_joint/weights/best.pt checkpoint")
    parser.add_argument("--descatter_weights", default=None, type=str, help="Optional separate descatter checkpoint")
    parser.add_argument("--dehoney_weights", default=None, type=str, help="Optional separate dehoney checkpoint")
    parser.add_argument("--output_dir", default="demo_outputs", type=str)
    parser.add_argument("--output_name", default=None, type=str, help="Optional base name for output files; defaults to the input filename")
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
    parser.add_argument("--no_glcme", action="store_true", help="Feed the raw scattering image directly. Required for stage3_joint_raw checkpoints trained without Local-GLCME.")
    parser.add_argument("--save_glcme_input", action="store_true")
    parser.add_argument("--save_mid_prediction", action="store_true")
    parser.add_argument("--save_panel", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_image)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if input_path.suffix.lower() not in VALID_EXTS:
        raise ValueError(f"Unsupported image extension: {input_path.suffix}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    descatter_model, dehoney_model, params = load_joint_models(args, device)

    raw_u8 = read_gray_image(input_path)
    if args.no_glcme:
        model_input_u8 = raw_u8
    else:
        model_input_u8 = preprocess_glcme_image(
            raw_u8,
            num_tiles=(args.tiles_y, args.tiles_x),
            clip_limit_ratio=args.clip_limit_ratio,
            return_uint8=True,
        )

    if args.inference_mode == "center_crop":
        mid_pred, final_pred = run_center_crop_inference(
            descatter_model,
            dehoney_model,
            model_input_u8,
            args.crop_size,
            device,
        )
        h, w = model_input_u8.shape
        top = (h - args.crop_size) // 2
        left = (w - args.crop_size) // 2
        raw_vis = raw_u8[top:top + args.crop_size, left:left + args.crop_size].astype(np.float32) / 255.0
        input_vis = model_input_u8[top:top + args.crop_size, left:left + args.crop_size].astype(np.float32) / 255.0
    else:
        mid_pred, final_pred = run_sliding_window_inference(
            descatter_model,
            dehoney_model,
            model_input_u8,
            args.crop_size,
            args.stride,
            device,
        )
        raw_vis = raw_u8.astype(np.float32) / 255.0
        input_vis = model_input_u8.astype(np.float32) / 255.0

    stem = args.output_name.strip() if args.output_name else input_path.stem
    final_path = output_dir / f"{stem}_restored.png"
    save_gray_float(final_pred, final_path)

    if args.save_mid_prediction:
        save_gray_float(mid_pred, output_dir / f"{stem}_mid.png")
    if args.save_glcme_input:
        input_name = "raw_input" if args.no_glcme else "glcme"
        save_gray_float(input_vis, output_dir / f"{stem}_{input_name}.png")
    if args.save_panel:
        panel_images = [raw_vis, input_vis, mid_pred, final_pred]
        save_panel(panel_images, output_dir / f"{stem}_panel.png")

    print("[Done] ============================================")
    print(f"[Done] Input image  : {input_path}")
    print(f"[Done] Output image : {final_path}")
    if args.save_mid_prediction:
        print(f"[Done] Mid image    : {output_dir / f'{stem}_mid.png'}")
    if args.save_glcme_input:
        input_name = "raw_input" if args.no_glcme else "glcme"
        print(f"[Done] Model input  : {output_dir / f'{stem}_{input_name}.png'}")
    if args.save_panel:
        print(f"[Done] Panel image  : {output_dir / f'{stem}_panel.png'}")
    print(f"[Done] Device       : {device}")
    print(
        "[Done] Model params : "
        f"base_channels={params['base_channels']}, "
        f"dropout={params['dropout']}, "
        f"gate_harmonics={params['gate_harmonics']}, "
        f"gate_a_px_input={params['gate_a_px_input']}"
    )
    print(f"[Done] Input mode   : {'raw scatter' if args.no_glcme else 'Local-GLCME'}")


if __name__ == "__main__":
    main()
