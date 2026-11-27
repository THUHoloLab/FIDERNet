
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import torch

from train_stage1_3stage import run_stage1
from train_stage2_3stage import run_stage2
from train_stage3_3stage import run_stage3


def parse_stage_list(stage_str: str) -> List[int]:
    s = stage_str.strip().lower()
    if s == "all":
        return [1, 2, 3]
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val not in (1, 2, 3):
            raise ValueError("stages must be one of 1, 2, 3, or 'all'.")
        out.append(val)
    if not out:
        raise ValueError("At least one stage must be selected.")
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Three-stage training entry: stage1 scatter->fiber, stage2 fiber->GT, stage3 joint fine-tuning.")
    parser.add_argument("--scattering_root", required=True, type=str, help="Root containing scattering conditions.")
    parser.add_argument("--fiber_root", required=True, type=str, help="Root containing fiber-only images.")
    parser.add_argument("--clear_root", required=True, type=str, help="Root containing GT images.")
    parser.add_argument("--output_root", default="runs_torch_3stage", type=str)
    parser.add_argument("--cache_root", default=None, type=str)

    parser.add_argument("--crop_size", default=1024, type=int)
    parser.add_argument("--tiles_y", default=8, type=int)
    parser.add_argument("--tiles_x", default=8, type=int)
    parser.add_argument("--clip_limit_ratio", default=0.002, type=float)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs_stage1", default=50, type=int)
    parser.add_argument("--epochs_stage2", default=50, type=int)
    parser.add_argument("--epochs_stage3", default=50, type=int)
    parser.add_argument("--lr_stage1", default=1e-3, type=float)
    parser.add_argument("--lr_stage2", default=1e-3, type=float)
    parser.add_argument("--lr_stage3", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--lr_patience", default=20, type=int)

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--base_channels", default=16, type=int)
    parser.add_argument("--beta1", default=0.5, type=float)

    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--patches_per_image", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cache_workers", default=4, type=int)
    parser.add_argument("--hflip_prob", default=0.0, type=float)
    parser.add_argument("--vflip_prob", default=0.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--gate_a_px_input", default=4.0, type=float)
    parser.add_argument("--gate_bandwidth", default=0.06, type=float)
    parser.add_argument("--gate_harmonics", default="1,2", type=str)
    parser.add_argument("--gate_alpha_max", default=0.7, type=float)
    parser.add_argument("--gate_init_alpha", default=0.0, type=float)
    parser.add_argument("--gate_shared_alpha", action="store_true")

    parser.add_argument("--stage2_bridge_pred_prob", default=0.5, type=float,
                        help="Probability of using Stage1-predicted fiber-like input during Stage2 bridge training.")
    parser.add_argument("--stage2_bridge_gamma", default=1.0, type=float,
                        help="Mixing strength for Stage1-predicted fiber-like input: input=(1-gamma)*real_fiber + gamma*pred_fiber.")
    parser.add_argument("--stage2_bridge_warmup_epochs", default=10, type=int,
                        help="Number of epochs to linearly ramp stage2 bridge gamma from 0 to stage2_bridge_gamma.")
    parser.add_argument("--stage2_bridge_start_epoch", default=1, type=int,
                        help="Epoch to start Stage2 bridge training.")
    parser.add_argument("--stage2_bridge_eval_pred", action="store_true",
                        help="If set, Stage2 validation reports performance using Stage1-predicted fiber-like input.")

    parser.add_argument("--stage3_freeze_stage1_epochs", default=10, type=int,
                        help="Freeze descatter/Stage1 network for the first N Stage3 epochs.")
    parser.add_argument("--stage3_descatter_lr_scale", default=0.1, type=float,
                        help="Stage1 learning-rate scale during Stage3 joint fine-tuning.")
    parser.add_argument("--stage3_alpha_mid_after_unfreeze", default=1.0, type=float,
                        help="Mid loss weight after Stage1 is unfrozen in Stage3.")

    parser.add_argument("--alpha_mid", default=1.0, type=float)
    parser.add_argument("--beta_final", default=1.0, type=float)

    parser.add_argument("--stages", default="all", type=str, help="Use 'all' or a comma list like 1,2,3")
    parser.add_argument("--stage1_ckpt", default=None, type=str, help="Optional external checkpoint for stage1")
    parser.add_argument("--stage2_ckpt", default=None, type=str, help="Optional external checkpoint for stage2")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_stages = parse_stage_list(args.stages)
    master_run_dir = Path(args.output_root) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    master_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Master] Run dir: {master_run_dir}")
    print(f"[Master] Selected stages: {selected_stages}")

    stage1_result = None
    stage2_result = None
    stage3_result = None

    if 1 in selected_stages:
        stage1_result = run_stage1(args, master_run_dir)
        args.stage1_ckpt = stage1_result["best_ckpt"]

    if 2 in selected_stages:
        stage2_result = run_stage2(args, master_run_dir)
        args.stage2_ckpt = stage2_result["best_ckpt"]

    if 3 in selected_stages:
        stage1_ckpt = args.stage1_ckpt
        stage2_ckpt = args.stage2_ckpt
        if stage1_result is None and stage1_ckpt is None:
            candidate = master_run_dir / "stage1_descatter" / "weights" / "best.pt"
            if candidate.exists():
                stage1_ckpt = str(candidate)
        if stage2_result is None and stage2_ckpt is None:
            candidate = master_run_dir / "stage2_dehoney" / "weights" / "best.pt"
            if candidate.exists():
                stage2_ckpt = str(candidate)
        if stage1_ckpt is None:
            print("[Master] Warning: Stage3 starts without a Stage1 checkpoint.")
        if stage2_ckpt is None:
            print("[Master] Warning: Stage3 starts without a Stage2 checkpoint.")
        stage3_result = run_stage3(args, master_run_dir, stage1_ckpt=stage1_ckpt, stage2_ckpt=stage2_ckpt)

    print("[Master] Finished.")
    if stage1_result:
        print(f"[Master] Stage1 best: {stage1_result['best_ckpt']}")
    if stage2_result:
        print(f"[Master] Stage2 best: {stage2_result['best_ckpt']}")
    if stage3_result:
        print(f"[Master] Stage3 best joint: {stage3_result['best_joint_ckpt']}")


if __name__ == "__main__":
    main()
