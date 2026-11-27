
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from descatternet_model_3stage import build_descatternet
from train_pipeline_common_3stage import (
    Stage1Dataset,
    build_scheduler,
    evaluate_pair,
    make_dataloader,
    namespace_to_config,
    prepare_context,
    save_history_csv,
    save_history_plots,
    save_json,
    set_seed,
    train_one_epoch_pair,
)


def run_stage1(args, master_run_dir: Optional[Path] = None):
    set_seed(args.seed)
    if master_run_dir is None:
        master_run_dir = Path(args.output_root) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    master_run_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = master_run_dir / "stage1_descatter"
    weights_dir = stage_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    context = prepare_context(args, master_run_dir)
    train_dataset = Stage1Dataset(context.train_stage13, args.crop_size, "train", args.patches_per_image, args.hflip_prob, args.vflip_prob)
    val_dataset = Stage1Dataset(context.val_stage13, args.crop_size, "val", 1, 0.0, 0.0)
    train_loader = make_dataloader(train_dataset, args.batch_size, args.num_workers, True)
    val_loader = make_dataloader(val_dataset, args.batch_size, args.num_workers, False)

    device = torch.device(args.device)
    model = build_descatternet(base_channels=args.base_channels, dropout=args.dropout, batchnorm=True, use_spectral_gate=False).to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr_stage1, betas=(args.beta1, 0.999))
    scheduler = build_scheduler(optimizer, args.min_lr, args.lr_patience)
    use_amp = torch.cuda.is_available() and (not args.no_amp)

    save_json(namespace_to_config(args), stage_dir / "config.json")
    history = []
    best_val_loss = float("inf")

    print(f"[Stage1] Train samples: {len(context.train_stage13)} | Val samples: {len(context.val_stage13)}")
    for epoch in range(1, args.epochs_stage1 + 1):
        train_loss = train_one_epoch_pair(model, train_loader, optimizer, criterion, device, use_amp)
        val_metrics = evaluate_pair(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_ssim": val_metrics["ssim"],
            "val_psnr": val_metrics["psnr"],
        }
        history.append(row)
        print(f"[Stage1] Epoch {epoch:03d} | lr={current_lr:.6g} | train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | val_psnr={val_metrics['psnr']:.3f}")
        payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_metrics["loss"], "config": namespace_to_config(args)}
        torch.save(payload, weights_dir / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(payload, weights_dir / "best.pt")
        save_history_csv(history, stage_dir / "history.csv")
        save_history_plots(history, stage_dir / "training_curves.png", [(["train_loss", "val_loss"], "Loss"), (["val_ssim", "val_psnr"], "Validation Metrics")])

    result = {"stage_dir": str(stage_dir), "best_ckpt": str(weights_dir / "best.pt"), "last_ckpt": str(weights_dir / "last.pt"), "split_json": str(context.split_json)}
    save_json(result, stage_dir / "stage_result.json")
    return result


if __name__ == "__main__":
    raise SystemExit("Please run train_3stage.py or import run_stage1().")
