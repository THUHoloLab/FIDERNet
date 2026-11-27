
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from descatternet_model_3stage import build_descatternet
from train_pipeline_common_3stage import (
    Stage2Dataset,
    Stage3Dataset,
    build_scheduler,
    evaluate_pair,
    evaluate_stage2_bridge,
    get_stage2_bridge_gamma,
    load_single_model_checkpoint,
    make_dataloader,
    namespace_to_config,
    parse_harmonics,
    prepare_context,
    save_history_csv,
    save_history_plots,
    save_json,
    set_seed,
    train_one_epoch_stage2_bridge,
    train_one_epoch_pair,
)


def run_stage2(args, master_run_dir: Optional[Path] = None):
    set_seed(args.seed)
    if master_run_dir is None:
        master_run_dir = Path(args.output_root) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    master_run_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = master_run_dir / "stage2_dehoney"
    weights_dir = stage_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    context = prepare_context(args, master_run_dir)
    stage1_ckpt = getattr(args, "stage1_ckpt", None)
    bridge_pred_prob = float(getattr(args, "stage2_bridge_pred_prob", 0.0))
    bridge_requested = bridge_pred_prob > 0.0
    bridge_enabled = bridge_requested and bool(stage1_ckpt) and Path(stage1_ckpt).is_file()
    if bridge_requested and not bridge_enabled:
        print("[Stage2] Warning: bridge mode requested but no valid Stage1 checkpoint is available; falling back to normal Stage2 training.")

    if bridge_enabled:
        train_dataset = Stage3Dataset(context.train_stage13, args.crop_size, "train", args.patches_per_image, args.hflip_prob, args.vflip_prob)
        val_dataset = Stage3Dataset(context.val_stage13, args.crop_size, "val", 1, 0.0, 0.0)
    else:
        train_dataset = Stage2Dataset(context.train_stage2, args.crop_size, "train", args.patches_per_image, args.hflip_prob, args.vflip_prob)
        val_dataset = Stage2Dataset(context.val_stage2, args.crop_size, "val", 1, 0.0, 0.0)
    train_loader = make_dataloader(train_dataset, args.batch_size, args.num_workers, True)
    val_loader = make_dataloader(val_dataset, args.batch_size, args.num_workers, False)

    harmonics = parse_harmonics(args.gate_harmonics)
    model = build_descatternet(
        base_channels=args.base_channels,
        dropout=args.dropout,
        batchnorm=True,
        use_spectral_gate=True,
        gate_a_px_input=args.gate_a_px_input,
        gate_bandwidth=args.gate_bandwidth,
        gate_harmonics=harmonics,
        gate_alpha_max=args.gate_alpha_max,
        gate_init_alpha=args.gate_init_alpha,
        gate_per_channel=(not args.gate_shared_alpha),
    ).to(device)
    descatter_model = None
    if bridge_enabled:
        descatter_model = build_descatternet(
            base_channels=args.base_channels,
            dropout=args.dropout,
            batchnorm=True,
            use_spectral_gate=False,
        ).to(device)
        load_single_model_checkpoint(descatter_model, stage1_ckpt, device)
        descatter_model.eval()
        for p in descatter_model.parameters():
            p.requires_grad = False

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr_stage2, betas=(args.beta1, 0.999))
    scheduler = build_scheduler(optimizer, args.min_lr, args.lr_patience)
    use_amp = torch.cuda.is_available() and (not args.no_amp)

    args.gate_harmonics = harmonics
    save_json(namespace_to_config(args), stage_dir / "config.json")
    history = []
    best_val_loss = float("inf")

    print(f"[Stage2] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    if bridge_enabled:
        print(
            f"[Stage2] Bridge mode enabled | stage1_ckpt={stage1_ckpt} "
            f"| pred_prob={bridge_pred_prob:.3f} | max_gamma={float(getattr(args, 'stage2_bridge_gamma', 1.0)):.3f}"
        )
    for epoch in range(1, args.epochs_stage2 + 1):
        if bridge_enabled:
            bridge_gamma = get_stage2_bridge_gamma(
                epoch,
                int(getattr(args, "stage2_bridge_start_epoch", 1)),
                int(getattr(args, "stage2_bridge_warmup_epochs", 10)),
                float(getattr(args, "stage2_bridge_gamma", 1.0)),
            )
            train_loss = train_one_epoch_stage2_bridge(
                model,
                descatter_model,
                train_loader,
                optimizer,
                criterion,
                device,
                use_amp,
                bridge_pred_prob,
                bridge_gamma,
            )
            val_metrics = evaluate_stage2_bridge(model, descatter_model, val_loader, criterion, device, bridge_gamma)
        else:
            bridge_gamma = 0.0
            train_loss = train_one_epoch_pair(model, train_loader, optimizer, criterion, device, use_amp)
            val_metrics = evaluate_pair(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        if bridge_enabled:
            row = {
                "epoch": epoch,
                "lr": current_lr,
                "bridge_gamma": bridge_gamma,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_oracle_loss": val_metrics["oracle_loss"],
                "val_oracle_psnr": val_metrics["oracle_psnr"],
                "val_oracle_ssim": val_metrics["oracle_ssim"],
                "val_pred_loss": val_metrics["pred_loss"],
                "val_pred_psnr": val_metrics["pred_psnr"],
                "val_pred_ssim": val_metrics["pred_ssim"],
                "val_mixed_loss": val_metrics["mixed_loss"],
                "val_mixed_psnr": val_metrics["mixed_psnr"],
                "val_mixed_ssim": val_metrics["mixed_ssim"],
            }
        else:
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
        if bridge_enabled:
            print(
                f"[Stage2] Epoch {epoch:03d} | lr={current_lr:.6g} | bridge_gamma={bridge_gamma:.3f} "
                f"| train_loss={train_loss:.6f} | val_pred_loss={val_metrics['pred_loss']:.6f} "
                f"| pred_psnr={val_metrics['pred_psnr']:.3f} | oracle_psnr={val_metrics['oracle_psnr']:.3f}"
            )
        else:
            print(f"[Stage2] Epoch {epoch:03d} | lr={current_lr:.6g} | train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} | val_psnr={val_metrics['psnr']:.3f}")
        payload = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_metrics["loss"], "config": namespace_to_config(args)}
        torch.save(payload, weights_dir / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(payload, weights_dir / "best.pt")
        save_history_csv(history, stage_dir / "history.csv")
        if bridge_enabled:
            save_history_plots(
                history,
                stage_dir / "training_curves.png",
                [
                    (["train_loss", "val_loss", "val_oracle_loss", "val_pred_loss"], "Loss"),
                    (["val_oracle_psnr", "val_pred_psnr", "val_mixed_psnr"], "Validation PSNR"),
                    (["val_oracle_ssim", "val_pred_ssim", "val_mixed_ssim"], "Validation SSIM"),
                ],
            )
        else:
            save_history_plots(history, stage_dir / "training_curves.png", [(["train_loss", "val_loss"], "Loss"), (["val_ssim", "val_psnr"], "Validation Metrics")])

    result = {"stage_dir": str(stage_dir), "best_ckpt": str(weights_dir / "best.pt"), "last_ckpt": str(weights_dir / "last.pt"), "split_json": str(context.split_json)}
    save_json(result, stage_dir / "stage_result.json")
    return result
