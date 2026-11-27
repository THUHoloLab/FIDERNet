
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from descatternet_model_3stage import build_descatternet
from train_pipeline_common_3stage import (
    Stage3Dataset,
    build_scheduler,
    evaluate_joint,
    make_dataloader,
    maybe_load_joint_checkpoints,
    namespace_to_config,
    parse_harmonics,
    prepare_context,
    save_history_csv,
    save_history_plots,
    save_json,
    set_seed,
    train_one_epoch_joint,
)


def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def run_stage3(args, master_run_dir: Optional[Path] = None, stage1_ckpt: Optional[str] = None, stage2_ckpt: Optional[str] = None):
    set_seed(args.seed)
    if master_run_dir is None:
        master_run_dir = Path(args.output_root) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    master_run_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = master_run_dir / "stage3_joint"
    weights_dir = stage_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    context = prepare_context(args, master_run_dir)
    train_dataset = Stage3Dataset(context.train_stage13, args.crop_size, "train", args.patches_per_image, args.hflip_prob, args.vflip_prob)
    val_dataset = Stage3Dataset(context.val_stage13, args.crop_size, "val", 1, 0.0, 0.0)
    train_loader = make_dataloader(train_dataset, args.batch_size, args.num_workers, True)
    val_loader = make_dataloader(val_dataset, args.batch_size, args.num_workers, False)

    device = torch.device(args.device)
    harmonics = parse_harmonics(args.gate_harmonics)
    descatter_model = build_descatternet(base_channels=args.base_channels, dropout=args.dropout, batchnorm=True, use_spectral_gate=False).to(device)
    dehoney_model = build_descatternet(
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
    maybe_load_joint_checkpoints(descatter_model, dehoney_model, stage1_ckpt, stage2_ckpt, device)

    criterion_mid = nn.L1Loss()
    criterion_final = nn.L1Loss()
    optimizer = Adam(
        [
            {
                "params": descatter_model.parameters(),
                "lr": args.lr_stage3 * float(getattr(args, "stage3_descatter_lr_scale", 0.1)),
                "name": "descatter",
            },
            {
                "params": dehoney_model.parameters(),
                "lr": args.lr_stage3,
                "name": "dehoney",
            },
        ],
        betas=(args.beta1, 0.999),
    )
    scheduler = build_scheduler(optimizer, args.min_lr, args.lr_patience)
    use_amp = torch.cuda.is_available() and (not args.no_amp)

    args.gate_harmonics = harmonics
    save_json(namespace_to_config(args), stage_dir / "config.json")
    history = []
    best_val_loss = float("inf")

    print(f"[Stage3] Train samples: {len(context.train_stage13)} | Val samples: {len(context.val_stage13)}")
    for epoch in range(1, args.epochs_stage3 + 1):
        freeze_stage1 = epoch <= int(getattr(args, "stage3_freeze_stage1_epochs", 0))
        set_requires_grad(descatter_model, not freeze_stage1)
        alpha_mid_eff = 0.0 if freeze_stage1 else float(getattr(args, "stage3_alpha_mid_after_unfreeze", args.alpha_mid))
        beta_final_eff = float(args.beta_final)
        train_metrics = train_one_epoch_joint(
            descatter_model,
            dehoney_model,
            train_loader,
            optimizer,
            criterion_mid,
            criterion_final,
            device,
            use_amp,
            alpha_mid_eff,
            beta_final_eff,
            freeze_descatter=freeze_stage1,
        )
        val_metrics = evaluate_joint(descatter_model, dehoney_model, val_loader, criterion_mid, criterion_final, device, alpha_mid_eff, beta_final_eff)
        scheduler.step(val_metrics["loss"])
        lr_descatter = optimizer.param_groups[0]["lr"]
        lr_dehoney = optimizer.param_groups[1]["lr"]
        row = {
            "epoch": epoch,
            "freeze_stage1": int(freeze_stage1),
            "alpha_mid_eff": alpha_mid_eff,
            "beta_final_eff": beta_final_eff,
            "lr_descatter": lr_descatter,
            "lr_dehoney": lr_dehoney,
            "train_loss": train_metrics["loss"],
            "train_mid_loss": train_metrics["mid_loss"],
            "train_final_loss": train_metrics["final_loss"],
            "val_loss": val_metrics["loss"],
            "val_mid_loss": val_metrics["mid_loss"],
            "val_final_loss": val_metrics["final_loss"],
            "val_mid_ssim": val_metrics["mid_ssim"],
            "val_mid_psnr": val_metrics["mid_psnr"],
            "val_final_ssim": val_metrics["final_ssim"],
            "val_final_psnr": val_metrics["final_psnr"],
        }
        history.append(row)
        print(
            f"[Stage3] Epoch {epoch:03d} | freeze_stage1={freeze_stage1} "
            f"| lr_descatter={lr_descatter:.6g} | lr_dehoney={lr_dehoney:.6g} "
            f"| alpha_mid={alpha_mid_eff:.3f} | train_loss={train_metrics['loss']:.6f} "
            f"| val_loss={val_metrics['loss']:.6f} | mid_psnr={val_metrics['mid_psnr']:.3f} "
            f"| final_psnr={val_metrics['final_psnr']:.3f}"
        )
        payload = {
            "epoch": epoch,
            "descatter_state_dict": descatter_model.state_dict(),
            "dehoney_state_dict": dehoney_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "config": namespace_to_config(args),
        }
        torch.save(payload, weights_dir / "last.pt")
        torch.save({"epoch": epoch, "model_state_dict": descatter_model.state_dict(), "config": namespace_to_config(args)}, weights_dir / "last_descatter.pt")
        torch.save({"epoch": epoch, "model_state_dict": dehoney_model.state_dict(), "config": namespace_to_config(args)}, weights_dir / "last_dehoney.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(payload, weights_dir / "best.pt")
            torch.save({"epoch": epoch, "model_state_dict": descatter_model.state_dict(), "config": namespace_to_config(args)}, weights_dir / "best_descatter.pt")
            torch.save({"epoch": epoch, "model_state_dict": dehoney_model.state_dict(), "config": namespace_to_config(args)}, weights_dir / "best_dehoney.pt")
        save_history_csv(history, stage_dir / "history.csv")
        save_history_plots(
            history,
            stage_dir / "training_curves.png",
            [
                (["train_loss", "val_loss", "val_mid_loss", "val_final_loss"], "Joint Loss"),
                (["val_mid_psnr", "val_final_psnr", "val_mid_ssim", "val_final_ssim"], "Validation Metrics"),
                (["lr_descatter", "lr_dehoney"], "Learning Rates"),
            ],
        )

    result = {
        "stage_dir": str(stage_dir),
        "best_joint_ckpt": str(weights_dir / "best.pt"),
        "best_descatter_ckpt": str(weights_dir / "best_descatter.pt"),
        "best_dehoney_ckpt": str(weights_dir / "best_dehoney.pt"),
        "last_joint_ckpt": str(weights_dir / "last.pt"),
        "split_json": str(context.split_json),
    }
    save_json(result, stage_dir / "stage_result.json")
    return result
