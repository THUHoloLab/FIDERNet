
from __future__ import annotations

import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from glcme_preprocess_3stage import (
    center_paired_crop,
    center_triplet_crop,
    preprocess_and_cache_glcme_image,
    random_paired_crop,
    random_triplet_crop,
    read_gray_image,
)

VALID_EXTS = {".png"}


@dataclass(frozen=True)
class Stage13Sample:
    condition: str
    basename: str
    scatter_path: str
    fiber_path: str
    gt_path: str
    scatter_cache_path: str
    fiber_cache_path: str


@dataclass(frozen=True)
class Stage2Sample:
    basename: str
    fiber_path: str
    gt_path: str
    fiber_cache_path: str


@dataclass
class TrainContext:
    run_dir: Path
    cache_root: Path
    condition_dirs: List[Path]
    train_names: List[str]
    val_names: List[str]
    test_names: List[str]
    train_stage13: List[Stage13Sample]
    val_stage13: List[Stage13Sample]
    test_stage13: List[Stage13Sample]
    train_stage2: List[Stage2Sample]
    val_stage2: List[Stage2Sample]
    test_stage2: List[Stage2Sample]
    split_json: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_harmonics(harmonics_str):
    if isinstance(harmonics_str, (tuple, list)):
        if len(harmonics_str) == 0:
            raise ValueError("gate_harmonics cannot be empty.")
        return tuple(int(v) for v in harmonics_str)

    text = str(harmonics_str).strip()
    text = text.strip("()[]")
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("gate_harmonics cannot be empty.")
    return tuple(int(v) for v in values)


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


def _sort_name(x: str):
    return int(x) if x.isdigit() else x


def build_split_basenames(all_basenames: Sequence[str], val_ratio: float, test_ratio: float, seed: int):
    names = list(all_basenames)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_total = len(names)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("The split ratios leave no training data.")
    train_names = sorted(names[:n_train], key=_sort_name)
    val_names = sorted(names[n_train:n_train + n_val], key=_sort_name)
    test_names = sorted(names[n_train + n_val:], key=_sort_name)
    return train_names, val_names, test_names


def prepare_split_json(run_dir: Path, shared_basenames, condition_names, val_ratio, test_ratio, seed):
    split_json = run_dir / "split_basenames.json"
    if split_json.exists():
        data = json.loads(split_json.read_text(encoding="utf-8"))
        return list(data["train"]), list(data["val"]), list(data["test"]), split_json
    train_names, val_names, test_names = build_split_basenames(shared_basenames, val_ratio, test_ratio, seed)
    split_json.write_text(json.dumps({
        "train": train_names,
        "val": val_names,
        "test": test_names,
        "conditions": list(condition_names),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return train_names, val_names, test_names, split_json


def build_stage13_samples(basenames, condition_dirs, fiber_root, gt_root, cache_root):
    fiber_map = list_image_files(fiber_root)
    gt_map = list_image_files(gt_root)
    basename_set = set(basenames)
    samples = []
    for cond_dir in condition_dirs:
        condition = cond_dir.name
        scatter_map = list_image_files(cond_dir)
        common = sorted(basename_set.intersection(scatter_map.keys()).intersection(fiber_map.keys()).intersection(gt_map.keys()), key=_sort_name)
        for name in common:
            samples.append(Stage13Sample(
                condition=condition,
                basename=name,
                scatter_path=str(scatter_map[name]),
                fiber_path=str(fiber_map[name]),
                gt_path=str(gt_map[name]),
                scatter_cache_path=str(cache_root / "scatter_glcme" / condition / f"{name}.png"),
                fiber_cache_path=str(cache_root / "fiber_glcme" / f"{name}.png"),
            ))
    return samples


def build_stage2_samples(basenames, fiber_root, gt_root, cache_root):
    fiber_map = list_image_files(fiber_root)
    gt_map = list_image_files(gt_root)
    common = sorted(set(basenames).intersection(fiber_map.keys()).intersection(gt_map.keys()), key=_sort_name)
    return [Stage2Sample(
        basename=name,
        fiber_path=str(fiber_map[name]),
        gt_path=str(gt_map[name]),
        fiber_cache_path=str(cache_root / "fiber_glcme" / f"{name}.png"),
    ) for name in common]


def _cache_worker(args):
    src_path, dst_path, tiles_y, tiles_x, clip_limit_ratio = args
    preprocess_and_cache_glcme_image(src_path, dst_path, num_tiles=(tiles_y, tiles_x), clip_limit_ratio=clip_limit_ratio)
    return dst_path


def build_glcme_caches(stage13_samples, stage2_samples, num_tiles, clip_limit_ratio, num_workers):
    unique_jobs = {}
    for s in stage13_samples:
        unique_jobs[s.scatter_path] = (s.scatter_path, s.scatter_cache_path)
        unique_jobs[s.fiber_path] = (s.fiber_path, s.fiber_cache_path)
    for s in stage2_samples:
        unique_jobs[s.fiber_path] = (s.fiber_path, s.fiber_cache_path)

    jobs = []
    for src, dst in unique_jobs.values():
        dst_path = Path(dst)
        if dst_path.exists():
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((src, dst, num_tiles[0], num_tiles[1], clip_limit_ratio))

    if not jobs:
        print("[Cache] All GLCME cache images already exist.")
        return

    print(f"[Cache] Building {len(jobs)} GLCME cache images ...")
    with ProcessPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        futures = [ex.submit(_cache_worker, job) for job in jobs]
        for idx, fut in enumerate(as_completed(futures), start=1):
            fut.result()
            if idx % 50 == 0 or idx == len(futures):
                print(f"[Cache] {idx}/{len(futures)} done")


def prepare_context(args, run_dir: Path) -> TrainContext:
    scattering_root = Path(args.scattering_root)
    fiber_root = Path(args.fiber_root)
    clear_root = Path(args.clear_root)
    cache_root = Path(args.cache_root) if getattr(args, "cache_root", None) else scattering_root.parent / "glcme_cache_3stage"
    cache_root.mkdir(parents=True, exist_ok=True)

    condition_dirs = discover_condition_dirs(scattering_root)
    fiber_map = list_image_files(fiber_root)
    gt_map = list_image_files(clear_root)
    if not fiber_map:
        raise FileNotFoundError(f"No fiber images found in {fiber_root}")
    if not gt_map:
        raise FileNotFoundError(f"No GT images found in {clear_root}")

    shared_basenames = set(fiber_map.keys()).intersection(gt_map.keys())
    for cond in condition_dirs:
        shared_basenames &= set(list_image_files(cond).keys())
    shared_basenames = sorted(shared_basenames, key=_sort_name)
    if not shared_basenames:
        raise RuntimeError("No common basenames were found across scatter/fiber/GT data.")

    train_names, val_names, test_names, split_json = prepare_split_json(
        run_dir, shared_basenames, [p.name for p in condition_dirs], args.val_ratio, args.test_ratio, args.seed
    )

    train_stage13 = build_stage13_samples(train_names, condition_dirs, fiber_root, clear_root, cache_root)
    val_stage13 = build_stage13_samples(val_names, condition_dirs, fiber_root, clear_root, cache_root)
    test_stage13 = build_stage13_samples(test_names, condition_dirs, fiber_root, clear_root, cache_root)

    train_stage2 = build_stage2_samples(train_names, fiber_root, clear_root, cache_root)
    val_stage2 = build_stage2_samples(val_names, fiber_root, clear_root, cache_root)
    test_stage2 = build_stage2_samples(test_names, fiber_root, clear_root, cache_root)

    build_glcme_caches(
        stage13_samples=train_stage13 + val_stage13 + test_stage13,
        stage2_samples=train_stage2 + val_stage2 + test_stage2,
        num_tiles=(args.tiles_y, args.tiles_x),
        clip_limit_ratio=args.clip_limit_ratio,
        num_workers=args.cache_workers,
    )

    return TrainContext(
        run_dir=run_dir,
        cache_root=cache_root,
        condition_dirs=condition_dirs,
        train_names=train_names,
        val_names=val_names,
        test_names=test_names,
        train_stage13=train_stage13,
        val_stage13=val_stage13,
        test_stage13=test_stage13,
        train_stage2=train_stage2,
        val_stage2=val_stage2,
        test_stage2=test_stage2,
        split_json=split_json,
    )


def _apply_flips(images, rng, hflip_prob, vflip_prob):
    if hflip_prob > 0.0 and rng.random() < hflip_prob:
        images = [np.fliplr(img).copy() for img in images]
    if vflip_prob > 0.0 and rng.random() < vflip_prob:
        images = [np.flipud(img).copy() for img in images]
    return images


class Stage1Dataset(Dataset):
    def __init__(self, samples, crop_size, split, patches_per_image=1, hflip_prob=0.0, vflip_prob=0.0):
        self.samples = list(samples)
        self.crop_size = int(crop_size)
        self.split = split
        self.patches_per_image = max(1, int(patches_per_image)) if split == "train" else 1
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        scatter_img = read_gray_image(sample.scatter_cache_path)
        fiber_img = read_gray_image(sample.fiber_cache_path)
        if self.split == "train":
            rng = np.random.default_rng(None)
            scatter_img, fiber_img = random_paired_crop(scatter_img, fiber_img, self.crop_size, rng=rng)
            scatter_img, fiber_img = _apply_flips([scatter_img, fiber_img], rng, self.hflip_prob, self.vflip_prob)
        else:
            scatter_img, fiber_img = center_paired_crop(scatter_img, fiber_img, self.crop_size)
        x = torch.from_numpy(scatter_img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy(fiber_img.astype(np.float32) / 255.0).unsqueeze(0)
        return x, y, sample.condition, sample.basename


class Stage2Dataset(Dataset):
    def __init__(self, samples, crop_size, split, patches_per_image=1, hflip_prob=0.0, vflip_prob=0.0):
        self.samples = list(samples)
        self.crop_size = int(crop_size)
        self.split = split
        self.patches_per_image = max(1, int(patches_per_image)) if split == "train" else 1
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        fiber_img = read_gray_image(sample.fiber_cache_path)
        gt_img = read_gray_image(sample.gt_path)
        if self.split == "train":
            rng = np.random.default_rng(None)
            fiber_img, gt_img = random_paired_crop(fiber_img, gt_img, self.crop_size, rng=rng)
            fiber_img, gt_img = _apply_flips([fiber_img, gt_img], rng, self.hflip_prob, self.vflip_prob)
        else:
            fiber_img, gt_img = center_paired_crop(fiber_img, gt_img, self.crop_size)
        x = torch.from_numpy(fiber_img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy(gt_img.astype(np.float32) / 255.0).unsqueeze(0)
        return x, y, "fiber_only", sample.basename


class Stage3Dataset(Dataset):
    def __init__(self, samples, crop_size, split, patches_per_image=1, hflip_prob=0.0, vflip_prob=0.0):
        self.samples = list(samples)
        self.crop_size = int(crop_size)
        self.split = split
        self.patches_per_image = max(1, int(patches_per_image)) if split == "train" else 1
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        scatter_img = read_gray_image(sample.scatter_cache_path)
        fiber_img = read_gray_image(sample.fiber_cache_path)
        gt_img = read_gray_image(sample.gt_path)
        if self.split == "train":
            rng = np.random.default_rng(None)
            scatter_img, fiber_img, gt_img = random_triplet_crop(scatter_img, fiber_img, gt_img, self.crop_size, rng=rng)
            scatter_img, fiber_img, gt_img = _apply_flips([scatter_img, fiber_img, gt_img], rng, self.hflip_prob, self.vflip_prob)
        else:
            scatter_img, fiber_img, gt_img = center_triplet_crop(scatter_img, fiber_img, gt_img, self.crop_size)
        x = torch.from_numpy(scatter_img.astype(np.float32) / 255.0).unsqueeze(0)
        y_mid = torch.from_numpy(fiber_img.astype(np.float32) / 255.0).unsqueeze(0)
        y_final = torch.from_numpy(gt_img.astype(np.float32) / 255.0).unsqueeze(0)
        return x, y_mid, y_final, sample.condition, sample.basename


def make_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    mse_vals, ssim_vals, psnr_vals = [], [], []
    for i in range(pred_np.shape[0]):
        p = np.clip(pred_np[i, 0], 0.0, 1.0)
        t = np.clip(target_np[i, 0], 0.0, 1.0)
        mse_vals.append(float(np.mean((p - t) ** 2)))
        psnr_vals.append(float(peak_signal_noise_ratio(t, p, data_range=1.0)))
        ssim_vals.append(float(structural_similarity(t, p, data_range=1.0)))
    return float(np.mean(mse_vals)), float(np.mean(ssim_vals)), float(np.mean(psnr_vals))


def train_one_epoch_pair(model, loader, optimizer, criterion, device, use_amp):
    model.train()
    total_loss = 0.0
    total_count = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for inputs, targets, _, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = inputs.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
    return total_loss / max(1, total_count)


def evaluate_pair(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    mse_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for inputs, targets, _, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            bs = inputs.size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs
            mse_val, ssim_val, psnr_val = compute_batch_metrics(outputs, targets)
            mse_list.append(mse_val)
            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)
    return {
        "loss": total_loss / max(1, total_count),
        "mse": float(np.mean(mse_list)) if mse_list else float("nan"),
        "ssim": float(np.mean(ssim_list)) if ssim_list else float("nan"),
        "psnr": float(np.mean(psnr_list)) if psnr_list else float("nan"),
    }


def get_stage2_bridge_gamma(epoch, start_epoch, warmup_epochs, max_gamma):
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return float(max_gamma)
    progress = (epoch - start_epoch + 1) / float(warmup_epochs)
    return float(max_gamma) * min(1.0, max(0.0, progress))


def train_one_epoch_stage2_bridge(
    dehoney_model,
    descatter_model,
    loader,
    optimizer,
    criterion,
    device,
    use_amp,
    pred_prob,
    bridge_gamma,
):
    descatter_model.eval()
    dehoney_model.train()
    total_loss = 0.0
    total_count = 0
    pred_prob = min(1.0, max(0.0, float(pred_prob)))
    bridge_gamma = float(bridge_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for scatter_inputs, real_fiber, clear_targets, _, _ in loader:
        scatter_inputs = scatter_inputs.to(device, non_blocking=True)
        real_fiber = real_fiber.to(device, non_blocking=True)
        clear_targets = clear_targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                pred_fiber = descatter_model(scatter_inputs).detach()
            mixed_pred_input = (1.0 - bridge_gamma) * real_fiber + bridge_gamma * pred_fiber
            if pred_prob >= 1.0:
                dehoney_input = mixed_pred_input
            elif pred_prob <= 0.0:
                dehoney_input = real_fiber
            else:
                use_pred = torch.rand((real_fiber.size(0), 1, 1, 1), device=device) < pred_prob
                dehoney_input = torch.where(use_pred, mixed_pred_input, real_fiber)
            pred_clear = dehoney_model(dehoney_input)
            loss = criterion(pred_clear, clear_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = scatter_inputs.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs
    return total_loss / max(1, total_count)


def evaluate_stage2_bridge(dehoney_model, descatter_model, loader, criterion, device, bridge_gamma):
    descatter_model.eval()
    dehoney_model.eval()
    bridge_gamma = float(bridge_gamma)
    totals = {
        "oracle": {"loss": 0.0, "mse": [], "ssim": [], "psnr": []},
        "pred": {"loss": 0.0, "mse": [], "ssim": [], "psnr": []},
        "mixed": {"loss": 0.0, "mse": [], "ssim": [], "psnr": []},
    }
    total_count = 0

    def update(prefix, outputs, targets, loss, bs):
        totals[prefix]["loss"] += float(loss.item()) * bs
        mse_val, ssim_val, psnr_val = compute_batch_metrics(outputs, targets)
        totals[prefix]["mse"].append(mse_val)
        totals[prefix]["ssim"].append(ssim_val)
        totals[prefix]["psnr"].append(psnr_val)

    with torch.no_grad():
        for scatter_inputs, real_fiber, clear_targets, _, _ in loader:
            scatter_inputs = scatter_inputs.to(device, non_blocking=True)
            real_fiber = real_fiber.to(device, non_blocking=True)
            clear_targets = clear_targets.to(device, non_blocking=True)
            pred_fiber = descatter_model(scatter_inputs).detach()
            mixed_input = (1.0 - bridge_gamma) * real_fiber + bridge_gamma * pred_fiber

            oracle_outputs = dehoney_model(real_fiber)
            pred_outputs = dehoney_model(pred_fiber)
            mixed_outputs = dehoney_model(mixed_input)

            oracle_loss = criterion(oracle_outputs, clear_targets)
            pred_loss = criterion(pred_outputs, clear_targets)
            mixed_loss = criterion(mixed_outputs, clear_targets)

            bs = scatter_inputs.size(0)
            total_count += bs
            update("oracle", oracle_outputs, clear_targets, oracle_loss, bs)
            update("pred", pred_outputs, clear_targets, pred_loss, bs)
            update("mixed", mixed_outputs, clear_targets, mixed_loss, bs)

    denom = max(1, total_count)
    metrics = {"bridge_gamma": bridge_gamma}
    for prefix, values in totals.items():
        metrics[f"{prefix}_loss"] = values["loss"] / denom
        metrics[f"{prefix}_mse"] = float(np.mean(values["mse"])) if values["mse"] else float("nan")
        metrics[f"{prefix}_ssim"] = float(np.mean(values["ssim"])) if values["ssim"] else float("nan")
        metrics[f"{prefix}_psnr"] = float(np.mean(values["psnr"])) if values["psnr"] else float("nan")
    metrics["loss"] = metrics["pred_loss"]
    return metrics


def train_one_epoch_joint(
    descatter_model,
    dehoney_model,
    loader,
    optimizer,
    criterion_mid,
    criterion_final,
    device,
    use_amp,
    alpha_mid,
    beta_final,
    freeze_descatter=False,
):
    if freeze_descatter:
        descatter_model.eval()
    else:
        descatter_model.train()
    dehoney_model.train()
    total_loss = total_mid_loss = total_final_loss = 0.0
    total_count = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for inputs, mid_targets, final_targets, _, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        mid_targets = mid_targets.to(device, non_blocking=True)
        final_targets = final_targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if freeze_descatter:
                with torch.no_grad():
                    mid_outputs = descatter_model(inputs).detach()
            else:
                mid_outputs = descatter_model(inputs)
            final_outputs = dehoney_model(mid_outputs)
            mid_loss = criterion_mid(mid_outputs, mid_targets)
            final_loss = criterion_final(final_outputs, final_targets)
            if freeze_descatter:
                loss = beta_final * final_loss
            else:
                loss = alpha_mid * mid_loss + beta_final * final_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = inputs.size(0)
        total_loss += float(loss.item()) * bs
        total_mid_loss += float(mid_loss.item()) * bs
        total_final_loss += float(final_loss.item()) * bs
        total_count += bs
    denom = max(1, total_count)
    return {"loss": total_loss / denom, "mid_loss": total_mid_loss / denom, "final_loss": total_final_loss / denom}


def evaluate_joint(descatter_model, dehoney_model, loader, criterion_mid, criterion_final, device, alpha_mid, beta_final):
    descatter_model.eval()
    dehoney_model.eval()
    total_loss = total_mid_loss = total_final_loss = 0.0
    total_count = 0
    mid_mse_list, mid_ssim_list, mid_psnr_list = [], [], []
    final_mse_list, final_ssim_list, final_psnr_list = [], [], []
    with torch.no_grad():
        for inputs, mid_targets, final_targets, _, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            mid_targets = mid_targets.to(device, non_blocking=True)
            final_targets = final_targets.to(device, non_blocking=True)
            mid_outputs = descatter_model(inputs)
            final_outputs = dehoney_model(mid_outputs)
            mid_loss = criterion_mid(mid_outputs, mid_targets)
            final_loss = criterion_final(final_outputs, final_targets)
            loss = alpha_mid * mid_loss + beta_final * final_loss
            bs = inputs.size(0)
            total_loss += float(loss.item()) * bs
            total_mid_loss += float(mid_loss.item()) * bs
            total_final_loss += float(final_loss.item()) * bs
            total_count += bs
            mmse, mssim, mpsnr = compute_batch_metrics(mid_outputs, mid_targets)
            fmse, fssim, fpsnr = compute_batch_metrics(final_outputs, final_targets)
            mid_mse_list.append(mmse)
            mid_ssim_list.append(mssim)
            mid_psnr_list.append(mpsnr)
            final_mse_list.append(fmse)
            final_ssim_list.append(fssim)
            final_psnr_list.append(fpsnr)
    denom = max(1, total_count)
    return {
        "loss": total_loss / denom,
        "mid_loss": total_mid_loss / denom,
        "final_loss": total_final_loss / denom,
        "mid_mse": float(np.mean(mid_mse_list)) if mid_mse_list else float("nan"),
        "mid_ssim": float(np.mean(mid_ssim_list)) if mid_ssim_list else float("nan"),
        "mid_psnr": float(np.mean(mid_psnr_list)) if mid_psnr_list else float("nan"),
        "final_mse": float(np.mean(final_mse_list)) if final_mse_list else float("nan"),
        "final_ssim": float(np.mean(final_ssim_list)) if final_ssim_list else float("nan"),
        "final_psnr": float(np.mean(final_psnr_list)) if final_psnr_list else float("nan"),
    }


def save_history_csv(history, path: Path):
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_history_plots(history, output_png: Path, metric_groups):
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig = plt.figure(figsize=(6 * len(metric_groups), 5))
    for idx, (keys, title) in enumerate(metric_groups, start=1):
        ax = fig.add_subplot(1, len(metric_groups), idx)
        for key in keys:
            if key in history[0]:
                ax.plot(epochs, [row[key] for row in history], label=key)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_scheduler(optimizer, min_lr, patience):
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=int(patience), min_lr=float(min_lr))


def maybe_load_checkpoint(model, checkpoint_path: Optional[str], device):
    if checkpoint_path is None:
        return 0
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    return int(ckpt.get("epoch", 0))


def load_single_model_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "descatter_state_dict" in ckpt:
        state = ckpt["descatter_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    return model


def maybe_load_joint_checkpoints(descatter_model, dehoney_model, stage1_ckpt, stage2_ckpt, device):
    if stage1_ckpt:
        maybe_load_checkpoint(descatter_model, stage1_ckpt, device)
    if stage2_ckpt:
        maybe_load_checkpoint(dehoney_model, stage2_ckpt, device)


def namespace_to_config(args):
    cfg = vars(args).copy()
    if "gate_harmonics" in cfg and isinstance(cfg["gate_harmonics"], tuple):
        cfg["gate_harmonics"] = list(cfg["gate_harmonics"])
    return cfg
