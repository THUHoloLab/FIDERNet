from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Tuple

VALID_EXTS = {".png"}


DEFAULT_SEEN = ("fat_0.05", "fat_0.1", "fat_0.25", "fat_0.40")
DEFAULT_UNSEEN = ("fat_0.35", "fat_0.20")


def parse_mapping(text: str) -> Tuple[str, str]:
    """
    Accept either "output_name=source_name" or a plain condition name.
    Plain names are used as both output and source names, with numeric aliases
    resolved later, e.g. fat_0.20 can resolve to source fat_0.2.
    """
    if "=" not in text:
        return text, text
    out_name, src_name = text.split("=", 1)
    out_name = out_name.strip()
    src_name = src_name.strip()
    if not out_name or not src_name:
        raise ValueError(f"Invalid condition mapping: {text}")
    return out_name, src_name


def canonical_condition_name(name: str) -> str:
    if not name.startswith("fat_"):
        return name
    value = name[len("fat_") :]
    try:
        return f"fat_{float(value):g}"
    except ValueError:
        return name


def resolve_source_condition(source_root: Path, requested: str) -> Path:
    exact = source_root / requested
    if exact.is_dir():
        return exact

    canonical = canonical_condition_name(requested)
    canonical_path = source_root / canonical
    if canonical_path.is_dir():
        return canonical_path

    available = {canonical_condition_name(p.name): p for p in source_root.iterdir() if p.is_dir()}
    if canonical in available:
        return available[canonical]

    names = ", ".join(sorted(p.name for p in source_root.iterdir() if p.is_dir()))
    raise FileNotFoundError(f"Condition '{requested}' was not found under {source_root}. Available: {names}")


def link_or_copy_file(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def mirror_condition(source_dir: Path, output_dir: Path, mode: str) -> Tuple[int, int, int]:
    linked = copied = existed = 0
    images = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
    if not images:
        raise FileNotFoundError(f"No PNG images found in {source_dir}")

    for src in images:
        result = link_or_copy_file(src, output_dir / src.name, mode)
        if result == "linked":
            linked += 1
        elif result == "copied":
            copied += 1
        else:
            existed += 1
    return linked, copied, existed


def build_group(
    source_root: Path,
    output_root: Path,
    group_name: str,
    conditions: Iterable[str],
    mode: str,
) -> None:
    group_dir = output_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Group] {group_name}: {group_dir}")

    for item in conditions:
        output_name, source_name = parse_mapping(item)
        source_dir = resolve_source_condition(source_root, source_name)
        target_dir = group_dir / output_name
        linked, copied, existed = mirror_condition(source_dir, target_dir, mode)
        print(
            f"  {output_name} <- {source_dir.name}: "
            f"linked={linked}, copied={copied}, existed={existed}, total={linked + copied + existed}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a seen/unseen scattering-condition split folder for FIDER-Net experiments."
    )
    parser.add_argument("--source_scatter_root", default="datasets/train_scatter", type=str)
    parser.add_argument("--output_root", default="dataset_unseen_split", type=str)
    parser.add_argument("--mode", default="hardlink", choices=["hardlink", "copy"], help="Use hardlinks by default; fallback to copy if hardlink fails.")
    parser.add_argument("--seen", nargs="+", default=list(DEFAULT_SEEN))
    parser.add_argument("--unseen", nargs="+", default=list(DEFAULT_UNSEEN))
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_scatter_root)
    output_root = Path(args.output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"source_scatter_root does not exist: {source_root}")

    mode = "copy" if args.mode == "copy" else "hardlink"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Source scatter root: {source_root}")
    print(f"[Info] Output root: {output_root}")
    print(f"[Info] Mode: {mode}")
    build_group(source_root, output_root, "train_scatter_seen", args.seen, mode)
    build_group(source_root, output_root, "test_scatter_unseen", args.unseen, mode)
    print("[Done] Unseen scattering split is ready.")


if __name__ == "__main__":
    main()
