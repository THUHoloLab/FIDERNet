
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix

PathLike = Union[str, Path]


def read_gray_image(path: PathLike) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_gray_image(img: np.ndarray, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)


def _clip_glcm(g_2d: np.ndarray, clip_limit_ratio: float) -> np.ndarray:
    total = float(g_2d.sum())
    if total <= 0:
        return g_2d
    limit_val = total * clip_limit_ratio
    g_2d = g_2d.astype(np.float64, copy=True)
    mask = g_2d > limit_val
    excess = float((g_2d[mask] - limit_val).sum())
    g_2d[mask] = limit_val
    g_2d += excess / g_2d.size
    return g_2d


def local_glcme_equalize(
    img: np.ndarray,
    num_tiles: Tuple[int, int] = (8, 8),
    clip_limit_ratio: float = 0.002,
    levels: int = 256,
) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("local_glcme_equalize expects a 2D grayscale image.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape
    ty, tx = num_tiles
    h_step = h / ty
    w_step = w / tx

    mappings = np.zeros((levels, ty, tx), dtype=np.float32)
    angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]

    for i in range(ty):
        for j in range(tx):
            r1 = max(0, int(round(i * h_step)))
            r2 = min(h, int(round((i + 1) * h_step)))
            c1 = max(0, int(round(j * w_step)))
            c2 = min(w, int(round((j + 1) * w_step)))
            tile = img[r1:r2, c1:c2]
            if tile.size == 0:
                mappings[:, i, j] = np.linspace(0.0, 1.0, levels, dtype=np.float32)
                continue

            glcm = graycomatrix(
                tile,
                distances=[1],
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=False,
            )
            g_2d = glcm[:, :, 0, :].sum(axis=2) / float(len(angles))
            g_2d = _clip_glcm(g_2d, clip_limit_ratio)

            pdf = g_2d.sum(axis=1).astype(np.float32)
            pdf_sum = float(pdf.sum())
            if pdf_sum <= 0:
                mappings[:, i, j] = np.linspace(0.0, 1.0, levels, dtype=np.float32)
            else:
                pdf /= pdf_sum
                mappings[:, i, j] = np.cumsum(pdf, axis=0)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cx = ((xx + 0.5) - (w_step / 2.0)) / w_step
    cy = ((yy + 0.5) - (h_step / 2.0)) / h_step
    cx = np.clip(cx, 0.0, tx - 1.0)
    cy = np.clip(cy, 0.0, ty - 1.0)

    x1 = np.floor(cx).astype(np.int32)
    x2 = np.minimum(x1 + 1, tx - 1)
    y1 = np.floor(cy).astype(np.int32)
    y2 = np.minimum(y1 + 1, ty - 1)

    wx2 = cx - x1
    wx1 = 1.0 - wx2
    wy2 = cy - y1
    wy1 = 1.0 - wy2

    idx = img.astype(np.int32)
    m11 = mappings[idx, y1, x1]
    m12 = mappings[idx, y1, x2]
    m21 = mappings[idx, y2, x1]
    m22 = mappings[idx, y2, x2]

    out_float = wy1 * (wx1 * m11 + wx2 * m12) + wy2 * (wx1 * m21 + wx2 * m22)
    return np.clip(np.round(out_float * 255.0), 0, 255).astype(np.uint8)


def preprocess_glcme_image(
    image_or_path: Union[PathLike, np.ndarray],
    num_tiles: Tuple[int, int] = (8, 8),
    clip_limit_ratio: float = 0.002,
    return_uint8: bool = False,
) -> np.ndarray:
    if isinstance(image_or_path, (str, Path)):
        img = read_gray_image(image_or_path)
    else:
        img = np.asarray(image_or_path)
        if img.ndim != 2:
            raise ValueError("preprocess_glcme_image expects a 2D grayscale image.")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    img = local_glcme_equalize(img, num_tiles=num_tiles, clip_limit_ratio=clip_limit_ratio)
    return img if return_uint8 else img.astype(np.float32) / 255.0


def preprocess_gt_image(
    image_or_path: Union[PathLike, np.ndarray],
    return_uint8: bool = False,
) -> np.ndarray:
    if isinstance(image_or_path, (str, Path)):
        img = read_gray_image(image_or_path)
    else:
        img = np.asarray(image_or_path)
        if img.ndim != 2:
            raise ValueError("preprocess_gt_image expects a 2D grayscale image.")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img if return_uint8 else img.astype(np.float32) / 255.0


def _resolve_crop_size(crop_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return (crop_size, crop_size) if isinstance(crop_size, int) else crop_size


def _crop_coords_random(shape: Tuple[int, int], crop_size: Union[int, Tuple[int, int]], rng=None):
    crop_h, crop_w = _resolve_crop_size(crop_size)
    h, w = shape
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size {(crop_h, crop_w)} exceeds image size {(h, w)}")
    rng = np.random.default_rng() if rng is None else rng
    top = int(rng.integers(0, h - crop_h + 1))
    left = int(rng.integers(0, w - crop_w + 1))
    return top, left, crop_h, crop_w


def _crop_coords_center(shape: Tuple[int, int], crop_size: Union[int, Tuple[int, int]]):
    crop_h, crop_w = _resolve_crop_size(crop_size)
    h, w = shape
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size {(crop_h, crop_w)} exceeds image size {(h, w)}")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return top, left, crop_h, crop_w


def _validate_same_shape(images: Sequence[np.ndarray]) -> None:
    shapes = [tuple(img.shape) for img in images]
    if len(set(shapes)) != 1:
        raise ValueError(f"Input shape mismatch: {shapes}")


def _apply_crop(img: np.ndarray, coords) -> np.ndarray:
    top, left, crop_h, crop_w = coords
    return img[top:top + crop_h, left:left + crop_w]


def random_paired_crop(input_img, gt_img, crop_size, rng=None):
    _validate_same_shape([input_img, gt_img])
    coords = _crop_coords_random(input_img.shape, crop_size, rng)
    return _apply_crop(input_img, coords), _apply_crop(gt_img, coords)


def center_paired_crop(input_img, gt_img, crop_size):
    _validate_same_shape([input_img, gt_img])
    coords = _crop_coords_center(input_img.shape, crop_size)
    return _apply_crop(input_img, coords), _apply_crop(gt_img, coords)


def random_triplet_crop(img_a, img_b, img_c, crop_size, rng=None):
    _validate_same_shape([img_a, img_b, img_c])
    coords = _crop_coords_random(img_a.shape, crop_size, rng)
    return _apply_crop(img_a, coords), _apply_crop(img_b, coords), _apply_crop(img_c, coords)


def center_triplet_crop(img_a, img_b, img_c, crop_size):
    _validate_same_shape([img_a, img_b, img_c])
    coords = _crop_coords_center(img_a.shape, crop_size)
    return _apply_crop(img_a, coords), _apply_crop(img_b, coords), _apply_crop(img_c, coords)


def compute_start_positions(length: int, window: int, stride: int) -> List[int]:
    if window > length:
        raise ValueError(f"Window size {window} exceeds image length {length}")
    starts = list(range(0, max(1, length - window + 1), stride))
    if not starts or starts[-1] != length - window:
        starts.append(length - window)
    return starts


def sliding_window_boxes(image_shape, crop_size, stride):
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least (H, W)")
    h, w = int(image_shape[0]), int(image_shape[1])
    crop_h, crop_w = _resolve_crop_size(crop_size)
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    ys = compute_start_positions(h, crop_h, stride_h)
    xs = compute_start_positions(w, crop_w, stride_w)
    return [(top, left, top + crop_h, left + crop_w) for top in ys for left in xs]


def preprocess_and_cache_glcme_image(src_path: PathLike, dst_path: PathLike, num_tiles=(8, 8), clip_limit_ratio=0.002):
    out = preprocess_glcme_image(src_path, num_tiles=num_tiles, clip_limit_ratio=clip_limit_ratio, return_uint8=True)
    save_gray_image(out, dst_path)
