"""Đọc/ghi một lớp lưới: mảng NumPy 2D (.npy) kèm file meta (.json) cùng tên.

Quy ước (xem docs/interface.md):
- mảng 2 chiều, kiểu float32, ô không có dữ liệu = NaN
- meta bắt buộc có: region, layer, cell_size_m; rows/cols do hàm tự ghi
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REQUIRED_META = ("region", "layer", "cell_size_m")


def _paths(path: str | Path) -> tuple[Path, Path]:
    p = Path(path)
    if p.suffix in (".npy", ".json"):
        p = p.with_suffix("")
    return p.with_suffix(".npy"), p.with_suffix(".json")


def save_layer(path: str | Path, array: np.ndarray, meta: dict) -> Path:
    """Lưu `array` thành <path>.npy và `meta` thành <path>.json. Trả về đường dẫn .npy."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"lớp phải là mảng 2 chiều, nhận {arr.ndim} chiều")
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"meta thiếu trường bắt buộc: {missing}")
    if not float(meta["cell_size_m"]) > 0:
        raise ValueError("cell_size_m phải dương")

    npy_path, json_path = _paths(path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    full_meta = {**meta, "rows": int(arr.shape[0]), "cols": int(arr.shape[1])}
    np.save(npy_path, arr)
    json_path.write_text(json.dumps(full_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return npy_path


def load_layer(path: str | Path) -> tuple[np.ndarray, dict]:
    """Đọc lại một lớp. Báo lỗi nếu kích thước mảng không khớp meta."""
    npy_path, json_path = _paths(path)
    arr = np.load(npy_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if arr.shape != (meta["rows"], meta["cols"]):
        raise ValueError(
            f"{npy_path.name}: mảng {arr.shape} không khớp meta ({meta['rows']}, {meta['cols']})"
        )
    return arr, meta


def list_layers(region_dir: str | Path) -> list[str]:
    """Tên các lớp (không đuôi) có đủ cả .npy và .json trong một thư mục vùng."""
    d = Path(region_dir)
    return sorted(p.stem for p in d.glob("*.npy") if p.with_suffix(".json").exists())
