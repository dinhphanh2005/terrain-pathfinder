"""Sinh dữ liệu giả để thử thuật toán khi chưa cần dữ liệu thật.

Luôn truyền `seed`: cùng seed thì ra đúng cùng một lưới, nên test chạy lại
bao nhiêu lần cũng cho cùng kết quả.
"""
from __future__ import annotations

import numpy as np


def make_dem(rows: int, cols: int, seed: int = 0, n_hills: int = 8,
             max_height: float = 500.0) -> np.ndarray:
    """DEM giả: tổng vài ngọn đồi hình chuông (Gauss) cộng nhiễu nhỏ. Đơn vị: mét."""
    rng = np.random.default_rng(seed)
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float64)
    dem = np.zeros((rows, cols))
    for _ in range(n_hills):
        r0, c0 = rng.uniform(0, rows), rng.uniform(0, cols)
        sigma = rng.uniform(0.05, 0.25) * min(rows, cols)
        height = rng.uniform(0.3, 1.0) * max_height
        dem += height * np.exp(-((r - r0) ** 2 + (c - c0) ** 2) / (2 * sigma ** 2))
    dem += rng.normal(0.0, 0.01 * max_height, size=(rows, cols))
    return dem.astype(np.float32)


def make_cost(rows: int, cols: int, seed: int = 0, n_blocks: int = 5,
              keep_free: tuple[tuple[int, int], ...] | None = None) -> np.ndarray:
    """Bản đồ chi phí giả: giá trị dương trong [1, 10), vài khối chữ nhật cấm (= np.inf).

    Các ô trong `keep_free` (mặc định: góc trên trái và góc dưới phải) luôn đi được,
    để có chỗ đặt điểm đi và điểm đến khi thử.
    """
    rng = np.random.default_rng(seed)
    cost = (1.0 + 9.0 * rng.random((rows, cols))).astype(np.float32)
    for _ in range(n_blocks):
        h = int(rng.integers(max(1, rows // 10), max(2, rows // 4)))
        w = int(rng.integers(max(1, cols // 10), max(2, cols // 4)))
        r0 = int(rng.integers(0, max(1, rows - h)))
        c0 = int(rng.integers(0, max(1, cols - w)))
        cost[r0:r0 + h, c0:c0 + w] = np.inf
    if keep_free is None:
        keep_free = ((0, 0), (rows - 1, cols - 1))
    for rr, cc in keep_free:
        if not np.isfinite(cost[rr, cc]):
            cost[rr, cc] = 1.0
    return cost
