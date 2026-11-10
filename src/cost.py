"""Bản đồ chi phí: số giây để đi bộ qua mỗi ô.

Tốc độ theo độ dốc: hàm đi bộ Tobler (1993)  v = 6·exp(-3,5·|S + 0,05|)  km/h, S = tan(độ dốc).
Lưới không biết đi lên hay đi xuống (chi phí đặt cho ô, không cho cạnh) nên dùng bản đối xứng:
S = |tan θ| và bỏ số hạng 0,05 -> v_max = 6 km/h trên đất bằng. Ghi rõ đây là xấp xỉ.
"""
from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = {
    "slope": 1.0,       # nhân vào hệ số 3,5 của Tobler: >1 phạt dốc nặng hơn
    "rough": 0.05,      # thêm 5% thời gian cho mỗi mét độ gồ ghề
    "max_slope": 40.0,  # dốc hơn mức này (độ) coi như không đi bộ được
    "road": 1.0,        # nhân thời gian trên ô có đường (<1: ưu tiên đường)
}


def tobler_speed_kmh(slope_deg: np.ndarray, slope_weight: float = 1.0) -> np.ndarray:
    """Tốc độ đi bộ (km/h) theo độ dốc (độ), bản đối xứng của hàm Tobler."""
    s = np.abs(np.tan(np.radians(np.asarray(slope_deg, np.float64))))
    return 6.0 * np.exp(-3.5 * slope_weight * s)


def build_cost(slope_deg: np.ndarray, roughness: np.ndarray, forbidden: np.ndarray | None,
               cell_size_m: float, weights: dict | None = None,
               roads: np.ndarray | None = None) -> np.ndarray:
    """Chi phí ô C(r,c) = Δ / v(θ) × (1 + w_r·σ) × (hệ số đường), đơn vị giây.

    Ô cấm (forbidden, dốc quá max_slope, không có dữ liệu) = np.inf.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    v_ms = tobler_speed_kmh(slope_deg, w["slope"]) / 3.6
    cost = cell_size_m / v_ms * (1.0 + w["rough"] * np.nan_to_num(roughness, nan=0.0))
    if roads is not None and w["road"] != 1.0:
        cost = np.where(np.asarray(roads) > 0.5, cost * w["road"], cost)
    cost = cost.astype(np.float32)
    bad = ~np.isfinite(cost) | (np.asarray(slope_deg) > w["max_slope"]) | np.isnan(slope_deg)
    if forbidden is not None:
        bad |= np.asarray(forbidden) > 0.5
    cost[bad] = np.inf
    return cost
