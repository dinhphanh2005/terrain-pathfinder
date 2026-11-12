"""Ghép các bước thành lớp hoàn chỉnh cho mỗi vùng (dùng chung cho script, giao diện, thí nghiệm).

Không chứa thuật toán mới: chỉ gọi terrain, hazards, cost theo đúng thứ tự và tham số mặc định.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src import hazards as hz
from src import terrain as tr
from src.cost import build_cost
from src.grid_io import list_layers, load_layer, save_layer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FLOOD_LEVELS = (1, 2, 4, 6, 8)          # mực nước dâng (m) được tính sẵn
DEFAULTS = {
    "mars": {"buffer_m": 60.0, "flood_m": None, "rough_radius": 1},
    "vn": {"buffer_m": 60.0, "flood_m": 4, "rough_radius": 1},
}
NAMES = {"mars": "Sao Hỏa · hố Jezero", "vn": "Việt Nam · Bảo Yên, Lào Cai"}
PLACES = {  # (hàng, cột) trên lưới, để giao diện và thí nghiệm dùng làm điểm mẫu
    "vn": {"Làng Nủ": (666, 829), "Phố Ràng": (333, 707), "Bảo Hà": (507, 174)},
    "mars": {"Chỗ Perseverance hạ cánh": (500, 500)},
}


def build_region(region: str) -> dict[str, np.ndarray]:
    """Tính mọi lớp dẫn xuất cho một vùng và lưu vào data/<region>/. Trả về dict các lớp."""
    d = DATA / region
    dem, meta = load_layer(d / "dem")
    cell = float(meta["cell_size_m"])
    base = {k: meta[k] for k in ("region", "cell_size_m", "crs", "transform") if k in meta}
    out: dict[str, np.ndarray] = {}

    def put(name, arr, unit, **extra):
        """Lưu lớp `name` (float32 .npy + .json meta: đơn vị và tham số) và giữ lại trong `out`."""
        a = np.asarray(arr, dtype=np.float32)
        save_layer(d / name, a, {**base, "layer": name, "unit": unit, **extra})
        out[name] = a

    slope, aspect = tr.slope_aspect(dem, cell)
    put("slope", slope, "deg")
    put("aspect", aspect, "deg")
    put("roughness", tr.local_std(dem, DEFAULTS[region]["rough_radius"]), "m",
        window=2 * DEFAULTS[region]["rough_radius"] + 1)
    put("hillshade", tr.hillshade(dem, cell), "0-1")

    if region == "mars":
        crater, steps = hz.crater_mask(dem, return_steps=True)
        put("tophat", steps["tophat"], "m")
        put("hazard_crater", crater, "0/1")
    else:
        L = lambda n: load_layer(d / n)[0]
        scl0, scl1 = L("s2_20240427_scl"), L("s2_20250512_scl")
        ls, st = hz.landslide_mask(L("s2_20240427_b04"), L("s2_20240427_b08"), scl0,
                                   L("s2_20250512_b04"), L("s2_20250512_b08"), scl1,
                                   slope, return_steps=True)
        put("ndvi_before", st["ndvi_before"], "NDVI")
        put("ndvi_after", st["ndvi_after"], "NDVI")
        put("dndvi", st["dndvi"], "NDVI", otsu_before=st["otsu_before"], otsu_after=st["otsu_after"])
        put("hazard_landslide", ls, "0/1")
        river = hz.river_mask(L("water"), scl0)
        put("river", river, "0/1")
        ref = hz.river_reference(dem, river)          # tính một lần cho mọi mức nước
        for h in FLOOD_LEVELS:
            put(f"hazard_flood_h{h}", hz.flood_by_level(dem, river, h, ref), "0/1", level_m=h)
        s1, thr = hz.s1_flood_change(L("s1_20240903_vv"), L("s1_20240915_vv"), slope)
        put("s1_flood_20240915", s1, "0/1", otsu_db=thr)
    return out


def load_region(region: str) -> tuple[dict[str, np.ndarray], dict]:
    """Đọc mọi lớp của vùng. meta lấy từ dem."""
    d = DATA / region
    layers = {n: load_layer(d / n)[0] for n in list_layers(d)}
    meta = load_layer(d / "dem")[1]
    return layers, meta


def hazard_union(layers: dict, region: str, flood_m=None, use_landslide=True) -> np.ndarray:
    """Hợp các lớp nguy hiểm (chưa có vùng đệm, chưa tính sông)."""
    m = np.zeros(layers["dem"].shape, bool)
    if region == "mars":
        m |= layers["hazard_crater"] > 0.5
    else:
        if use_landslide:
            m |= layers["hazard_landslide"] > 0.5
        if flood_m:
            m |= layers[f"hazard_flood_h{int(flood_m)}"] > 0.5
    return m


def make_cost(layers: dict, meta: dict, region: str, *, flood_m="default", buffer_m=None,
              weights=None, use_landslide=True, roads_weight=None, extra_hazard=None):
    """Bản đồ chi phí đầy đủ: lớp nguy hiểm -> vùng đệm -> forbidden -> build_cost.

    extra_hazard: mặt nạ nguy hiểm bổ sung (vd lớp động theo thời gian trong src/live.py),
    được hợp vào trước khi tạo vùng đệm.
    """
    cfg = DEFAULTS[region]
    flood = cfg["flood_m"] if flood_m == "default" else flood_m
    buf = cfg["buffer_m"] if buffer_m is None else buffer_m
    cell = float(meta["cell_size_m"])
    hazard = hazard_union(layers, region, flood, use_landslide)
    if extra_hazard is not None:
        hazard = hazard | np.asarray(extra_hazard, bool)
    forbidden = hz.buffer(hazard, cell, buf)
    if region == "vn":
        # Sông: không lội qua, nhưng đi được trên đường cắt ngang sông (cầu).
        river_zone = hz.buffer(layers["river"] > 0.5, cell, cell)
        forbidden |= river_zone & ~(layers["roads"] > 0.5)
    w = dict(weights or {})
    roads = None
    if roads_weight is not None and "roads" in layers:
        w["road"] = roads_weight
        roads = layers["roads"]
    cost = build_cost(layers["slope"], layers["roughness"], forbidden, cell, w, roads)
    return cost, forbidden
