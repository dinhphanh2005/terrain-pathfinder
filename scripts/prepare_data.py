"""Biến dữ liệu thô trong data/raw/ thành các lớp .npy trên một lưới chung cho mỗi vùng.

Chạy từ thư mục gốc repo:
    python scripts/prepare_data.py            # cả hai vùng
    python scripts/prepare_data.py --only vn  # chỉ Việt Nam
    python scripts/prepare_data.py --preview  # thêm ảnh xem nhanh vào data/preview/

Script này CHỈ cắt, chiếu lại và đổi định dạng. Không tính độ dốc, NDVI, Otsu...
(đó là thuật toán cốt lõi, nằm trong src/).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_geom
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.grid_io import save_layer  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

# ---------------------------------------------------------------- Sao Hỏa
MARS_DEM = RAW / "M20_JezeroCrater_CTXDEM_20m.tif"
# Tâm cửa sổ mặc định: chỗ rover Perseverance hạ cánh, cạnh châu thổ Jezero.
MARS_CENTER = (18.4447, 77.4508)  # (vĩ độ, kinh độ)
MARS_SIZE = 1000                  # 1000 x 1000 ô x 20 m = 20 x 20 km


def mars_lonlat_to_xy(lat: float, lon: float, crs_params: dict) -> tuple[float, float]:
    """Phép chiếu equirectangular: x = R*cos(lat_ts)*lon, y = R*lat (lon, lat tính bằng radian)."""
    R = float(crs_params["R"])
    lat_ts = math.radians(float(crs_params.get("lat_ts", 0.0)))
    return R * math.cos(lat_ts) * math.radians(lon), R * math.radians(lat)


def prepare_mars(center=MARS_CENTER, size=MARS_SIZE) -> dict[str, np.ndarray]:
    with rasterio.open(MARS_DEM) as src:
        params = src.crs.to_dict()
        x, y = mars_lonlat_to_xy(*center, params)
        row, col = src.index(x, y)
        r0 = int(np.clip(row - size // 2, 0, src.height - size))
        c0 = int(np.clip(col - size // 2, 0, src.width - size))
        win = Window(c0, r0, size, size)
        dem = src.read(1, window=win).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        dem[dem < -1e30] = np.nan
        transform = src.window_transform(win)
        cell = abs(transform.a)
        meta = {
            "region": "mars", "layer": "dem", "cell_size_m": cell, "unit": "m",
            "source": MARS_DEM.name, "crs": src.crs.to_proj4() if hasattr(src.crs, "to_proj4") else str(src.crs),
            "transform": list(transform)[:6],
            "window": {"row_off": r0, "col_off": c0, "center_latlon": list(center)},
        }
    save_layer(OUT / "mars" / "dem", dem, meta)
    print(f"[mars] dem {dem.shape}, {cell} m/ô, NaN {np.isnan(dem).mean():.2%}, "
          f"độ cao {np.nanmin(dem):.0f}..{np.nanmax(dem):.0f} m")
    return {"dem": dem}


# ---------------------------------------------------------------- Việt Nam
VN = RAW / "vn"
VN_CELL = 30.0
VN_CRS = "EPSG:32648"  # UTM 48N
S2_DATES = ["20240427", "20250512"]
S1_DATES = ["20240903", "20240915", "20240927"]
# Những loại đường người đi bộ không dùng được
NOT_WALKABLE = {"motorway", "motorway_link", "construction", "proposed", "raceway", "bus_guideway"}

# Bảng màu "Scene classification map" của Copernicus Browser -> mã lớp SCL
SCL_PALETTE = {
    0: (0.0, 0.0, 0.0),        # không có dữ liệu
    1: (1.0, 0.0, 0.0),        # bão hoà / lỗi
    2: (0.184, 0.184, 0.184),  # bóng địa hình (vùng tối)
    3: (0.392, 0.196, 0.0),    # bóng mây
    4: (0.0, 0.627, 0.0),      # thực vật
    5: (1.0, 0.902, 0.353),    # không có thực vật
    6: (0.0, 0.0, 1.0),        # nước
    7: (0.502, 0.502, 0.502),  # chưa phân loại
    8: (0.753, 0.753, 0.753),  # mây, xác suất trung bình
    9: (1.0, 1.0, 1.0),        # mây, xác suất cao
    10: (0.392, 0.784, 1.0),   # mây ti mỏng (cirrus)
    11: (1.0, 0.588, 1.0),     # tuyết
}


def vn_grid() -> tuple[Affine, int, int]:
    """Lưới chung 30 m: cùng gốc và cùng phạm vi với ảnh Sentinel-2 đã tải (đã cắt theo aoi_vn)."""
    with rasterio.open(VN / f"s2_{S2_DATES[0]}" / "B04.tif") as s:
        b = s.bounds
    cols = int((b.right - b.left) // VN_CELL)
    rows = int((b.top - b.bottom) // VN_CELL)
    return Affine(VN_CELL, 0, b.left, 0, -VN_CELL, b.top), rows, cols


def warp(src_path: Path, transform: Affine, shape: tuple[int, int], method: Resampling,
         band: int = 1, src_nodata=None) -> np.ndarray:
    dst = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as s:
        data = s.read(band).astype(np.float32)
        nd = s.nodata if src_nodata is None else src_nodata
        if nd is not None and not (isinstance(nd, float) and math.isnan(nd)):
            data[data == nd] = np.nan
        reproject(data, dst, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=transform, dst_crs=VN_CRS, resampling=method,
                  src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def decode_scl(path: Path):
    """Đổi ảnh màu SCL (3 băng RGB 0..1) về mã lớp bằng màu gần nhất trong bảng."""
    with rasterio.open(path) as s:
        rgb = s.read().astype(np.float32)  # (3, rows, cols)
        tr, crs = s.transform, s.crs
    codes = np.array(list(SCL_PALETTE.keys()), dtype=np.float32)
    pal = np.array(list(SCL_PALETTE.values()), dtype=np.float32)  # (k, 3)
    px = rgb.reshape(3, -1).T                                      # (n, 3)
    d2 = ((px[:, None, :] - pal[None, :, :]) ** 2).sum(-1)        # (n, k)
    best = d2.argmin(1)
    unmatched = float((d2.min(1) > 0.01).mean())
    return codes[best].reshape(rgb.shape[1:]), unmatched, tr, crs


def warp_array(arr, src_tr, src_crs, transform, shape, method) -> np.ndarray:
    dst = np.full(shape, np.nan, dtype=np.float32)
    reproject(arr.astype(np.float32), dst, src_transform=src_tr, src_crs=src_crs,
              dst_transform=transform, dst_crs=VN_CRS, resampling=method,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def read_swb(transform: Affine, shape) -> np.ndarray:
    """Mặt nạ sông hồ NASADEM: file nhị phân 3601x3601 uint8, 0 = đất, 255 = nước."""
    swb = np.fromfile(VN / "nasadem_n22e104" / "n22e104.swb", dtype=np.uint8).reshape(3601, 3601)
    with rasterio.open(VN / "nasadem_n22e104" / "n22e104.hgt") as s:  # dùng chung georeference
        tr, crs = s.transform, s.crs
    water = (swb == 255).astype(np.float32)
    return warp_array(water, tr, crs, transform, shape, Resampling.nearest)


def rasterize_roads(transform: Affine, shape) -> tuple[np.ndarray, int, int]:
    gj = json.loads((VN / "osm_roads.geojson").read_text(encoding="utf-8"))
    geoms, skipped = [], 0
    for f in gj["features"]:
        hw = f["properties"].get("highway")
        if f["geometry"]["type"] != "LineString" or hw in NOT_WALKABLE:
            skipped += 1
            continue
        geoms.append(transform_geom("EPSG:4326", VN_CRS, f["geometry"]))
    roads = rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                      fill=0, all_touched=True, dtype="uint8")
    return roads.astype(np.float32), len(geoms), skipped


def prepare_vn() -> dict[str, np.ndarray]:
    transform, rows, cols = vn_grid()
    shape = (rows, cols)
    base = {"region": "vn", "cell_size_m": VN_CELL, "crs": VN_CRS, "transform": list(transform)[:6]}
    layers: dict[str, np.ndarray] = {}

    def put(name, arr, unit, source, **extra):
        save_layer(OUT / "vn" / name, arr, {**base, "layer": name, "unit": unit, "source": source, **extra})
        layers[name] = arr
        nan = np.isnan(arr).mean()
        print(f"[vn] {name:22s} {arr.shape}  NaN {nan:6.2%}  {np.nanmin(arr):.4g}..{np.nanmax(arr):.4g}")

    # DEM: liên tục -> nội suy song tuyến (bilinear)
    put("dem", warp(VN / "nasadem_n22e104" / "n22e104.hgt", transform, shape, Resampling.bilinear),
        "m", "NASADEM_HGT n22e104")
    # Mặt nạ sông hồ: phân loại -> lân cận gần nhất (nearest)
    put("water", read_swb(transform, shape), "0/1", "NASADEM SWB n22e104")

    for d in S2_DATES:
        for b in ("B04", "B08"):
            # Phản xạ 15 m -> 30 m: lấy trung bình 4 ô con (average)
            layers[f"s2_{d}_{b.lower()}"] = warp(VN / f"s2_{d}" / f"{b}.tif", transform, shape,
                                                 Resampling.average)
        scl, unmatched, tr, crs = decode_scl(VN / f"s2_{d}" / "SCL_vis.tif")
        # Mã lớp -> lấy lớp xuất hiện nhiều nhất trong 4 ô con (mode), không bao giờ nội suy
        scl30 = warp_array(scl, tr, crs, transform, shape, Resampling.mode)
        scl30[scl30 == 0] = np.nan  # lớp 0 = không có dữ liệu (góc ngoài khung cắt)
        put(f"s2_{d}_scl", scl30, "SCL class", f"Sentinel-2 L2A {d} SCL",
            unmatched_color_fraction=unmatched)
        # Ô không có dữ liệu trong SCL thì phản xạ (đang = 0) cũng không có dữ liệu
        for b in ("b04", "b08"):
            name = f"s2_{d}_{b}"
            layers[name][np.isnan(scl30)] = np.nan
            put(name, layers[name], "reflectance", f"Sentinel-2 L2A {d} {b.upper()}")

    for d in S1_DATES:
        # Tán xạ ngược ở dạng tuyến tính: trung bình phải lấy trên số tuyến tính, không lấy trên dB
        put(f"s1_{d}_vv", warp(VN / f"s1_{d}" / "VV.tif", transform, shape, Resampling.average),
            "gamma0 linear", f"Sentinel-1 IW VV ascending {d}")

    roads, n_used, n_skipped = rasterize_roads(transform, shape)
    put("roads", roads, "0/1", "OpenStreetMap highway (bỏ đường người đi bộ không dùng được)",
        n_ways=n_used, n_ways_skipped=n_skipped)
    return layers


# ---------------------------------------------------------------- ảnh xem nhanh
def save_previews(region: str, layers: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = OUT / "preview"
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in layers.items():
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        cmap = "terrain" if name == "dem" else ("gray" if "s1_" in name else "viridis")
        if "s1_" in name:
            arr = 10 * np.log10(np.clip(arr, 1e-4, None))  # chỉ để xem: đổi sang dB
        im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
        ax.set_title(f"{region}/{name}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(out / f"{region}_{name}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"[{region}] ảnh xem nhanh: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["mars", "vn"])
    ap.add_argument("--preview", action="store_true", help="lưu ảnh PNG xem nhanh vào data/preview/")
    args = ap.parse_args()
    if args.only in (None, "mars"):
        layers = prepare_mars()
        if args.preview:
            save_previews("mars", layers)
    if args.only in (None, "vn"):
        layers = prepare_vn()
        if args.preview:
            save_previews("vn", layers)


if __name__ == "__main__":
    main()
