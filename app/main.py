"""Giao diện demo: streamlit run app/main.py  (chạy từ thư mục gốc repo).

Chỉ gọi hàm trong src/ — không có thuật toán ở đây.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.layers import DEFAULTS, FLOOD_LEVELS, NAMES, PLACES, load_region, make_cost  # noqa: E402
from src.pathfinding import (astar, dijkstra, hierarchical_astar, line_cells, path_cost,  # noqa: E402
                             path_metrics, rest_stops, snap_to_free, weighted_astar)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # vẫn chạy được, chọn điểm bằng ô nhập số
    streamlit_image_coordinates = None

st.set_page_config(page_title="Lộ trình an toàn · Đề 26", layout="wide")
DISPLAY = 760  # bề rộng ảnh hiển thị (px)

OVERLAYS = {  # tên lớp: (khóa, màu RGB, độ đậm, vùng)
    "Sông": ("river", (20, 60, 200), 0.85, "vn"),
    "Ngập (theo mực nước)": ("__flood__", (60, 150, 255), 0.55, "vn"),
    "Sạt lở": ("hazard_landslide", (240, 80, 20), 0.9, "vn"),
    "Đường OSM": ("roads", (250, 220, 40), 0.55, "vn"),
    "Hố va chạm": ("hazard_crater", (230, 40, 40), 0.85, "mars"),
    "Vùng cấm (gồm vùng đệm)": ("__forbidden__", (120, 0, 120), 0.35, "all"),
}


@st.cache_resource(show_spinner="Đang đọc các lớp bản đồ…")
def get_region(region):
    return load_region(region)


@st.cache_data(show_spinner="Đang dựng bản đồ chi phí…", max_entries=8)
def get_cost(region, flood_m, buffer_m, w_slope, w_rough, max_slope, use_landslide):
    layers, meta = get_region(region)
    return make_cost(layers, meta, region, flood_m=flood_m, buffer_m=buffer_m,
                     weights={"slope": w_slope, "rough": w_rough, "max_slope": max_slope},
                     use_landslide=use_landslide)


def to_rgb(gray):
    g = np.nan_to_num(gray, nan=0.0)
    g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], -1).astype(np.float32)


def blend(img, mask, color, alpha):
    m = mask.astype(bool)
    img[m] = img[m] * (1 - alpha) + np.array(color, np.float32) * alpha


def compose(layers, region, chosen, flood_m, forbidden, base):
    if base == "Hillshade":
        img = to_rgb(layers["hillshade"])
    elif base == "Độ dốc":
        s = np.clip(layers["slope"] / 45.0, 0, 1)
        img = np.stack([255 * s, 255 * (1 - s), 60 + 0 * s], -1).astype(np.float32) * 0.6 + to_rgb(layers["hillshade"]) * 0.4
    else:  # Độ cao
        z = layers["dem"]; z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))
        img = np.stack([80 + 175 * z, 140 + 60 * z, 90 + 30 * (1 - z)], -1).astype(np.float32) * 0.6 + to_rgb(layers["hillshade"]) * 0.4
    for name in chosen:
        key, color, alpha, reg = OVERLAYS[name]
        if key == "__flood__":
            if flood_m:
                blend(img, layers[f"hazard_flood_h{int(flood_m)}"] > 0.5, color, alpha)
        elif key == "__forbidden__":
            blend(img, forbidden, color, alpha)
        elif key in layers:
            blend(img, layers[key] > 0.5, color, alpha)
    return img.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- thanh bên
st.sidebar.title("Lộ trình an toàn")
region = st.sidebar.radio("Vùng", ["vn", "mars"], format_func=lambda r: NAMES[r])
layers, meta = get_region(region)
cell = float(meta["cell_size_m"])
rows, cols = layers["dem"].shape

if st.session_state.get("region") != region:
    st.session_state.region = region
    st.session_state.points = []
    st.session_state.result = None

st.sidebar.subheader("Lớp nguy hiểm")
flood_m = None
use_landslide = True
if region == "vn":
    flood_m = st.sidebar.select_slider("Mực nước dâng h (m)", options=[0, *FLOOD_LEVELS],
                                       value=DEFAULTS["vn"]["flood_m"]) or None
    use_landslide = st.sidebar.checkbox("Tránh vùng sạt lở", True)
buffer_m = st.sidebar.slider("Bán kính vùng đệm (m)", 0, 150, int(DEFAULTS[region]["buffer_m"]), step=30)

with st.sidebar.expander("Trọng số chi phí"):
    w_slope = st.slider("Phạt độ dốc (× hệ số Tobler)", 0.5, 3.0, 1.0, 0.1)
    w_rough = st.slider("Phạt độ gồ ghề (/m)", 0.0, 0.3, 0.05, 0.01)
    max_slope = st.slider("Độ dốc tối đa đi bộ được (°)", 20, 60, 40, 1)

st.sidebar.subheader("Thuật toán")
method = st.sidebar.selectbox("Phương pháp", ["A*", "Dijkstra", "A* có trọng số", "A* phân cấp"])
eps = st.sidebar.slider("ε (A* có trọng số)", 1.0, 3.0, 1.5, 0.1) if method == "A* có trọng số" else 1.0
k = st.sidebar.select_slider("Hệ số thu nhỏ k (A* phân cấp)", [4, 8, 16], 8) if method == "A* phân cấp" else 8
compare = st.sidebar.checkbox("So với tuyến thẳng", True)
stop_h = st.sidebar.number_input("Dừng nghỉ mỗi (giờ đi bộ)", 0.5, 5.0, 1.0, 0.5)

st.sidebar.subheader("Hiển thị")
base = st.sidebar.radio("Nền", ["Hillshade", "Độ dốc", "Độ cao"], horizontal=True)
avail = [n for n, v in OVERLAYS.items() if v[3] in (region, "all")]
chosen = st.sidebar.multiselect("Lớp phủ", avail, default=avail)

cost, forbidden = get_cost(region, flood_m, buffer_m, w_slope, w_rough, max_slope, use_landslide)

# ---------------------------------------------------------------- bản đồ
st.title(NAMES[region])
left, right = st.columns([3, 2])
with left:
    st.caption("Bấm lên bản đồ: lần 1 chọn điểm đi, lần 2 chọn điểm đến, lần 3 chọn lại từ đầu.")
    img = Image.fromarray(compose(layers, region, chosen, flood_m, forbidden, base))
    scale = DISPLAY / cols
    disp = img.resize((DISPLAY, int(rows * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(disp)
    res = st.session_state.get("result")
    if res and res.get("path"):
        pts = [(c * scale, r * scale) for r, c in res["path"]]
        draw.line(pts, fill=(255, 0, 200), width=3)
        if res.get("line"):
            draw.line([(c * scale, r * scale) for r, c in res["line"]], fill=(255, 255, 255), width=1)
        for s in res.get("stops", []):
            r, c = s["cell"]
            draw.ellipse([c * scale - 5, r * scale - 5, c * scale + 5, r * scale + 5], fill=(255, 255, 255), outline=(0, 0, 0))
    for i, (r, c) in enumerate(st.session_state.points):
        col = (0, 200, 0) if i == 0 else (220, 0, 0)
        draw.ellipse([c * scale - 6, r * scale - 6, c * scale + 6, r * scale + 6], fill=col, outline=(255, 255, 255))

    if streamlit_image_coordinates is not None:
        click = streamlit_image_coordinates(disp, key=f"map_{region}")
        if click and click != st.session_state.get("last_click"):
            st.session_state.last_click = click
            p = (min(rows - 1, int(click["y"] / scale)), min(cols - 1, int(click["x"] / scale)))
            pts = st.session_state.points
            st.session_state.points = [p] if len(pts) != 1 else [pts[0], p]
            st.session_state.result = None
            st.rerun()
    else:
        st.image(disp)

    with st.expander("Chọn điểm bằng tọa độ ô / địa điểm mẫu"):
        places = PLACES.get(region, {})
        c1, c2 = st.columns(2)
        a = c1.selectbox("Điểm đi", ["(tự nhập)"] + list(places))
        b = c2.selectbox("Điểm đến", ["(tự nhập)"] + list(places))
        r1 = c1.number_input("Hàng đi", 0, rows - 1, places.get(a, (rows // 4, cols // 4))[0])
        q1 = c1.number_input("Cột đi", 0, cols - 1, places.get(a, (rows // 4, cols // 4))[1])
        r2 = c2.number_input("Hàng đến", 0, rows - 1, places.get(b, (3 * rows // 4, 3 * cols // 4))[0])
        q2 = c2.number_input("Cột đến", 0, cols - 1, places.get(b, (3 * rows // 4, 3 * cols // 4))[1])
        if st.button("Dùng hai điểm này"):
            st.session_state.points = [(int(r1), int(q1)), (int(r2), int(q2))]
            st.session_state.result = None
            st.rerun()

# ---------------------------------------------------------------- tìm đường
with right:
    pts = st.session_state.points
    if len(pts) < 2:
        st.info("Chọn đủ điểm đi và điểm đến.")
    else:
        if st.button("Tìm đường", type="primary") or st.session_state.get("result") is None:
            s = snap_to_free(cost, pts[0]); g = snap_to_free(cost, pts[1])
            if s is None or g is None:
                st.session_state.result = {"error": "Điểm chọn nằm sâu trong vùng cấm."}
            else:
                fn = {"A*": lambda: astar(cost, s, g), "Dijkstra": lambda: dijkstra(cost, s, g),
                      "A* có trọng số": lambda: weighted_astar(cost, s, g, eps),
                      "A* phân cấp": lambda: hierarchical_astar(cost, s, g, k=k)}[method]
                t0 = time.perf_counter(); path, tot, n_exp = fn(); dt = time.perf_counter() - t0
                res = {"path": path, "total": tot, "expanded": n_exp, "time": dt, "snapped": (s, g)}
                if path:
                    res["metrics"] = path_metrics(path, layers["dem"], layers["slope"], cost, cell)
                    res["stops"] = rest_stops(path, cost, layers["slope"], every_s=stop_h * 3600)
                    if compare:
                        res["line"] = line_cells(s, g)
                st.session_state.result = res
            st.rerun()
        res = st.session_state.result or {}
        if res.get("error"):
            st.error(res["error"])
        elif not res.get("path"):
            st.warning("Không có đường: điểm đến bị cô lập bởi vùng cấm.")
        else:
            m = res["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Thời gian đi bộ", f"{m['time_s'] / 3600:.1f} giờ")
            c2.metric("Độ dài", f"{m['length_m'] / 1000:.2f} km")
            c3.metric("Dốc lớn nhất", f"{m['max_slope_deg']:.0f}°")
            c1.metric("Tổng leo", f"{m['ascent_m']:.0f} m")
            c2.metric("Nút đã duyệt", f"{res['expanded']:,}")
            c3.metric("Thời gian tính", f"{res['time']:.2f} s")
            prof = pd.DataFrame({"Khoảng cách (km)": m["profile_dist_m"] / 1000, "Độ cao (m)": m["profile_z_m"]})
            st.subheader("Mặt cắt độ cao dọc tuyến")
            st.line_chart(prof, x="Khoảng cách (km)", y="Độ cao (m)", height=220)
            if res.get("line"):
                lc = path_cost(cost, res["line"])
                lm = path_metrics(res["line"], layers["dem"], layers["slope"], cost, cell)
                st.caption(
                    f"Tuyến thẳng (nét trắng): {lm['length_m'] / 1000:.2f} km, dốc lớn nhất {lm['max_slope_deg']:.0f}°, "
                    + ("**đi qua vùng cấm**" if not np.isfinite(lc) else f"{lc / 3600:.1f} giờ"))
            if res.get("stops"):
                st.subheader("Điểm dừng nghỉ")
                st.dataframe(pd.DataFrame([{"Sau (giờ)": round(s["time_s"] / 3600, 2), "Hàng": s["cell"][0],
                                            "Cột": s["cell"][1], "Độ dốc (°)": round(float(layers["slope"][s["cell"]]), 1)}
                                           for s in res["stops"]]), hide_index=True)

st.caption("Dữ liệu: USGS Astrogeology (CTX DEM Jezero), NASA NASADEM, ESA Copernicus Sentinel-1/2, "
           "© OpenStreetMap contributors (ODbL).")
