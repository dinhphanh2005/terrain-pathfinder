"""Tái tạo toàn bộ bảng và hình Chương 4 (và bảng kiểm chứng Chương 3) bằng một lệnh:

    python experiments/run_experiments.py            # đầy đủ (~5–10 phút)
    python experiments/run_experiments.py --quick    # ít cặp điểm, để thử nhanh
    python experiments/run_experiments.py --resume   # dùng lại phần đã tính (bảng .csv đã có) nếu bị ngắt giữa chừng

Kết quả: experiments/results/tables/*.csv, experiments/results/figures/*.png, experiments/results/summary.md
Seed cố định (SEED) nên chạy lại cho cùng kết quả (trừ cột thời gian chạy).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402
from skimage.graph import MCP_Geometric  # noqa: E402

from src import hazards as hz  # noqa: E402
from src import terrain as tr  # noqa: E402
from src.cost import build_cost  # noqa: E402
from src.layers import DEFAULTS, FLOOD_LEVELS, NAMES, PLACES, load_region, make_cost  # noqa: E402
from src.pathfinding import (astar, dijkstra, hierarchical_astar, line_cells, path_cost,  # noqa: E402
                             path_metrics, snap_to_free, weighted_astar)

SEED = 2026
OUT = ROOT / "experiments" / "results"
TAB, FIG = OUT / "tables", OUT / "figures"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10})


def pm(x):
    """'trung bình ± độ lệch chuẩn' cho cột số (bỏ NaN/inf)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "—"
    return f"{x.mean():.3g} ± {x.std(ddof=1) if len(x) > 1 else 0:.2g}"


def sample_pairs(cost, n, rng, min_cells=300):
    rows, cols = cost.shape
    free = np.argwhere(np.isfinite(cost))
    pairs = []
    while len(pairs) < n:
        a, b = free[rng.integers(len(free))], free[rng.integers(len(free))]
        if np.hypot(*(a - b)) >= min_cells:
            p, c, _ = astar(cost, tuple(a), tuple(b))
            if p:                                   # chỉ lấy cặp nối được (cặp cô lập để thí nghiệm c)
                pairs.append((tuple(map(int, a)), tuple(map(int, b))))
    return pairs


def mcp_cost(cost, s, g):
    m = MCP_Geometric(cost.astype(np.float64), fully_connected=True)
    cum, _ = m.find_costs([s], [g])
    return float(cum[g])


def run(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


# ---------------------------------------------------------------- kiểm chứng (Chương 3.5)
def verification(L, meta):
    rows = []
    dem = L["dem"].astype(np.float64)
    cell = float(meta["cell_size_m"])

    def add(name, ref_name, a, b, t_own, t_ref, note=""):
        d = np.asarray(a, float) - np.asarray(b, float)
        d = d[np.isfinite(d)]
        rows.append({"Hàm tự cài": name, "Đối chiếu": ref_name,
                     "Sai số tuyệt đối lớn nhất": float(np.abs(d).max()) if d.size else 0.0,
                     "MSE": float((d ** 2).mean()) if d.size else 0.0,
                     "t tự cài (s)": round(t_own, 4), "t thư viện (s)": round(t_ref, 4), "Ghi chú": note})

    k = np.random.default_rng(1).random((3, 3))
    (own, t1), (ref, t2) = run(tr.correlate3x3, dem, k), run(ndimage.correlate, dem, k, mode="nearest")
    add("Tương quan 3×3", "scipy.ndimage.correlate", own, ref, t1, t2)
    (sa, t1) = run(tr.slope_aspect, dem, cell)
    t0 = time.perf_counter(); gy, gx = np.gradient(dem, cell)
    ref = np.degrees(np.arctan(np.hypot(gx, gy))); t2 = time.perf_counter() - t0
    add("Độ dốc (Sobel)", "np.gradient", sa[0][2:-2, 2:-2], ref[2:-2, 2:-2], t1, t2,
        "Khác do Sobel làm trơn thêm (1,2,1); không phải lỗi")
    sub = dem[:300, :300]
    (own, t1), (ref, t2) = run(tr.local_std, sub, 1), run(ndimage.generic_filter, sub, np.std, size=3, mode="nearest")
    add("Độ lệch chuẩn cục bộ 3×3", "ndimage.generic_filter(np.std)", own, ref, t1, t2, "Vùng 300×300 ô")
    if "dndvi" in L:
        v = L["ndvi_after"][np.isfinite(L["ndvi_after"])]
        (own, t1), (ref, t2) = run(hz.otsu_threshold, v), run(threshold_otsu, v)
        add("Ngưỡng Otsu", "skimage.filters.threshold_otsu", [own], [ref], t1, t2, f"Trên NDVI sau bão, t = {own:.4f}")
        m = L["hazard_landslide"] > 0.5
    else:
        m = L["hazard_crater"] > 0.5
    se = hz.disk(2)
    (own, t1), (ref, t2) = run(hz.dilate, m, se), run(ndimage.binary_dilation, m, se)
    add("Giãn nhị phân, đĩa r=2", "ndimage.binary_dilation", own, ref, t1, t2)
    (own, t1), (ref, t2) = run(hz.erode, m, se), run(ndimage.binary_erosion, m, se, border_value=1)
    add("Co nhị phân, đĩa r=2", "ndimage.binary_erosion", own, ref, t1, t2)
    (own, t1), (ref, t2) = run(hz.grey_dilate, dem, se), run(ndimage.grey_dilation, dem, footprint=se, mode="nearest")
    add("Giãn mức xám, đĩa r=2", "ndimage.grey_dilation", own, ref, t1, t2)
    ((lab, n), t1), ((ref, nref), t2) = run(hz.label, m), run(ndimage.label, m, np.ones((3, 3)))
    add("Gán nhãn 8 láng giềng (số thành phần)", "ndimage.label", [n], [nref], t1, t2, f"{n} thành phần")
    return rows


# ---------------------------------------------------------------- thí nghiệm (a)
def exp_a(region, L, meta, cost, pairs, with_mcp=True):
    cell = float(meta["cell_size_m"])
    road_cost = None
    if region == "vn":
        road_cost, _ = make_cost(L, meta, region, roads_weight=0.5)
    out = []
    for i, (s, g) in enumerate(pairs):
        (pd_, cd, ed), td = run(dijkstra, cost, s, g)
        methods = {"Dijkstra": ((pd_, cd, ed), td),
                   "A*": run(astar, cost, s, g),
                   "A* có trọng số (ε=1,5)": run(weighted_astar, cost, s, g, 1.5),
                   "A* phân cấp (k=8)": run(hierarchical_astar, cost, s, g, 8)}
        if road_cost is not None:
            (pr, _, er), trd = run(astar, road_cost, s, g)
            methods["Ưu tiên đường OSM"] = ((pr, path_cost(cost, pr) if pr else np.inf, er), trd)
        line = line_cells(s, g)
        lc = path_cost(cost, line)
        mcp = mcp_cost(cost, s, g) if with_mcp else np.nan
        for name, ((p, c, e), t) in methods.items():
            mtr = path_metrics(p, L["dem"], L["slope"], cost, cell)
            out.append({"vùng": region, "cặp": i, "phương pháp": name, "chi phí (giờ)": c / 3600,
                        "tỉ số chi phí": c / cd, "độ dài (km)": mtr["length_m"] / 1000,
                        "dốc lớn nhất (°)": mtr["max_slope_deg"], "tổng leo (m)": mtr["ascent_m"],
                        "nút duyệt": e, "thời gian chạy (s)": t,
                        "lệch so với MCP": abs(c - mcp) if name in ("Dijkstra", "A*") else np.nan})
        lm = path_metrics(line, L["dem"], L["slope"], cost, cell)
        blocked = int(np.sum(~np.isfinite(cost[tuple(np.array(line).T)])))
        out.append({"vùng": region, "cặp": i, "phương pháp": "Tuyến thẳng", "chi phí (giờ)": lc / 3600,
                    "tỉ số chi phí": lc / cd, "độ dài (km)": lm["length_m"] / 1000,
                    "dốc lớn nhất (°)": lm["max_slope_deg"], "tổng leo (m)": lm["ascent_m"],
                    "nút duyệt": 0, "thời gian chạy (s)": 0.0, "ô cấm trên tuyến": blocked})
        print(f"  [{region}] cặp {i + 1}/{len(pairs)}", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- thí nghiệm (b)
def coarsen_layers(L, f):
    """Lưới thô f lần: DEM trung bình khối; mặt nạ = có ít nhất một ô bật trong khối."""
    R, C = L["dem"].shape[0] // f, L["dem"].shape[1] // f
    blk = lambda a: a[:R * f, :C * f].reshape(R, f, C, f).swapaxes(1, 2).reshape(R, C, f * f)
    out = {"dem": np.nanmean(blk(L["dem"].astype(np.float64)), -1)}
    for k in L:
        if k.startswith("hazard_") or k in ("river", "roads"):
            out[k] = (blk(L[k]) > 0.5).any(-1).astype(np.float32)
    return out


def exp_b(region, L, meta, pairs):
    cell = float(meta["cell_size_m"])
    rows = []

    def evaluate(group, value, cost, slope, dem, cellsize, pairs_, eps=1.0, scale=1):
        for i, (s, g) in enumerate(pairs_):
            s2, g2 = (s[0] // scale, s[1] // scale), (g[0] // scale, g[1] // scale)
            s2, g2 = snap_to_free(cost, s2), snap_to_free(cost, g2)
            if s2 is None or g2 is None:
                rows.append({"vùng": region, "tham số": group, "giá trị": value, "cặp": i, "tìm được": 0}); continue
            (p, c, e), t = run(weighted_astar if eps != 1.0 else astar, cost, s2, g2, *( [eps] if eps != 1.0 else []))
            if not p:
                rows.append({"vùng": region, "tham số": group, "giá trị": value, "cặp": i, "tìm được": 0}); continue
            mtr = path_metrics(p, dem, slope, cost, cellsize)
            rows.append({"vùng": region, "tham số": group, "giá trị": value, "cặp": i, "tìm được": 1,
                         "chi phí (giờ)": c / 3600, "độ dài (km)": mtr["length_m"] / 1000,
                         "dốc lớn nhất (°)": mtr["max_slope_deg"], "nút duyệt": e, "thời gian chạy (s)": t})

    base = lambda **kw: make_cost(L, meta, region, **kw)[0]
    for w in (0.5, 1.0, 2.0):
        evaluate("trọng số độ dốc", w, base(weights={"slope": w}), L["slope"], L["dem"], cell, pairs)
    for w in (0.0, 0.05, 0.15):
        evaluate("trọng số gồ ghề (/m)", w, base(weights={"rough": w}), L["slope"], L["dem"], cell, pairs)
    for b in (0, 60, 120):
        evaluate("vùng đệm (m)", b, base(buffer_m=b), L["slope"], L["dem"], cell, pairs)
    if region == "vn":
        for h in (0, 2, 4, 8):
            evaluate("mực nước h (m)", h, base(flood_m=h or None), L["slope"], L["dem"], cell, pairs)
    for e in (1.0, 1.2, 1.5, 2.0, 3.0):
        evaluate("ε (A* có trọng số)", e, base(), L["slope"], L["dem"], cell, pairs, eps=e)
    # độ phân giải: lưới gốc so với lưới thô 3 lần
    evaluate("ô lưới (m)", int(cell), base(), L["slope"], L["dem"], cell, pairs)
    C = coarsen_layers(L, 3)
    c3 = cell * 3
    sl3, _ = tr.slope_aspect(C["dem"], c3)
    rough3 = tr.local_std(C["dem"], 1)
    hzm = (C["hazard_crater"] > 0.5) if region == "mars" else (C["hazard_landslide"] > 0.5) | (C[f"hazard_flood_h{DEFAULTS['vn']['flood_m']}"] > 0.5)
    forb = hz.buffer(hzm, c3, DEFAULTS[region]["buffer_m"])
    if region == "vn":
        forb |= hz.buffer(C["river"] > 0.5, c3, c3) & ~(C["roads"] > 0.5)
    cost3 = build_cost(sl3, rough3, forb, c3)
    evaluate("ô lưới (m)", int(c3), cost3, sl3, C["dem"], c3, pairs, scale=3)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- thí nghiệm (c)
def exp_c(region, L, meta, cost, pairs, rng):
    rows = []
    s, g = pairs[0]
    # 1. đích bị cô lập hoàn toàn: vòng cấm bán kính 15 ô quanh đích
    c1 = cost.copy()
    rr, cc = np.ogrid[:cost.shape[0], :cost.shape[1]]
    d = np.hypot(rr - g[0], cc - g[1])
    c1[(d >= 15) & (d < 18)] = np.inf
    for name, fn in (("Dijkstra", dijkstra), ("A*", astar)):
        (p, c, e), t = run(fn, c1, s, g)
        rows.append({"vùng": region, "tình huống": "Đích bị cô lập", "phương pháp": name,
                     "tìm được": bool(p), "nút duyệt": e, "thời gian chạy (s)": t,
                     "nhận xét": "báo đúng 'không có đường'" if not p else "SAI"})
    # 2. lối đi hẹp 1 ô xuyên tường cấm
    c2 = cost.copy()
    mid = (s[1] + g[1]) // 2
    c2[:, mid - 2:mid + 3] = np.inf
    gap = cost.shape[0] // 3
    c2[gap, mid - 2:mid + 3] = cost[gap, mid - 2:mid + 3]
    for name, fn in (("A*", astar), ("A* phân cấp (k=8)", lambda *a: hierarchical_astar(*a, 8)),
                     ("A* phân cấp (k=16)", lambda *a: hierarchical_astar(*a, 16))):
        (p, c, e), t = run(fn, c2, s, g)
        rows.append({"vùng": region, "tình huống": "Lối đi hẹp 1 ô", "phương pháp": name,
                     "tìm được": bool(p), "chi phí (giờ)": c / 3600 if p else np.nan, "nút duyệt": e,
                     "thời gian chạy (s)": t,
                     "nhận xét": "đi qua khe" if p and any(q[1] == mid for q in p) else "—"})
    # 3. DEM có lỗ trống 2% (các khối 5x5 NaN): ô NaN -> cấm
    dem = L["dem"].astype(np.float64).copy()
    for _ in range(int(0.02 * dem.size / 25)):
        r0, c0 = rng.integers(0, dem.shape[0] - 5), rng.integers(0, dem.shape[1] - 5)
        dem[r0:r0 + 5, c0:c0 + 5] = np.nan
    sl, _ = tr.slope_aspect(dem, float(meta["cell_size_m"]))
    cost3 = np.where(np.isnan(sl), np.inf, cost)
    ok = 0; ratios = []
    for s_, g_ in pairs:
        s2, g2 = snap_to_free(cost3, s_), snap_to_free(cost3, g_)
        p, c, _ = astar(cost3, s2, g2)
        _, c0_, _ = astar(cost, s_, g_)
        ok += bool(p)
        if p:
            ratios.append(c / c0_)
    rows.append({"vùng": region, "tình huống": "DEM có lỗ trống 2%", "phương pháp": "A*",
                 "tìm được": f"{ok}/{len(pairs)}", "nhận xét": f"chi phí tăng {pm(np.array(ratios) - 1)} (tỉ lệ)"})
    if region == "vn":
        # 4. mây: lớp sạt lở khi loại thêm mây ti mỏng (SCL 10)
        LL = {k: L[k] for k in L}
        a = hz.landslide_mask(L["s2_20240427_b04"], L["s2_20240427_b08"], L["s2_20240427_scl"],
                              L["s2_20250512_b04"], L["s2_20250512_b08"], L["s2_20250512_scl"], L["slope"])
        b = hz.landslide_mask(L["s2_20240427_b04"], L["s2_20240427_b08"], L["s2_20240427_scl"],
                              L["s2_20250512_b04"], L["s2_20250512_b08"], L["s2_20250512_scl"], L["slope"],
                              drop_cirrus=True)
        rows.append({"vùng": region, "tình huống": "Mây ti mỏng (SCL=10)", "phương pháp": "Lớp sạt lở",
                     "nhận xét": f"giữ SCL 10: {a.sum() * 0.0009:.2f} km²; loại SCL 10: {b.sum() * 0.0009:.2f} km² "
                                 f"(mất {100 * (1 - b.sum() / max(a.sum(), 1)):.0f}%)"})
        # 5. mực nước dâng: bao nhiêu cặp còn nối được
        for h in FLOOD_LEVELS:
            ch, _ = make_cost(L, meta, region, flood_m=h)
            ok = sum(bool(astar(ch, snap_to_free(ch, s_) or s_, snap_to_free(ch, g_) or g_)[0]) for s_, g_ in pairs)
            rows.append({"vùng": region, "tình huống": f"Mực nước dâng {h} m", "phương pháp": "A*",
                         "tìm được": f"{ok}/{len(pairs)}", "nhận xét": f"diện tích ngập {L[f'hazard_flood_h{h}'].sum() * 0.0009:.1f} km²"})
        # 6. kiểm chứng lớp ngập bằng Sentinel-1
        s1 = L["s1_flood_20240915"] > 0.5
        for h in FLOOD_LEVELS:
            f = L[f"hazard_flood_h{h}"] > 0.5
            inter = (f & s1).sum()
            rows.append({"vùng": region, "tình huống": f"So lớp ngập h={h} m với Sentinel-1 15/9", "phương pháp": "—",
                         "nhận xét": f"{100 * inter / max(s1.sum(), 1):.0f}% ô ngập radar nằm trong lớp; IoU {inter / max((f | s1).sum(), 1):.3f}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ví dụ số 5x5 (Chương 2)
def worked_examples(L):
    lines = ["# Ví dụ số tính tay (đã kiểm chứng bằng NumPy)\n"]
    dem = np.round(L["dem"][660:665, 826:831].astype(np.float64), 1)
    cell = 30.0
    gx = tr.correlate3x3(dem, tr.SOBEL_X); gy = tr.correlate3x3(dem, tr.SOBEL_Y)
    sl, asp = tr.slope_aspect(dem, cell)
    lines += ["## 1. Độ dốc bằng Sobel tại ô giữa (hàng 2, cột 2) của DEM 5×5 gần Làng Nủ, ô 30 m\n",
              "DEM (m):\n", "```", np.array2string(dem, precision=1, suppress_small=True), "```",
              f"\nG_x (Sobel theo cột) tại ô giữa = {gx[2, 2]:.2f}; G_y (theo hàng) = {gy[2, 2]:.2f}",
              f"\ndz/dx = G_x / (8·30) = {gx[2, 2] / 240:.4f}; dz/dy (Bắc) = −G_y / (8·30) = {-gy[2, 2] / 240:.4f}",
              f"\nĐộ dốc = arctan(√(dz/dx² + dz/dy²)) = **{sl[2, 2]:.2f}°**; hướng dốc = **{asp[2, 2]:.1f}°**\n"]
    cost = np.array([[1, 1, 1, 1, 1], [1, 9, 9, 9, 1], [1, 1, 1, 9, 1], [9, 9, 1, 9, 1], [1, 1, 1, 1, 1]], float)
    s, g = (0, 0), (4, 4)
    # A* từng bước, ghi lại thứ tự lấy ra khỏi hàng đợi
    import heapq, math
    SQ = math.sqrt(2)
    cmin = cost.min()
    h = lambda r, c: ((SQ - 1) * min(abs(r - 4), abs(c - 4)) + max(abs(r - 4), abs(c - 4))) * cmin
    gs = {s: 0.0}; par = {s: None}; heap = [(h(*s), 0.0, s)]; closed = set(); steps = []
    while heap:
        f, gu, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        steps.append((len(steps) + 1, u, round(gu, 3), round(h(*u), 3), round(f, 3)))
        if u == g:
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                v = (u[0] + dr, u[1] + dc)
                if 0 <= v[0] < 5 and 0 <= v[1] < 5 and v not in closed:
                    ng = gu + 0.5 * (cost[u] + cost[v]) * (SQ if dr and dc else 1)
                    if ng < gs.get(v, math.inf):
                        gs[v] = ng; par[v] = u; heapq.heappush(heap, (ng + h(*v), ng, v))
    path = []; u = g
    while u is not None:
        path.append(u); u = par[u]
    path = path[::-1]
    p2, c2, e2 = astar(cost, s, g); pd_, cd, ed = dijkstra(cost, s, g)
    lines += ["## 2. A* trên lưới chi phí 5×5, đi từ (0,0) đến (4,4)\n", "Chi phí ô:\n", "```",
              np.array2string(cost.astype(int)), "```",
              "\nChi phí cạnh = (C[u] + C[v]) / 2 × (1 hoặc √2); h = octile × C_min (C_min = 1).\n",
              "| Bước | Ô lấy ra | g | h | f = g + h |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {u} | {gg} | {hh} | {ff} |" for k, u, gg, hh, ff in steps]
    lines += [f"\nĐường: {path}", f"\nTổng chi phí A* = **{gs[g]:.4f}** (hàm src.pathfinding.astar: {c2:.4f}, "
              f"{e2} nút); Dijkstra: {cd:.4f}, **{ed} nút**.\n"]
    (OUT / "worked_examples.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- hình
def figures(region, L, cost, pairs, A):
    hs = L["hillshade"]
    # 1. lớp nguy hiểm + tuyến các phương pháp cho cặp đầu
    s, g = pairs[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(hs, cmap="gray")
    ov = np.zeros(hs.shape + (4,))
    if region == "vn":
        ov[L[f"hazard_flood_h{DEFAULTS['vn']['flood_m']}"] > 0.5] = [0.2, 0.5, 1, 0.6]
        ov[L["river"] > 0.5] = [0, 0, 0.6, 0.9]
        ov[L["hazard_landslide"] > 0.5] = [1, 0.3, 0, 0.9]
    else:
        ov[L["hazard_crater"] > 0.5] = [1, 0.2, 0.2, 0.9]
    ax.imshow(ov)
    styles = {"A*": ("m", 2.2), "A* có trọng số (ε=1,5)": ("c", 1.2), "A* phân cấp (k=8)": ("y", 1.2),
              "Ưu tiên đường OSM": ("lime", 1.2)}
    for name, (col, lw) in styles.items():
        fn = {"A*": lambda: astar(cost, s, g), "A* có trọng số (ε=1,5)": lambda: weighted_astar(cost, s, g, 1.5),
              "A* phân cấp (k=8)": lambda: hierarchical_astar(cost, s, g, 8),
              "Ưu tiên đường OSM": lambda: astar(make_cost(L, META[region], region, roads_weight=0.5)[0], s, g)}
        if name == "Ưu tiên đường OSM" and region != "vn":
            continue
        p, _, _ = fn[name]()
        ax.plot([q[1] for q in p], [q[0] for q in p], color=col, lw=lw, label=name)
    ln = line_cells(s, g)
    ax.plot([q[1] for q in ln], [q[0] for q in ln], "w--", lw=1, label="Tuyến thẳng")
    ax.plot(s[1], s[0], "go", ms=8); ax.plot(g[1], g[0], "ro", ms=8)
    ax.legend(loc="lower left", fontsize=8); ax.set_axis_off()
    ax.set_title(f"{NAMES[region]}: các tuyến cho cặp điểm 1")
    fig.savefig(FIG / f"{region}_routes.png", bbox_inches="tight"); plt.close(fig)
    # 2. mặt cắt độ cao
    p, _, _ = astar(cost, s, g)
    m = path_metrics(p, L["dem"], L["slope"], cost, float(META[region]["cell_size_m"]))
    lm = path_metrics(ln, L["dem"], L["slope"], cost, float(META[region]["cell_size_m"]))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(m["profile_dist_m"] / 1000, m["profile_z_m"], "m", label="A*")
    ax.plot(lm["profile_dist_m"] / 1000, lm["profile_z_m"], "k--", lw=1, label="Tuyến thẳng")
    ax.set_xlabel("Khoảng cách dọc tuyến (km)"); ax.set_ylabel("Độ cao (m)"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(FIG / f"{region}_profile.png", bbox_inches="tight"); plt.close(fig)
    # 3. số nút và thời gian chạy theo phương pháp
    sub = A[(A["vùng"] == region) & (A["phương pháp"] != "Tuyến thẳng")]
    g_ = sub.groupby("phương pháp")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    order = [k for k in ["Dijkstra", "A*", "A* có trọng số (ε=1,5)", "A* phân cấp (k=8)", "Ưu tiên đường OSM"] if k in g_.groups]
    ax[0].bar(range(len(order)), [g_.get_group(k)["nút duyệt"].mean() / 1e3 for k in order],
              yerr=[g_.get_group(k)["nút duyệt"].std() / 1e3 for k in order], color="tab:blue")
    ax[0].set_ylabel("Nút đã duyệt (nghìn)")
    ax[1].bar(range(len(order)), [g_.get_group(k)["thời gian chạy (s)"].mean() for k in order],
              yerr=[g_.get_group(k)["thời gian chạy (s)"].std() for k in order], color="tab:orange")
    ax[1].set_ylabel("Thời gian chạy (s)")
    for a in ax:
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=25, ha="right", fontsize=8); a.grid(axis="y", alpha=.3)
    fig.suptitle(NAMES[region]); fig.savefig(FIG / f"{region}_nodes_time.png", bbox_inches="tight"); plt.close(fig)


def layer_figures(Lv, Lm):
    # sạt lở các bước, phóng to quanh Làng Nủ
    r0, c0, w = 600, 760, 140
    win = (slice(r0, r0 + w), slice(c0, c0 + w))
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, (k, t, cm, vr) in zip(ax, [("ndvi_before", "NDVI 27/4/2024", "RdYlGn", (0, 0.9)),
                                       ("ndvi_after", "NDVI 12/5/2025", "RdYlGn", (0, 0.9)),
                                       ("dndvi", "ΔNDVI = trước − sau", "coolwarm", (-0.5, 0.5)),
                                       ("hazard_landslide", "Lớp sạt lở cuối", "Reds", (0, 1))]):
        im = a.imshow(Lv[k][win], cmap=cm, vmin=vr[0], vmax=vr[1]); a.set_title(t); a.set_axis_off()
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Phát hiện sạt lở quanh Làng Nủ (cửa sổ 4,2 × 4,2 km)")
    fig.savefig(FIG / "vn_landslide_steps.png", bbox_inches="tight"); plt.close(fig)
    # ngập theo mực nước
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
    for a, h in zip(ax, (1, 2, 4, 8)):
        a.imshow(Lv["hillshade"], cmap="gray")
        ov = np.zeros(Lv["dem"].shape + (4,))
        ov[Lv[f"hazard_flood_h{h}"] > 0.5] = [0.2, 0.5, 1, 0.8]; ov[Lv["river"] > 0.5] = [0, 0, 0.5, 1]
        ov[Lv["s1_flood_20240915"] > 0.5] = [1, 0, 0, 1]
        a.imshow(ov); a.set_title(f"h = {h} m ({Lv[f'hazard_flood_h{h}'].sum() * 0.0009:.1f} km²)"); a.set_axis_off()
    fig.suptitle("Lớp ngập theo mực nước (xanh) và ngập phát hiện từ Sentinel-1 ngày 15/9/2024 (đỏ)")
    fig.savefig(FIG / "vn_flood_levels.png", bbox_inches="tight"); plt.close(fig)
    # địa hình hai vùng
    for reg, L in (("vn", Lv), ("mars", Lm)):
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
        for a, (k, t, cm) in zip(ax, [("dem", "Độ cao (m)", "terrain"), ("hillshade", "Hillshade", "gray"),
                                       ("slope", "Độ dốc (°)", "magma"), ("roughness", "Độ gồ ghề 3×3 (m)", "viridis")]):
            v = L[k]
            im = a.imshow(v, cmap=cm, vmax=np.nanpercentile(v, 99) if k == "roughness" else None)
            a.set_title(t); a.set_axis_off(); fig.colorbar(im, ax=a, fraction=0.046)
        fig.suptitle(NAMES[reg]); fig.savefig(FIG / f"{reg}_terrain.png", bbox_inches="tight"); plt.close(fig)
    # hố Sao Hỏa
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    im = ax[0].imshow(Lm["tophat"], cmap="viridis", vmax=30); ax[0].set_title("Top-hat đen, đĩa r=12 ô (m)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].imshow(Lm["hillshade"], cmap="gray")
    ov = np.zeros(Lm["dem"].shape + (4,)); ov[Lm["hazard_crater"] > 0.5] = [1, 0.2, 0.2, 0.9]
    ax[1].imshow(ov); n = hz.label(Lm["hazard_crater"] > 0.5)[1]; ax[1].set_title(f"Hố được giữ lại: {n}")
    for a in ax: a.set_axis_off()
    fig.savefig(FIG / "mars_craters.png", bbox_inches="tight"); plt.close(fig)


def param_figure(B):
    for reg in B["vùng"].unique():
        sub = B[(B["vùng"] == reg) & (B["tìm được"] == 1)]
        params = list(dict.fromkeys(sub["tham số"]))
        fig, axes = plt.subplots(1, len(params), figsize=(3.1 * len(params), 3))
        for a, p in zip(np.atleast_1d(axes), params):
            d = sub[sub["tham số"] == p].groupby("giá trị")
            x = list(d.groups)
            a.errorbar(range(len(x)), d["chi phí (giờ)"].mean(), yerr=d["chi phí (giờ)"].std(), fmt="o-", capsize=3, label="chi phí (giờ)")
            a2 = a.twinx()
            a2.plot(range(len(x)), d["dốc lớn nhất (°)"].mean(), "s--", color="tab:red", label="dốc lớn nhất (°)")
            a.set_xticks(range(len(x))); a.set_xticklabels([str(v) for v in x]); a.set_title(p, fontsize=9)
            a.set_ylabel("Chi phí (giờ)", fontsize=8); a2.set_ylabel("Dốc lớn nhất (°)", fontsize=8, color="tab:red")
        fig.suptitle(f"{NAMES[reg]}: ảnh hưởng tham số (trung bình ± độ lệch chuẩn)")
        fig.tight_layout(); fig.savefig(FIG / f"{reg}_params.png", bbox_inches="tight"); plt.close(fig)


META = {}


def cached(name, fn, resume, tag=""):
    """Lưu kết quả từng phần ra CSV để chạy tiếp được khi bị ngắt (--resume)."""
    path = TAB / f"_part_{name}{tag}.csv"
    if resume and path.exists():
        return pd.read_csv(path)
    df = pd.DataFrame(fn())
    df.to_csv(path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pairs", type=int, default=20)
    a = ap.parse_args()
    n_pairs = 4 if a.quick else a.pairs
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    allA, allB, allC, ver = [], [], [], []
    regions = {}
    t_all = time.perf_counter()
    for k, region in enumerate(("vn", "mars")):
        rng = np.random.default_rng(SEED + k)
        L, meta = load_region(region); META[region] = meta; regions[region] = L
        cost, _ = make_cost(L, meta, region)
        pf = TAB / f"{region}_pairs_{n_pairs}.json"
        if a.resume and pf.exists():
            pairs = [tuple(map(tuple, p)) for p in json.load(open(pf))]
        else:
            pairs = sample_pairs(cost, n_pairs, rng)
            json.dump(pairs, open(pf, "w"))
        print(f"[{region}] kiểm chứng", flush=True)
        ver.append(cached(f"{region}_ver_{n_pairs}", lambda: [{"vùng": region, **r} for r in verification(L, meta)], a.resume))
        print(f"[{region}] thí nghiệm (a)", flush=True)
        A = cached(f"{region}_a_{n_pairs}", lambda: exp_a(region, L, meta, cost, pairs), a.resume); allA.append(A)
        print(f"[{region}] thí nghiệm (b)", flush=True)
        allB.append(cached(f"{region}_b_{n_pairs}", lambda: exp_b(region, L, meta, pairs[: max(4, len(pairs) // 2)]), a.resume))
        print(f"[{region}] thí nghiệm (c)", flush=True)
        allC.append(cached(f"{region}_c_{n_pairs}", lambda: exp_c(region, L, meta, cost, pairs, np.random.default_rng(SEED)), a.resume))
        done = TAB / f"_fig_{region}_{n_pairs}.done"
        if not (a.resume and done.exists()):
            figures(region, L, cost, pairs, A)
            done.write_text("ok")
    A, B, C, V = pd.concat(allA), pd.concat(allB), pd.concat(allC), pd.concat(ver)
    A.to_csv(TAB / "exp_a_raw.csv", index=False); B.to_csv(TAB / "exp_b_raw.csv", index=False)
    C.to_csv(TAB / "exp_c.csv", index=False); V.to_csv(TAB / "verification.csv", index=False)
    layer_figures(regions["vn"], regions["mars"]); param_figure(B); worked_examples(regions["vn"])

    # ---- tổng hợp trung bình ± độ lệch chuẩn
    cols = ["chi phí (giờ)", "tỉ số chi phí", "độ dài (km)", "dốc lớn nhất (°)", "tổng leo (m)", "nút duyệt", "thời gian chạy (s)"]
    SA = A.groupby(["vùng", "phương pháp"], sort=False)[cols].agg(pm).reset_index()
    SA.to_csv(TAB / "exp_a_summary.csv", index=False)
    cB = ["chi phí (giờ)", "độ dài (km)", "dốc lớn nhất (°)", "nút duyệt", "thời gian chạy (s)"]
    SB = B.groupby(["vùng", "tham số", "giá trị"], sort=False).agg(
        **{c: (c, pm) for c in cB}, **{"tìm được": ("tìm được", lambda x: f"{int(x.sum())}/{len(x)}")}).reset_index()
    SB.to_csv(TAB / "exp_b_summary.csv", index=False)
    mcp_dev = A["lệch so với MCP"].max()
    straight_blocked = A[A["phương pháp"] == "Tuyến thẳng"].groupby("vùng")["ô cấm trên tuyến"].apply(lambda x: f"{(x > 0).sum()}/{len(x)}")
    env = (f"{platform.platform()}; Python {platform.python_version()}; NumPy {np.__version__}; "
           f"pandas {pd.__version__}; CPU: {platform.processor() or platform.machine()}")
    md = ["# Kết quả thí nghiệm (tự sinh bởi experiments/run_experiments.py)\n",
          f"Môi trường chạy: {env}. Số cặp điểm mỗi vùng: {n_pairs} (seed {SEED}). Tổng thời gian: {time.perf_counter() - t_all:.0f} s.\n",
          "## Bảng kiểm chứng (Chương 3.5)\n", V.to_markdown(index=False),
          f"\n\nĐộ lệch tổng chi phí của Dijkstra/A* tự cài so với skimage.graph.MCP_Geometric trên mọi cặp: **{mcp_dev:.2e}**.\n",
          "## (a) So sánh phương pháp\n", SA.to_markdown(index=False),
          f"\n\nTuyến thẳng đi qua ô cấm: {dict(straight_blocked)} cặp.\n",
          "## (b) Khảo sát tham số\n", SB.to_markdown(index=False),
          "\n## (c) Điều kiện bất lợi\n", C.to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Xong sau {time.perf_counter() - t_all:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
