"""Tìm đường trên lưới raster 8 hướng: Dijkstra, A*, A* có trọng số, A* phân cấp; tiện ích đánh giá tuyến.

Đồ thị: mỗi ô là một đỉnh, nối 8 láng giềng. Chi phí cạnh u->v = (C[u] + C[v]) / 2 × độ dài cạnh
(1 cho cạnh thẳng, √2 cho cạnh chéo, tính theo số ô) -> nửa ô đầu và nửa ô cuối.
Định nghĩa này giống skimage.graph.MCP_Geometric nên tổng chi phí đối chiếu được.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

SQRT2 = math.sqrt(2.0)
NEIGH8 = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0),
          (0, 1, 1.0), (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2)]


def octile(r0, c0, r1, c1) -> float:
    """Khoảng cách octile (số ô): đi chéo hết phần chung rồi đi thẳng phần dư."""
    dr, dc = abs(r0 - r1), abs(c0 - c1)
    return (SQRT2 - 1.0) * min(dr, dc) + max(dr, dc)


def _search(cost: np.ndarray, start, goal, eps: float, allowed: np.ndarray | None = None):
    """Lõi chung. eps = 0: Dijkstra; eps = 1: A*; eps > 1: A* có trọng số.

    f(n) = g(n) + eps·h(n), h(n) = octile(n, goal) × C_min  (C_min: chi phí ô nhỏ nhất).
    h chấp nhận được: mọi đường từ n tới đích đi ít nhất octile ô, mỗi ô tốn ít nhất C_min
    -> h không bao giờ vượt chi phí thật -> A* (eps=1) cho đường tối ưu.
    h còn nhất quán (h(u) <= w(u,v) + h(v)) nên mỗi đỉnh chỉ cần lấy ra khỏi hàng đợi một lần.
    """
    rows, cols = cost.shape
    c = np.asarray(cost, np.float64)
    if allowed is not None:
        c = np.where(allowed, c, np.inf)
    sr, sc = start
    gr, gc = goal
    if not (np.isfinite(c[sr, sc]) and np.isfinite(c[gr, gc])):
        return [], math.inf, 0
    cmin = float(np.min(c[np.isfinite(c)]))
    cl = c.ravel().tolist()                       # list Python truy cập nhanh hơn mảng NumPy từng phần tử
    n = rows * cols
    g = [math.inf] * n
    parent = [-1] * n
    closed = bytearray(n)
    s, t = sr * cols + sc, gr * cols + gc
    g[s] = 0.0
    heap = [(eps * octile(sr, sc, gr, gc) * cmin, 0.0, s)]
    expanded = 0
    while heap:
        f, gu, u = heapq.heappop(heap)
        if closed[u]:
            continue                               # bản ghi cũ (đã tìm được g tốt hơn)
        closed[u] = 1
        expanded += 1
        if u == t:
            break
        ur, uc = divmod(u, cols)
        cu = cl[u]
        for dr, dc, length in NEIGH8:
            vr, vc = ur + dr, uc + dc
            if 0 <= vr < rows and 0 <= vc < cols:
                v = vr * cols + vc
                if closed[v]:
                    continue
                cv = cl[v]
                if cv == math.inf:
                    continue
                ng = gu + 0.5 * (cu + cv) * length
                if ng < g[v]:
                    g[v] = ng
                    parent[v] = u
                    h = eps * octile(vr, vc, gr, gc) * cmin if eps else 0.0
                    heapq.heappush(heap, (ng + h, ng, v))
    if g[t] == math.inf:
        return [], math.inf, expanded
    path = []
    u = t
    while u != -1:
        path.append(divmod(u, cols))
        u = parent[u]
    return path[::-1], g[t], expanded


def dijkstra(cost, start, goal, allowed=None):
    """Dijkstra (1959): như A* với h = 0, mở rộng đều mọi hướng."""
    return _search(cost, start, goal, 0.0, allowed)


def astar(cost, start, goal, allowed=None):
    """A* (Hart, Nilsson, Raphael 1968) với heuristic octile × C_min."""
    return _search(cost, start, goal, 1.0, allowed)


def weighted_astar(cost, start, goal, eps=1.5, allowed=None):
    """A* có trọng số: f = g + ε·h, ε > 1. Nhanh hơn, chi phí <= ε × tối ưu."""
    return _search(cost, start, goal, float(eps), allowed)


def find_path(cost, start, goal, method="astar", **kw):
    """Hàm chung theo docs/interface.md: trả (path, total_cost, n_expanded)."""
    fn = {"dijkstra": dijkstra, "astar": astar, "weighted": weighted_astar,
          "hierarchical": hierarchical_astar}[method]
    return fn(cost, start, goal, **kw)


# ---------------------------------------------------------------- A* phân cấp
def coarsen_cost(cost: np.ndarray, k: int) -> np.ndarray:
    """Thu nhỏ lưới chi phí k lần: khối k×k -> trung bình các ô đi được × k; khối > 50% ô cấm -> cấm."""
    rows, cols = cost.shape
    R, C = rows // k, cols // k
    blk = np.asarray(cost[:R * k, :C * k], np.float64).reshape(R, k, C, k).swapaxes(1, 2).reshape(R, C, k * k)
    fin = np.isfinite(blk)
    frac = fin.mean(-1)
    mean = np.where(fin, blk, 0).sum(-1) / np.maximum(fin.sum(-1), 1)
    out = mean * k
    out[frac < 0.5] = np.inf
    return out


def hierarchical_astar(cost, start, goal, k=8, corridor=2):
    """A* phân cấp (hay dùng trong game):
    1. Tìm đường thô trên lưới thu nhỏ k lần (ít đỉnh hơn k² lần).
    2. Lấy hành lang quanh đường thô (nới thêm `corridor` ô thô), phóng về lưới gốc.
    3. Chạy A* trên lưới gốc nhưng chỉ trong hành lang.
    Không đảm bảo tối ưu (đường tốt nhất có thể nằm ngoài hành lang); nếu thất bại thì chạy A* đầy đủ.
    n_expanded = tổng số nút đã duyệt ở cả hai mức.
    """
    rows, cols = cost.shape
    cc = coarsen_cost(cost, k)
    R, C = cc.shape
    cs = (min(start[0] // k, R - 1), min(start[1] // k, C - 1))
    cg = (min(goal[0] // k, R - 1), min(goal[1] // k, C - 1))
    cc[cs], cc[cg] = (cc[cs] if np.isfinite(cc[cs]) else k * 1.0), (cc[cg] if np.isfinite(cc[cg]) else k * 1.0)
    cpath, _, e1 = astar(cc, cs, cg)
    if not cpath:
        p, tot, e2 = astar(cost, start, goal)
        return p, tot, e1 + e2
    mask_c = np.zeros((R, C), bool)
    for r, c in cpath:
        mask_c[max(0, r - corridor):r + corridor + 1, max(0, c - corridor):c + corridor + 1] = True
    allowed = np.zeros((rows, cols), bool)
    allowed[:R * k, :C * k] = np.repeat(np.repeat(mask_c, k, 0), k, 1)
    allowed[start] = allowed[goal] = True
    p, tot, e2 = astar(cost, start, goal, allowed=allowed)
    if not p:
        p, tot, e3 = astar(cost, start, goal)
        return p, tot, e1 + e2 + e3
    return p, tot, e1 + e2


# ---------------------------------------------------------------- tuyến so sánh
def line_cells(start, goal):
    """Các ô trên đoạn thẳng (Bresenham), dùng làm 'tuyến thẳng' để so sánh."""
    (r0, c0), (r1, c1) = start, goal
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr
    out = []
    r, c = r0, c0
    while True:
        out.append((r, c))
        if (r, c) == (r1, c1):
            return out
        e2 = 2 * err
        if e2 > -dr:
            err -= dr; c += sc
        if e2 < dc:
            err += dc; r += sr


def path_cost(cost, path) -> float:
    """Tổng chi phí của một dãy ô theo cùng định nghĩa cạnh. Đi qua ô cấm -> inf."""
    tot = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        a, b = float(cost[r0, c0]), float(cost[r1, c1])
        tot += 0.5 * (a + b) * (SQRT2 if (r0 != r1 and c0 != c1) else 1.0)
    return tot


# ---------------------------------------------------------------- đánh giá tuyến
def path_metrics(path, dem, slope_deg, cost, cell_size_m) -> dict:
    """Độ dài (m), độ dốc lớn nhất (độ), tổng leo (m), thời gian (giây) và mặt cắt độ cao."""
    if not path:
        return {"found": False}
    rr = np.array([p[0] for p in path]); cc = np.array([p[1] for p in path])
    step = np.hypot(np.diff(rr), np.diff(cc)) * cell_size_m
    dist = np.concatenate(([0.0], np.cumsum(step)))
    z = np.asarray(dem)[rr, cc].astype(np.float64)
    dz = np.diff(z)
    return {
        "found": True,
        "length_m": float(dist[-1]),
        "max_slope_deg": float(np.nanmax(np.asarray(slope_deg)[rr, cc])),
        "ascent_m": float(dz[dz > 0].sum()),
        "time_s": float(path_cost(cost, path)),
        "profile_dist_m": dist,
        "profile_z_m": z,
    }


def rest_stops(path, cost, slope_deg, every_s: float = 3600.0, window: int = 15):
    """Điểm dừng nghỉ: cứ mỗi `every_s` giây đi bộ, chọn ô bằng phẳng nhất trong ±window ô quanh mốc đó."""
    if len(path) < 2:
        return []
    t = np.zeros(len(path))
    for i in range(1, len(path)):
        t[i] = t[i - 1] + path_cost(cost, path[i - 1:i + 1])
    stops = []
    mark = every_s
    while mark < t[-1]:
        i = int(np.searchsorted(t, mark))
        lo, hi = max(0, i - window), min(len(path), i + window + 1)
        j = lo + int(np.argmin([slope_deg[path[k]] for k in range(lo, hi)]))
        stops.append({"index": j, "cell": path[j], "time_s": float(t[j])})
        mark += every_s
    return stops


def timed(fn, *a, **kw):
    """Gọi fn(*a, **kw) và đo thời gian chạy. Ra: (kết quả, số giây) theo time.perf_counter."""
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def snap_to_free(cost: np.ndarray, cell, max_radius: int = 30):
    """Ô đi được gần nhất với `cell` (điểm người dùng bấm có thể rơi vào ô cấm). None nếu không có."""
    r0, c0 = cell
    if np.isfinite(cost[r0, c0]):
        return (r0, c0)
    rows, cols = cost.shape
    for rad in range(1, max_radius + 1):
        r1, r2 = max(0, r0 - rad), min(rows, r0 + rad + 1)
        c1, c2 = max(0, c0 - rad), min(cols, c0 + rad + 1)
        win = cost[r1:r2, c1:c2]
        rr, cc = np.nonzero(np.isfinite(win))
        if len(rr):
            d = (rr + r1 - r0) ** 2 + (cc + c1 - c0) ** 2
            k = int(np.argmin(d))
            return (int(rr[k] + r1), int(cc[k] + c1))
    return None
