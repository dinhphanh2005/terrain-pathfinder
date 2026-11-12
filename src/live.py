"""Lớp nguy hiểm thay đổi theo thời gian + điểm ngập quan sát (vd từ camera, Đề 21) + tìm lại đường.

Hai ý tưởng phát triển tiếp của Đề 26:
1. "Cập nhật lớp nguy hiểm theo thời gian": tại mỗi thời điểm t, lớp nguy hiểm động gồm
   - ngập do sông dâng h(t): h nội suy tuyến tính giữa các mốc (thời gian, mực nước) của kịch bản,
     rồi loang từ sông như hazards.flood_by_level;
   - các quan sát điểm còn hiệu lực ở t (thời điểm báo <= t < thời điểm báo + ttl).
   Khi lớp đổi, kiểm tra phần tuyến chưa đi; nếu chạm ô cấm mới thì tìm lại đường (A*) từ vị trí hiện tại.
2. "Tích hợp điểm ngập từ camera": camera (Đề 21) cho một điểm (kinh độ, vĩ độ) và độ sâu nước d.
   Mặt nước tại đó = z(p) + d; vùng ngập quanh camera = các ô liền kề có độ cao <= mặt nước
   (loang vùng / region growing), giới hạn trong bán kính R để không lan cả thung lũng.

Định dạng quan sát (CSV): time, kind, lat, lon, value, ttl_h, source, note
  kind = camera_flood (value = độ sâu nước, m) | landslide | blocked (value = bán kính, m)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rasterio.warp import transform as warp_transform

from src import hazards as hz
from src.layers import make_cost
from src.pathfinding import astar, path_cost

KINDS = ("camera_flood", "landslide", "blocked")
NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------- tọa độ
def lonlat_to_cell(lon, lat, meta) -> tuple[np.ndarray, np.ndarray]:
    """(kinh độ, vĩ độ) WGS84 -> (hàng, cột) trên lưới của vùng. Ngoài lưới vẫn trả về (kiểm tra sau)."""
    xs, ys = warp_transform("EPSG:4326", meta["crs"], np.atleast_1d(lon).tolist(), np.atleast_1d(lat).tolist())
    a, _, c, _, e, f = meta["transform"][:6]
    cols = np.floor((np.asarray(xs) - c) / a).astype(int)
    rows = np.floor((np.asarray(ys) - f) / e).astype(int)
    return rows, cols


def cell_to_lonlat(rows, cols, meta) -> tuple[np.ndarray, np.ndarray]:
    """(hàng, cột) -> (kinh độ, vĩ độ) của tâm ô."""
    a, _, c, _, e, f = meta["transform"][:6]
    xs = c + (np.atleast_1d(cols) + 0.5) * a
    ys = f + (np.atleast_1d(rows) + 0.5) * e
    lon, lat = warp_transform(meta["crs"], "EPSG:4326", xs.tolist(), ys.tolist())
    return np.asarray(lon), np.asarray(lat)


# ---------------------------------------------------------------- dữ liệu kịch bản
def load_observations(src, meta) -> pd.DataFrame:
    """Đọc quan sát từ CSV (đường dẫn, file tải lên hoặc DataFrame); thêm cột row, col, inside, end."""
    df = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    need = {"time", "kind", "lat", "lon", "value"}
    if not need <= set(df.columns):
        raise ValueError(f"thiếu cột: {sorted(need - set(df.columns))}")
    df["time"] = pd.to_datetime(df["time"])
    if "ttl_h" not in df:
        df["ttl_h"] = 24.0
    df["ttl_h"] = df["ttl_h"].fillna(24.0).astype(float)
    bad = ~df["kind"].isin(KINDS)
    if bad.any():
        raise ValueError(f"kind không hợp lệ: {df.loc[bad, 'kind'].unique().tolist()}")
    r, c = lonlat_to_cell(df["lon"].to_numpy(), df["lat"].to_numpy(), meta)
    df["row"], df["col"] = r, c
    df["inside"] = (r >= 0) & (r < meta["rows"]) & (c >= 0) & (c < meta["cols"])
    df["end"] = df["time"] + pd.to_timedelta(df["ttl_h"], unit="h")
    return df.sort_values("time").reset_index(drop=True)


def level_at(schedule, t) -> float:
    """Mực nước dâng h(t) (m): nội suy tuyến tính giữa các mốc (time, level_m); ngoài khoảng thì giữ đầu mút."""
    if schedule is None or len(schedule) == 0:
        return 0.0
    s = pd.DataFrame(schedule)
    # ép về nano giây: pandas 3 mặc định micro giây, còn Timestamp.value luôn là nano giây
    s["time"] = pd.to_datetime(s["time"]).astype("datetime64[ns]"); s = s.sort_values("time")
    x = s["time"].astype("int64").to_numpy(dtype=np.float64)
    return float(np.interp(pd.Timestamp(t).value, x, s["level_m"].to_numpy(np.float64)))


def active_at(obs: pd.DataFrame, t) -> pd.DataFrame:
    """Các quan sát còn hiệu lực tại t: time <= t < end, và nằm trong lưới."""
    t = pd.Timestamp(t)
    return obs[(obs["time"] <= t) & (t < obs["end"]) & obs["inside"]]


# ---------------------------------------------------------------- lớp từ quan sát
def flood_from_point(dem: np.ndarray, cell, depth_m: float, max_radius_cells: int) -> np.ndarray:
    """Vùng ngập quanh một camera: loang (BFS) từ ô camera sang ô kề có độ cao <= z(camera) + depth_m,
    chỉ trong bán kính max_radius_cells. Trả mặt nạ bool (luôn chứa ô camera)."""
    rows, cols = dem.shape
    r0, c0 = int(cell[0]), int(cell[1])
    surface = float(dem[r0, c0]) + float(depth_m)
    rad2 = float(max_radius_cells) ** 2
    out = np.zeros((rows, cols), bool)
    out[r0, c0] = True
    q = deque([(r0, c0)])
    z = dem
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGH8:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and not out[nr, nc]
                    and (nr - r0) ** 2 + (nc - c0) ** 2 <= rad2 and z[nr, nc] <= surface):
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def disk_at(shape, cell, radius_cells: int) -> np.ndarray:
    """Mặt nạ hình tròn tâm `cell` (hàng, cột), bán kính radius_cells ô: (r−r0)² + (c−c0)² ≤ R².

    Dùng cho quan sát landslide/blocked (value = bán kính ảnh hưởng, m, đã đổi sang ô).
    """
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - cell[0]) ** 2 + (cc - cell[1]) ** 2 <= max(radius_cells, 0) ** 2


def observation_layers(dem, obs_active: pd.DataFrame, cell_size_m: float,
                       camera_radius_m: float = 600.0) -> dict[str, np.ndarray]:
    """Mặt nạ từ các quan sát đang hiệu lực: camera_flood, landslide, blocked."""
    out = {k: np.zeros(dem.shape, bool) for k in KINDS}
    rad_cam = int(round(camera_radius_m / cell_size_m))
    for o in obs_active.itertuples():
        p = (int(o.row), int(o.col))
        if o.kind == "camera_flood":
            out[o.kind] |= flood_from_point(dem, p, float(o.value), rad_cam)
        else:
            out[o.kind] |= disk_at(dem.shape, p, int(round(float(o.value) / cell_size_m)))
    return out


# ---------------------------------------------------------------- lớp nguy hiểm tại thời điểm t
class RiverFloodCache:
    """Lớp ngập do sông theo h. Phần tốn thời gian (river_reference, ~1 s) tính một lần;
    mỗi mức h sau đó chỉ ~0,05 s. Làm tròn h tới 0,25 m và nhớ kết quả."""

    def __init__(self, dem, river, step=0.25):
        self.dem, self.river, self.step, self.cache = dem, river, step, {}
        self.ref = hz.river_reference(dem, river)

    def __call__(self, h: float) -> np.ndarray:
        key = round(max(h, 0.0) / self.step) * self.step
        if key <= 0:
            return np.zeros(self.dem.shape, bool)
        if key not in self.cache:
            self.cache[key] = hz.flood_by_level(self.dem, self.river, key, self.ref)
        return self.cache[key]


@dataclass
class Snapshot:
    time: pd.Timestamp
    level_m: float
    hazard: np.ndarray                 # nguy hiểm động (chưa có vùng đệm)
    parts: dict
    active: pd.DataFrame
    cost: np.ndarray
    forbidden: np.ndarray


def snapshot(layers, meta, t, schedule, obs, river_flood: RiverFloodCache, *,
             buffer_m=None, weights=None, use_landslide=True, camera_radius_m=600.0) -> Snapshot:
    """Lớp nguy hiểm, vùng cấm và bản đồ chi phí tại thời điểm t (Việt Nam)."""
    h = level_at(schedule, t)
    act = active_at(obs, t) if obs is not None else pd.DataFrame()
    parts = {"river_flood": river_flood(h)}
    if len(act):
        parts.update(observation_layers(layers["dem"], act, float(meta["cell_size_m"]), camera_radius_m))
    dyn = np.zeros(layers["dem"].shape, bool)
    for m in parts.values():
        dyn |= m
    cost, forbidden = make_cost(layers, meta, "vn", flood_m=None, buffer_m=buffer_m, weights=weights,
                                use_landslide=use_landslide, extra_hazard=dyn)
    return Snapshot(pd.Timestamp(t), h, dyn, parts, act, cost, forbidden)


# ---------------------------------------------------------------- đi bộ có cập nhật
def cumulative_time(cost, path) -> np.ndarray:
    """Thời gian (giây) tới từng ô trên tuyến, theo cùng định nghĩa cạnh của path_cost."""
    t = np.zeros(len(path))
    for i in range(1, len(path)):
        t[i] = t[i - 1] + path_cost(cost, path[i - 1:i + 1])
    return t


def blocked_ahead(path, idx: int, forbidden: np.ndarray) -> int | None:
    """Vị trí đầu tiên (>= idx) trên tuyến rơi vào ô cấm, hoặc None."""
    for j in range(idx, len(path)):
        if forbidden[path[j]]:
            return j
    return None


@dataclass
class Event:
    time: pd.Timestamp
    cell: tuple
    action: str            # "xuất phát" | "tìm lại đường" | "giữ tuyến" | "kẹt" | "đến nơi"
    level_m: float
    n_active: int
    remaining_h: float = float("nan")
    expanded: int = 0
    compute_s: float = 0.0
    path: list = field(default_factory=list)


def simulate_walk(layers, meta, start, goal, t0, schedule, obs, *, update_min=30.0,
                  replan=True, max_hours=48.0, **kw) -> list[Event]:
    """Mô phỏng người đi bộ theo tuyến; cứ update_min phút cập nhật lớp nguy hiểm một lần.

    replan=True: nếu phần tuyến còn lại chạm ô cấm mới -> A* lại từ vị trí hiện tại.
    replan=False: giữ nguyên tuyến ban đầu (để so sánh); tới sát ô cấm thì "kẹt".
    Tốc độ đi lấy từ chi phí địa hình (không tính vùng cấm): người đang đứng trong vùng vừa
    bị cấm vẫn đi ra được, còn việc có được đi tiếp hay không do vùng cấm tại thời điểm đó quyết định.
    """
    import time as _time
    from src.cost import build_cost
    rf = kw.pop("river_flood", None) or RiverFloodCache(layers["dem"], layers["river"] > 0.5)
    walk_cost = build_cost(layers["slope"], layers["roughness"], None, float(meta["cell_size_m"]),
                           kw.get("weights"))
    t = pd.Timestamp(t0)
    snap = snapshot(layers, meta, t, schedule, obs, rf, **kw)
    t1 = _time.perf_counter(); path, total, n = astar(snap.cost, start, goal); dt = _time.perf_counter() - t1
    ev = [Event(t, start, "xuất phát" if path else "kẹt", snap.level_m, len(snap.active),
                total / 3600 if path else float("nan"), n, dt, list(path))]
    if not path:
        return ev
    path = list(path)
    pos = 0
    cum = cumulative_time(walk_cost, path)
    while t - pd.Timestamp(t0) < pd.Timedelta(hours=max_hours):
        target = cum[pos] + update_min * 60.0
        new_pos = max(int(np.searchsorted(cum, target, side="right") - 1), pos)
        new_pos = min(new_pos, len(path) - 1)
        # Không bước vào ô đang cấm (theo lần cập nhật gần nhất): dừng ngay trước ô đó.
        jb = blocked_ahead(path[:new_pos + 1], pos + 1, snap.forbidden)
        stopped = jb is not None
        if stopped:
            new_pos = jb - 1
        if not stopped and new_pos >= len(path) - 1:
            arrive = t + pd.Timedelta(seconds=float(cum[-1] - cum[pos]))
            ev.append(Event(arrive, path[-1], "đến nơi", level_at(schedule, arrive), 0, 0.0))
            return ev
        t = t + pd.Timedelta(seconds=float(cum[new_pos] - cum[pos]))
        pos = new_pos
        if stopped and not replan:
            ev.append(Event(t, path[pos], "kẹt (đường bị chặn)", snap.level_m, len(snap.active)))
            return ev
        snap = snapshot(layers, meta, t, schedule, obs, rf, **kw)
        here = path[pos]
        j = blocked_ahead(path, pos + 1, snap.forbidden)
        if j is None:
            ev.append(Event(t, here, "giữ tuyến", snap.level_m, len(snap.active),
                            float((cum[-1] - cum[pos]) / 3600)))
            continue
        if not replan:
            if j == pos + 1:
                ev.append(Event(t, here, "kẹt", snap.level_m, len(snap.active)))
                return ev
            ev.append(Event(t, here, "giữ tuyến (phía trước bị chặn)", snap.level_m, len(snap.active),
                            float((cum[-1] - cum[pos]) / 3600)))
            continue
        c2 = snap.cost.copy()
        if not np.isfinite(c2[here]):
            c2[here] = walk_cost[here]              # đang đứng trong vùng vừa bị cấm: cho phép đi ra
        t1 = _time.perf_counter(); newp, tot, n = astar(c2, here, goal); dt = _time.perf_counter() - t1
        if not newp:
            ev.append(Event(t, here, "kẹt", snap.level_m, len(snap.active), float("nan"), n, dt))
            return ev
        path, pos = list(newp), 0
        cum = cumulative_time(walk_cost, path)
        ev.append(Event(t, here, "tìm lại đường", snap.level_m, len(snap.active), tot / 3600, n, dt, list(newp)))
    return ev


def events_table(events: list[Event]) -> pd.DataFrame:
    """Bảng nhật ký mô phỏng: mỗi Event một dòng (thời điểm, sự kiện, ô, mực nước,
    số quan sát hiệu lực, thời gian còn lại, số nút duyệt và thời gian tính nếu có tìm đường)."""
    return pd.DataFrame([{
        "thời điểm": e.time.strftime("%d/%m %H:%M"), "sự kiện": e.action, "ô": e.cell,
        "mực nước h (m)": round(e.level_m, 2), "quan sát hiệu lực": e.n_active,
        "còn lại (giờ)": None if e.remaining_h != e.remaining_h else round(e.remaining_h, 2),
        "nút duyệt": e.expanded or None, "tính (s)": round(e.compute_s, 2) if e.compute_s else None,
    } for e in events])


def walked_cells(events: list[Event]) -> list[tuple]:
    """Dãy ô thực sự đã đi (ghép các đoạn giữa các lần tìm lại đường)."""
    out, cur = [], None
    for e in events:
        if e.path:
            if cur is not None:
                k = cur.index(e.cell) if e.cell in cur else len(cur) - 1
                out.extend(cur[:k])
            cur = e.path
    if cur is not None:
        last = events[-1].cell
        k = cur.index(last) + 1 if last in cur else len(cur)
        out.extend(cur[:k])
    return out
