import numpy as np
import pandas as pd

from src import live
from src.pathfinding import astar

META = {"crs": "EPSG:32648", "transform": [30.0, 0.0, 427276.964549, 0.0, -30.0, 2468970.204538],
        "rows": 998, "cols": 1003, "cell_size_m": 30.0}


def test_lonlat_cell_round_trip():
    lon, lat = live.cell_to_lonlat([333, 666, 0], [707, 829, 0], META)
    r, c = live.lonlat_to_cell(lon, lat, META)
    assert r.tolist() == [333, 666, 0] and c.tolist() == [707, 829, 0]


def test_level_at_interpolates_and_clamps():
    sch = [{"time": "2024-09-10 00:00", "level_m": 0}, {"time": "2024-09-10 10:00", "level_m": 10}]
    assert live.level_at(sch, "2024-09-10 05:00") == 5.0
    assert live.level_at(sch, "2024-09-09") == 0.0 and live.level_at(sch, "2024-09-11") == 10.0
    assert live.level_at([], "2024-09-10") == 0.0


def test_active_window():
    df = pd.DataFrame({"time": ["2024-09-10 08:00"], "kind": ["landslide"], "lat": [22.2], "lon": [104.5],
                       "value": [100], "ttl_h": [2]})
    obs = live.load_observations(df, META)
    assert len(live.active_at(obs, "2024-09-10 07:59")) == 0
    assert len(live.active_at(obs, "2024-09-10 09:00")) == 1
    assert len(live.active_at(obs, "2024-09-10 10:00")) == 0   # hết hạn đúng lúc time + ttl


def test_flood_from_point_connected_and_radius():
    dem = np.full((21, 21), 10.0)
    dem[:, 10] = 50.0                     # bờ cao chắn giữa
    dem[10, 3] = 9.0                      # camera ở chỗ trũng bên trái
    m = live.flood_from_point(dem, (10, 3), 1.5, max_radius_cells=4)
    assert m[10, 3] and not m[:, 10:].any()                 # không vượt bờ cao
    rr, cc = np.nonzero(m)
    assert ((rr - 10) ** 2 + (cc - 3) ** 2 <= 16).all()     # trong bán kính
    assert not live.flood_from_point(dem, (10, 3), 0.5, 4)[10, 4]  # mặt nước 9,5 < 10: ô bên cạnh không ngập


def test_replan_avoids_new_block():
    cost = np.ones((30, 30), np.float32)
    p, _, _ = astar(cost, (15, 0), (15, 29))
    assert live.blocked_ahead(p, 0, np.zeros(cost.shape, bool)) is None
    block = np.zeros(cost.shape, bool); block[10:21, 15] = True
    assert live.blocked_ahead(p, 0, block) is not None
    c2 = cost.copy(); c2[block] = np.inf
    q, _, _ = astar(c2, (15, 0), (15, 29))
    assert q and live.blocked_ahead(q, 0, block) is None


def test_walked_cells_joins_segments():
    e1 = live.Event(pd.Timestamp("2024-01-01"), (0, 0), "xuất phát", 0, 0, path=[(0, 0), (0, 1), (0, 2), (0, 3)])
    e2 = live.Event(pd.Timestamp("2024-01-01 01:00"), (0, 2), "tìm lại đường", 0, 0, path=[(0, 2), (1, 3), (2, 3)])
    e3 = live.Event(pd.Timestamp("2024-01-01 02:00"), (2, 3), "đến nơi", 0, 0)
    assert live.walked_cells([e1, e2, e3]) == [(0, 0), (0, 1), (0, 2), (1, 3), (2, 3)]
