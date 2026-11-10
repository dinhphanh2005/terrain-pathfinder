import math

import numpy as np
import pytest
from skimage.graph import MCP_Geometric

from src.cost import build_cost, tobler_speed_kmh
from src.pathfinding import (astar, dijkstra, hierarchical_astar, line_cells, path_cost,
                             rest_stops, weighted_astar)
from src.synthetic import make_cost


def mcp_cost(cost, s, g):
    m = MCP_Geometric(np.where(np.isfinite(cost), cost, np.inf), fully_connected=True)
    cum, _ = m.find_costs([s], [g])
    return cum[g]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_dijkstra_and_astar_match_mcp(seed):
    cost = make_cost(60, 70, seed=seed)
    s, g = (0, 0), (59, 69)
    ref = mcp_cost(cost, s, g)
    p1, c1, e1 = dijkstra(cost, s, g)
    p2, c2, e2 = astar(cost, s, g)
    assert c1 == pytest.approx(ref, abs=1e-6)
    assert c2 == pytest.approx(ref, abs=1e-6)
    assert path_cost(cost, p2) == pytest.approx(c2, abs=1e-6)
    assert e2 <= e1                          # A* không duyệt nhiều hơn Dijkstra
    assert all(np.isfinite(cost[p]) for p in p2)
    for (a, b), (c, d) in zip(p2, p2[1:]):   # các ô liên tiếp là láng giềng 8 hướng
        assert max(abs(a - c), abs(b - d)) == 1


def test_weighted_bound():
    cost = make_cost(80, 80, seed=7)
    _, opt, _ = astar(cost, (0, 0), (79, 79))
    for eps in (1.2, 2.0, 3.0):
        _, c, _ = weighted_astar(cost, (0, 0), (79, 79), eps=eps)
        assert opt - 1e-9 <= c <= eps * opt + 1e-9


def test_hierarchical_valid_and_near_optimal():
    cost = make_cost(120, 120, seed=3, n_blocks=8)
    _, opt, _ = astar(cost, (0, 0), (119, 119))
    p, c, _ = hierarchical_astar(cost, (0, 0), (119, 119), k=8)
    assert p and all(np.isfinite(cost[q]) for q in p)
    assert opt - 1e-9 <= c <= 1.25 * opt


def test_no_path_when_isolated():
    cost = np.ones((10, 10), np.float32)
    cost[:, 5] = np.inf
    p, c, _ = astar(cost, (0, 0), (9, 9))
    assert p == [] and math.isinf(c)


def test_tobler_and_cost():
    assert tobler_speed_kmh(np.array([0.0]))[0] == pytest.approx(6.0)
    slope = np.array([[0.0, 10.0], [45.0, np.nan]])
    c = build_cost(slope, np.zeros((2, 2)), None, 30.0)
    assert c[0, 0] == pytest.approx(30 / (6 / 3.6))
    assert c[0, 1] > c[0, 0]
    assert np.isinf(c[1, 0]) and np.isinf(c[1, 1])


def test_line_and_stops():
    assert line_cells((0, 0), (3, 3)) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    cost = np.full((1, 200), 60.0)
    path = [(0, c) for c in range(200)]
    st = rest_stops(path, cost, np.zeros((1, 200)), every_s=3600, window=3)
    assert len(st) == 3
