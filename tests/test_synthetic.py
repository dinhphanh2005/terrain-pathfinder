import numpy as np

from src.synthetic import make_cost, make_dem


def test_same_seed_same_grid():
    np.testing.assert_array_equal(make_dem(50, 60, seed=3), make_dem(50, 60, seed=3))
    np.testing.assert_array_equal(make_cost(50, 60, seed=3), make_cost(50, 60, seed=3))


def test_cost_positive_or_inf_and_corners_free():
    cost = make_cost(80, 80, seed=1)
    finite = cost[np.isfinite(cost)]
    assert (finite >= 1.0).all()
    assert np.isinf(cost).any()
    assert np.isfinite(cost[0, 0]) and np.isfinite(cost[-1, -1])


def test_dem_shape_and_dtype():
    dem = make_dem(30, 40)
    assert dem.shape == (30, 40) and dem.dtype == np.float32
