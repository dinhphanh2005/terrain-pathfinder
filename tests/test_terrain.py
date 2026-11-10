import numpy as np
import pytest
from scipy import ndimage

from src.synthetic import make_dem
from src.terrain import (SOBEL_X, box_sum, correlate3x3, gradient_sobel, hillshade,
                         integral_image, local_std, slope_aspect)


def test_correlate_matches_scipy():
    img = make_dem(40, 50, seed=2).astype(np.float64)
    k = np.random.default_rng(0).normal(size=(3, 3))
    ref = ndimage.correlate(img, k, mode="nearest")
    np.testing.assert_allclose(correlate3x3(img, k), ref, atol=1e-9)


def test_plane_slope_exact():
    # z = 0.1·x(Đông) + 0.2·y(Bắc); hàng tăng về Nam nên y = −hàng
    cell = 30.0
    r, c = np.mgrid[0:20, 0:20]
    dem = 0.1 * c * cell + 0.2 * (-r) * cell
    slope, aspect = slope_aspect(dem, cell)
    expect = np.degrees(np.arctan(np.hypot(0.1, 0.2)))
    np.testing.assert_allclose(slope[1:-1, 1:-1], expect, atol=1e-4)
    # dốc xuống về phía Tây-Nam: atan2(−0.1, −0.2)
    np.testing.assert_allclose(aspect[1:-1, 1:-1], np.degrees(np.arctan2(-0.1, -0.2)) % 360, atol=1e-3)


def test_gradient_vs_numpy_on_plane_and_smooth():
    cell = 20.0
    dem = make_dem(60, 60, seed=5, n_hills=3).astype(np.float64)
    dem = ndimage.gaussian_filter(dem, 3)  # mặt trơn: Sobel ≈ sai phân trung tâm
    gx, gy = gradient_sobel(dem, cell)
    dr, dc = np.gradient(dem, cell)
    inner = (slice(2, -2), slice(2, -2))
    assert np.abs(gx[inner] - dc[inner]).max() < 0.05 * np.abs(dc[inner]).max()
    assert np.abs(gy[inner] + dr[inner]).max() < 0.05 * np.abs(dr[inner]).max()


def test_integral_and_box_sum():
    a = np.arange(12, dtype=float).reshape(3, 4)
    s = integral_image(a)
    assert s[3, 4] == a.sum() and s[2, 3] == a[:2, :3].sum()
    ref = ndimage.uniform_filter(a, 3, mode="nearest") * 9
    np.testing.assert_allclose(box_sum(a, 1), ref, atol=1e-9)


@pytest.mark.parametrize("offset", [0.0, -2600.0])
def test_local_std_matches_generic_filter(offset):
    dem = make_dem(30, 35, seed=1).astype(np.float64) + offset
    ref = ndimage.generic_filter(dem, np.std, size=5, mode="nearest")
    np.testing.assert_allclose(local_std(dem, 2), ref, atol=1e-3)


def test_hillshade_flat_is_uniform():
    hs = hillshade(np.full((10, 10), 123.0), 30.0, altitude_deg=45)
    np.testing.assert_allclose(hs, np.sin(np.radians(45)), atol=1e-6)
