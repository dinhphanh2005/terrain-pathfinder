import numpy as np
import pytest
from scipy import ndimage
from skimage.filters import threshold_otsu

from src import hazards as hz


def test_ndvi_basic():
    out = hz.ndvi(np.array([[0.1, 0.0]]), np.array([[0.5, 0.0]]))
    assert out[0, 0] == pytest.approx(0.4 / 0.6)
    assert np.isnan(out[0, 1])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_otsu_matches_skimage(seed):
    rng = np.random.default_rng(seed)
    v = np.concatenate([rng.normal(0.1, 0.05, 5000), rng.normal(0.6, 0.1, 3000)])
    assert hz.otsu_threshold(v) == pytest.approx(threshold_otsu(v), rel=1e-9)


def _rand_mask(seed, shape=(60, 70), p=0.45):
    return np.random.default_rng(seed).random(shape) < p


@pytest.mark.parametrize("r", [1, 2, 3])
def test_binary_morphology_matches_scipy(r):
    m, se = _rand_mask(r), hz.disk(r)
    np.testing.assert_array_equal(hz.dilate(m, se), ndimage.binary_dilation(m, se))
    np.testing.assert_array_equal(hz.erode(m, se), ndimage.binary_erosion(m, se, border_value=1))
    np.testing.assert_array_equal(hz.opening(m, se),
                                  ndimage.binary_dilation(ndimage.binary_erosion(m, se, border_value=1), se))


def test_grey_morphology_matches_scipy():
    img = np.random.default_rng(3).random((40, 45))
    se = hz.disk(2)
    np.testing.assert_allclose(hz.grey_dilate(img, se), ndimage.grey_dilation(img, footprint=se, mode="nearest"))
    np.testing.assert_allclose(hz.grey_erode(img, se), ndimage.grey_erosion(img, footprint=se, mode="nearest"))


@pytest.mark.parametrize("conn", [4, 8])
def test_label_matches_scipy(conn):
    m = _rand_mask(5, (80, 90), 0.4)
    st = ndimage.generate_binary_structure(2, 2 if conn == 8 else 1)
    ref, nref = ndimage.label(m, st)
    lab, n = hz.label(m, conn)
    assert n == nref
    # cùng cách chia vùng (nhãn có thể đánh số khác): mỗi cặp nhãn tương ứng 1-1
    pairs = set(zip(lab[m].tolist(), ref[m].tolist()))
    assert len(pairs) == n


def test_region_props_and_remove_small():
    m = np.zeros((20, 20), bool)
    m[2:6, 2:6] = True      # 16 ô
    m[10, 10] = True        # 1 ô
    lab, n = hz.label(m)
    pr = hz.region_props(lab, n)
    assert sorted(pr["area"][1:].tolist()) == [1, 16]
    assert pr["perimeter"][lab[2, 2]] == 16
    out = hz.remove_small(m, 5)
    assert out.sum() == 16


def test_flood_by_level_connectivity():
    dem = np.full((5, 7), 10.0)
    dem[:, 0] = 0.0                  # sông ở cột 0
    dem[:, 1] = 1.0                  # bãi ven sông thấp
    dem[2, 5] = 0.5                  # lòng chảo kín, không nối sông
    river = np.zeros_like(dem, bool); river[:, 0] = True
    f = hz.flood_by_level(dem, river, 2.0)
    assert f[:, 1].all() and not f[:, 2:].any()
    f = hz.flood_by_level(dem, river, 20.0)
    assert f[:, 1:].all()


def test_flood_by_level_nested():
    """Nước dâng cao hơn thì vùng ngập không được nhỏ đi (lồng nhau)."""
    rng = np.random.default_rng(0)
    dem = np.cumsum(rng.normal(0, 1, (60, 60)), axis=1) + np.linspace(0, 30, 60)[:, None]
    river = np.zeros(dem.shape, bool); river[:, 0] = True
    prev = np.zeros(dem.shape, bool)
    for h in (0.5, 1, 2, 4, 6, 8):
        f = hz.flood_by_level(dem, river, h)
        assert not (prev & ~f).any()
        prev = f


def test_crater_detects_synthetic_pit():
    r, c = np.mgrid[0:80, 0:80]
    dem = -2000.0 + 0.0 * r
    pit = (r - 40) ** 2 + (c - 40) ** 2 <= 8 ** 2
    dem[pit] -= 30.0
    out = hz.crater_mask(dem, se_radius=12, min_depth=8.0, min_cells=20)
    assert out[40, 40] and out.sum() == pit.sum() and not out[5, 5]


def test_buffer_radius():
    m = np.zeros((21, 21), bool); m[10, 10] = True
    b = hz.buffer(m, 30.0, 90.0)
    assert b[10, 13] and not b[10, 14]
