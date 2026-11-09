import numpy as np
import pytest

from src.grid_io import list_layers, load_layer, save_layer

META = {"region": "test", "layer": "dem", "cell_size_m": 30.0}


def test_round_trip(tmp_path):
    arr = np.random.default_rng(0).random((5, 7))
    save_layer(tmp_path / "dem", arr, META)
    out, meta = load_layer(tmp_path / "dem.npy")
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, arr.astype(np.float32))
    assert (meta["rows"], meta["cols"]) == (5, 7)
    assert list_layers(tmp_path) == ["dem"]


def test_nan_and_inf_survive(tmp_path):
    arr = np.array([[1.0, np.nan], [np.inf, 2.0]])
    save_layer(tmp_path / "x", arr, META)
    out, _ = load_layer(tmp_path / "x")
    assert np.isnan(out[0, 1]) and np.isinf(out[1, 0])


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((2, 2, 2))])
def test_rejects_non_2d(tmp_path, bad):
    with pytest.raises(ValueError):
        save_layer(tmp_path / "x", bad, META)


def test_rejects_missing_meta(tmp_path):
    with pytest.raises(ValueError):
        save_layer(tmp_path / "x", np.zeros((2, 2)), {"region": "test"})


def test_detects_shape_mismatch(tmp_path):
    save_layer(tmp_path / "x", np.zeros((3, 3)), META)
    np.save(tmp_path / "x.npy", np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        load_layer(tmp_path / "x")
