"""Lớp địa hình tính từ DEM (Phần 2).

Tất cả tự cài bằng NumPy theo kiểu vector hoá (dịch mảng), không lặp từng pixel.
Hàm thư viện (np.gradient, scipy.ndimage) chỉ dùng trong tests/ để đối chiếu.

Quy ước: hàng tăng xuống dưới (hướng Nam), cột tăng sang phải (hướng Đông).
Mép ảnh: đệm bằng cách lặp giá trị mép (edge), để mép không bị kéo về 0.
"""
from __future__ import annotations

import numpy as np

# Mặt nạ Sobel (công thức 2.x trong báo cáo)
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)   # đạo hàm theo cột (sang Đông)
SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float64)    # đạo hàm theo hàng (xuống Nam)


def correlate3x3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tương quan (correlation) ảnh với mặt nạ 3x3, đệm mép bằng giá trị mép.

    out[r, c] = sum_{i,j} kernel[i, j] * img[r + i - 1, c + j - 1]
    Cài bằng 9 phép dịch mảng: O(9·M·N) thời gian, O(M·N) bộ nhớ.
    """
    a = np.asarray(img, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError("kernel phải có kích thước 3x3")
    p = np.pad(a, 1, mode="edge")
    rows, cols = a.shape
    out = np.zeros_like(a)
    for i in range(3):
        for j in range(3):
            if k[i, j] != 0.0:
                out += k[i, j] * p[i:i + rows, j:j + cols]
    return out


def gradient_sobel(dem: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Đạo hàm riêng theo mét/mét: (dz/dx hướng Đông, dz/dy hướng Bắc).

    Sobel cho tổng có trọng số của các hiệu cách nhau 2 ô; tổng trọng số
    là 1+2+1 = 4, khoảng cách là 2·cell → chia cho 8·cell_size_m.
    """
    gx = correlate3x3(dem, SOBEL_X) / (8.0 * cell_size_m)        # dz/dx (Đông)
    gy_south = correlate3x3(dem, SOBEL_Y) / (8.0 * cell_size_m)  # dz/d(hàng) = hướng Nam
    return gx, -gy_south                                         # đổi sang hướng Bắc


def slope_aspect(dem: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Độ dốc (độ) và hướng dốc (độ, 0 = Bắc, theo chiều kim đồng hồ; NaN nếu phẳng).

    slope  = arctan( sqrt(gx² + gy²) )
    aspect = hướng của −∇z (hướng nước chảy xuống) = atan2(−gx, −gy)
    """
    gx, gy = gradient_sobel(dem, cell_size_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.degrees(np.arctan2(-gx, -gy)) % 360.0
    aspect[np.hypot(gx, gy) < 1e-9] = np.nan
    return slope.astype(np.float32), aspect.astype(np.float32)


def integral_image(a: np.ndarray) -> np.ndarray:
    """Ảnh tích phân có thêm 1 hàng, 1 cột số 0 ở đầu: S[r, c] = tổng a[:r, :c]."""
    a = np.asarray(a, dtype=np.float64)
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return s


def box_sum(a: np.ndarray, radius: int) -> np.ndarray:
    """Tổng trong cửa sổ vuông (2·radius+1)² quanh mỗi ô, dùng 4 phép tra ảnh tích phân.

    Độ phức tạp O(M·N), không phụ thuộc kích thước cửa sổ.
    """
    r = int(radius)
    p = np.pad(np.asarray(a, dtype=np.float64), r, mode="edge")
    s = integral_image(p)
    rows, cols = np.asarray(a).shape
    w = 2 * r + 1
    return (s[w:w + rows, w:w + cols] - s[0:rows, w:w + cols]
            - s[w:w + rows, 0:cols] + s[0:rows, 0:cols])


def local_std(dem: np.ndarray, radius: int = 2) -> np.ndarray:
    """Độ gồ ghề = độ lệch chuẩn độ cao trong cửa sổ (2r+1)², tính bằng ảnh tích phân.

    σ² = E[z²] − (E[z])². Trừ giá trị trung bình toàn ảnh trước để tránh mất
    chính xác khi độ cao lớn (vd −2600 m trên Sao Hỏa) mà σ chỉ vài mét.
    """
    z = np.asarray(dem, dtype=np.float64)
    z = z - np.nanmean(z)
    n = float((2 * radius + 1) ** 2)
    mean = box_sum(z, radius) / n
    mean_sq = box_sum(z * z, radius) / n
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32)


def hillshade(dem: np.ndarray, cell_size_m: float, azimuth_deg: float = 315.0,
              altitude_deg: float = 45.0) -> np.ndarray:
    """Ảnh bóng đổ 0..1 (chỉ để hiển thị), Mặt Trời ở phương vị `azimuth_deg`, độ cao `altitude_deg`.

    hs = cos(zenith)·cos(slope) + sin(zenith)·sin(slope)·cos(azimuth − aspect)
    """
    slope, aspect = slope_aspect(dem, cell_size_m)
    s = np.radians(slope.astype(np.float64))
    a = np.radians(np.nan_to_num(aspect.astype(np.float64)))
    zen = np.radians(90.0 - altitude_deg)
    az = np.radians(azimuth_deg)
    hs = np.cos(zen) * np.cos(s) + np.sin(zen) * np.sin(s) * np.cos(az - a)
    return np.clip(hs, 0.0, 1.0).astype(np.float32)


def terrain_layers(dem: np.ndarray, cell_size_m: float, rough_radius: int = 2) -> dict[str, np.ndarray]:
    slope, aspect = slope_aspect(dem, cell_size_m)
    return {
        "slope": slope,
        "aspect": aspect,
        "roughness": local_std(dem, rough_radius),
        "hillshade": hillshade(dem, cell_size_m),
    }
