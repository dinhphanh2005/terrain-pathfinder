"""Lớp nguy hiểm: NDVI, ngưỡng Otsu, hình thái học, gán nhãn, sạt lở, ngập theo mực nước, hố Sao Hỏa, vùng đệm.

Tất cả tự cài bằng NumPy (Chương 6, 7 giáo trình). tests/ đối chiếu với scipy.ndimage và skimage.
Mặt nạ nhị phân dùng kiểu bool; khi lưu ra .npy thì đổi sang 0/1 float32.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# ---------------------------------------------------------------- chỉ số phổ
def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red), trong [-1, 1]. Mẫu số 0 -> NaN."""
    red = np.asarray(red, dtype=np.float64)
    nir = np.asarray(nir, dtype=np.float64)
    den = nir + red
    out = np.full(red.shape, np.nan)
    ok = den > 0
    out[ok] = (nir[ok] - red[ok]) / den[ok]
    return out.astype(np.float32)


# ---------------------------------------------------------------- Otsu
def otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Ngưỡng Otsu (1979) trên các giá trị hữu hạn.

    Chia histogram thành 2 lớp tại ngưỡng t, chọn t làm cực đại phương sai giữa hai lớp:
        σ_B²(t) = ω0(t)·ω1(t)·(μ0(t) - μ1(t))²
    Tính cho mọi t cùng lúc bằng tổng tích lũy -> O(N + L).
    Cách chia bin giống skimage.filters.threshold_otsu (để đối chiếu được).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    lo, hi = v.min(), v.max()
    if lo == hi:
        return float(lo)
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = hist.astype(np.float64)
    w0 = np.cumsum(hist)                 # số phần tử lớp 0 (<= t)
    w1 = np.cumsum(hist[::-1])[::-1]     # số phần tử lớp 1 (>= t)
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-12)
    m1 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w1[::-1], 1e-12))[::-1]
    between = w0[:-1] * w1[1:] * (m0[:-1] - m1[1:]) ** 2
    return float(centers[:-1][np.argmax(between)])


# ---------------------------------------------------------------- hình thái học
def disk(radius: int) -> np.ndarray:
    """Phần tử cấu trúc hình tròn bán kính r ô (bool, (2r+1)x(2r+1))."""
    r = int(radius)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1e-9


def _shifted_reduce(img: np.ndarray, se: np.ndarray, op, pad_value):
    """Áp op (np.maximum / np.minimum / or / and) qua mọi vị trí bật của phần tử cấu trúc.

    out[p] = op_{q in SE} img[p + q]  -> O(M·N·|SE|), vector hóa theo từng dịch chuyển.
    """
    r0, c0 = se.shape[0] // 2, se.shape[1] // 2
    a = np.pad(img, ((r0, r0), (c0, c0)), constant_values=pad_value)
    rows, cols = img.shape
    out = None
    for i, j in zip(*np.nonzero(se)):
        view = a[i:i + rows, j:j + cols]
        out = view.copy() if out is None else op(out, view)
    return out


def dilate(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Giãn nhị phân A ⊕ B: ô bật nếu có ít nhất một ô bật trong cửa sổ B. Ngoài ảnh coi là 0."""
    return _shifted_reduce(np.asarray(mask, bool), se, np.logical_or, False)


def erode(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Co nhị phân A ⊖ B: ô bật nếu mọi ô trong cửa sổ B đều bật. Ngoài ảnh coi là 1 (không ăn mòn từ mép)."""
    return _shifted_reduce(np.asarray(mask, bool), se, np.logical_and, True)


def opening(mask, se):
    """Mở A ∘ B = (A ⊖ B) ⊕ B: xóa đốm nhỏ hơn B, giữ hình dạng vật lớn."""
    return dilate(erode(mask, se), se)


def closing(mask, se):
    """Đóng A • B = (A ⊕ B) ⊖ B: lấp lỗ và khe hẹp hơn B."""
    return erode(dilate(mask, se), se)


def grey_dilate(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Giãn mức xám: lấy max trong cửa sổ B (mép: lặp giá trị)."""
    r = se.shape[0] // 2
    a = np.pad(np.asarray(img, np.float64), r, mode="edge")
    rows, cols = img.shape
    out = np.full((rows, cols), -np.inf)
    for i, j in zip(*np.nonzero(se)):
        np.maximum(out, a[i:i + rows, j:j + cols], out=out)
    return out


def grey_erode(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Co mức xám: lấy min trong cửa sổ B (mép: lặp giá trị)."""
    return -grey_dilate(-np.asarray(img, np.float64), se)


def black_tophat(img: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Top-hat đen = đóng(img) - img: làm nổi các chỗ trũng nhỏ hơn B (đáy hố va chạm)."""
    return grey_erode(grey_dilate(img, se), se) - np.asarray(img, np.float64)


# ---------------------------------------------------------------- gán nhãn
def label(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Gán nhãn thành phần liên thông, hai lượt trên các đoạn chạy (run) theo hàng + union-find.

    Lượt 1: mỗi hàng tách thành các đoạn liên tiếp bật; đoạn ở hàng r nối với đoạn ở hàng r-1 nếu
    chồng nhau (8 láng giềng: chồng cả chéo, tức nới mỗi đầu 1 ô). Lượt 2: gom nhãn tương đương.
    Làm trên đoạn thay vì từng pixel: số đoạn nhỏ hơn số pixel rất nhiều -> nhanh trong Python.
    Nhãn đánh 1..n theo thứ tự quét (trên xuống, trái sang), 0 là nền.
    """
    m = np.asarray(mask, bool)
    rows, cols = m.shape
    ext = 1 if connectivity == 8 else 0
    starts, ends, rowof = [], [], []
    for r in range(rows):
        d = np.diff(np.concatenate(([0], m[r].view(np.int8), [0])))
        s = np.nonzero(d == 1)[0]
        e = np.nonzero(d == -1)[0]  # e là vị trí sau đoạn
        starts.extend(s.tolist()); ends.extend(e.tolist()); rowof.extend([r] * len(s))
    n = len(starts)
    parent = list(range(n))

    def find(x):
        """Gốc của tập chứa đoạn x (union-find), kèm nén đường đi: trỏ thẳng lên ông."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # chỉ số đoạn đầu tiên của mỗi hàng
    row_first = [0] * (rows + 1)
    k = 0
    for r in range(rows + 1):
        while k < n and rowof[k] < r:
            k += 1
        row_first[r] = k
    for r in range(1, rows):
        a, a_end = row_first[r - 1], row_first[r]
        b, b_end = row_first[r], row_first[r + 1]
        while a < a_end and b < b_end:
            if starts[a] < ends[b] + ext and starts[b] < ends[a] + ext:  # hai đoạn chồng nhau
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            if ends[a] < ends[b]:
                a += 1
            else:
                b += 1
    out = np.zeros((rows, cols), dtype=np.int32)
    root_label = {}
    for i in range(n):
        root = find(i)
        lab = root_label.setdefault(root, len(root_label) + 1)
        out[rowof[i], starts[i]:ends[i]] = lab
    return out, len(root_label)


def region_props(labels: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Diện tích (số ô), chu vi (số cạnh ô tiếp giáp nền, 4 hướng) và độ tròn 4πA/P² cho nhãn 1..n.

    Chu vi đếm cạnh pixel nên độ tròn của hình tròn thật chỉ khoảng 0,7–0,8 (không đạt 1).
    """
    area = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    p = np.pad(labels, 1)
    core = p[1:-1, 1:-1]
    per = np.zeros(n + 1)
    for nb in (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]):
        edge = (core > 0) & (nb != core)
        per += np.bincount(core[edge], minlength=n + 1)
    circ = np.where(per > 0, 4 * np.pi * area / np.maximum(per, 1) ** 2, 0)
    return {"area": area, "perimeter": per, "circularity": circ}


def remove_small(mask: np.ndarray, min_cells: int) -> np.ndarray:
    """Bỏ các thành phần liên thông nhỏ hơn min_cells ô."""
    lab, n = label(mask)
    if n == 0:
        return np.zeros_like(mask, bool)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = area >= min_cells
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------- lớp sạt lở
# Mã SCL bị coi là không dùng được khi so NDVI. 10 (mây ti mỏng) để riêng vì SCL hay gắn quá tay.
SCL_BAD = (0, 1, 3, 8, 9, 11)


def valid_scl(scl: np.ndarray, drop_cirrus: bool = False) -> np.ndarray:
    """Mặt nạ ô Sentinel-2 dùng được theo lớp phân loại cảnh SCL.

    Vào: scl (mã lớp SCL theo ô), drop_cirrus (True thì bỏ cả mây ti, mã 10).
    Ra: mảng bool, True = ô hợp lệ. Bỏ các mã trong SCL_BAD (0 không có dữ liệu, 1 điểm ảnh lỗi,
    3 bóng mây, 8–9 mây trung bình/dày, 11 tuyết) và ô NaN.
    """
    bad = list(SCL_BAD) + ([10] if drop_cirrus else [])
    return np.isfinite(scl) & ~np.isin(scl, bad)


def landslide_mask(red0, nir0, scl0, red1, nir1, scl1, slope_deg, *,
                   min_drop=0.25, min_slope=15.0, min_cells=10,
                   drop_cirrus=False, return_steps=False):
    """Vùng sạt lở = chỗ TRƯỚC là rừng, SAU thành đất trống, trên sườn dốc.

    1. NDVI hai ngày; bỏ ô mây/bóng mây theo SCL (cả hai ngày phải hợp lệ).
    2. Otsu trên NDVI_trước tách "có thực vật" / "ít thực vật" -> t0; tương tự NDVI_sau -> t1.
       (Không lấy Otsu trực tiếp trên dNDVI: sạt lở chỉ chiếm <1% diện tích nên histogram dNDVI
        chỉ có một đỉnh, Otsu sẽ cắt ngay giữa đỉnh đó -> sai. NDVI từng ngày thì có hai nhóm rõ.)
    3. Ứng viên: NDVI_trước >= t0 và NDVI_sau < t1 và dNDVI = trước - sau >= min_drop.
    4. Đóng bằng đĩa r=1 để nối khe hẹp. KHÔNG mở: phép mở với đĩa r=1 xóa mọi vệt hẹp hơn
       3 ô (90 m), mà dòng bùn đá ở Làng Nủ chỉ rộng 1–2 ô. Đốm lẻ được bỏ ở bước 6 thay vì bằng phép mở.
    5. Lọc theo độ dốc kiểu "ngưỡng trễ" (giống Canny): giữ cả THÀNH PHẦN liên thông nếu nó có
       ít nhất một ô dốc >= min_slope. Sạt lở bắt đầu trên sườn dốc rồi bùn đá chảy xuống lòng
       suối bằng phẳng (đúng như vệt ở Làng Nủ); lọc từng ô sẽ cắt mất phần dòng chảy đó.
       Ruộng mới phát quang ở chỗ bằng không chạm sườn dốc nên bị loại.
    6. Bỏ thành phần nhỏ hơn min_cells ô.
    """
    n0, n1 = ndvi(red0, nir0), ndvi(red1, nir1)
    valid = valid_scl(scl0, drop_cirrus) & valid_scl(scl1, drop_cirrus) & np.isfinite(n0) & np.isfinite(n1)
    t0 = otsu_threshold(n0[valid])
    t1 = otsu_threshold(n1[valid])
    d = (n0 - n1).astype(np.float64)
    d[~valid] = np.nan
    cand = valid & (n0 >= t0) & (n1 < t1) & (d >= min_drop)
    se = disk(1)
    smooth = closing(cand, se)
    lab, n = label(smooth)
    if n:
        has_steep = np.zeros(n + 1, bool)
        has_steep[np.unique(lab[smooth & (np.asarray(slope_deg) >= min_slope)])] = True
        area = np.bincount(lab.ravel(), minlength=n + 1)
        keep = has_steep & (area >= min_cells)
        keep[0] = False
        clean = keep[lab]
    else:
        clean = np.zeros_like(cand)
    filt = smooth
    if return_steps:
        return clean, {"ndvi_before": n0, "ndvi_after": n1, "dndvi": d, "otsu_before": t0,
                       "otsu_after": t1, "candidate": cand, "filtered": filt, "valid": valid}
    return clean


# ---------------------------------------------------------------- lớp ngập theo mực nước
NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def river_reference(dem: np.ndarray, river: np.ndarray) -> np.ndarray:
    """Độ cao của ô sông GẦN NHẤT cho mọi ô (loang BFS 8 hướng từ mọi ô sông, không xét độ cao).

    Không phụ thuộc mực nước nên tính MỘT lần rồi dùng cho mọi h. dem - river_reference(...)
    chính là "độ cao so với lòng sông gần nhất" (HAND, Nobre 2011), bản đơn giản: "gần nhất"
    tính theo số bước trên lưới thay vì theo hướng dòng chảy.
    """
    rows, cols = dem.shape
    z = np.asarray(dem, np.float64)
    ref = np.full((rows, cols), np.nan)
    rr, cc = np.nonzero(river)
    ref[rr, cc] = z[rr, cc]
    refl = ref.tolist()
    q = deque(zip(rr.tolist(), cc.tolist()))
    while q:
        r, c = q.popleft()
        v = refl[r][c]
        for dr, dc in NEIGH8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and refl[nr][nc] != refl[nr][nc]:  # NaN: chưa gán
                refl[nr][nc] = v
                q.append((nr, nc))
    return np.array(refl)


def flood_by_level(dem: np.ndarray, river: np.ndarray, level_m: float,
                   ref: np.ndarray | None = None) -> np.ndarray:
    """Vùng ngập khi nước sông dâng thêm level_m mét.

    1. ref = độ cao ô sông gần nhất (river_reference); HAND = dem - ref.
    2. Ứng viên ngập: HAND <= level_m (thấp hơn mặt nước của đoạn sông gần nó).
    3. Chỉ giữ ứng viên NỐI LIỀN với sông (gán nhãn thành phần liên thông, giữ thành phần chạm sông)
       -> không ngập lòng chảo kín trên núi.
    Lấy ô sông gần nhất (không lấy mực cao nhất) vì sông dốc dần: nếu lấy mực của thượng lưu
    thì nước "tràn" xuống cả thung lũng hạ lưu thấp hơn hàng chục mét -> ngập giả.
    Vì ref không phụ thuộc h nên vùng ngập LỒNG NHAU: ngập ở h thì ngập ở mọi mức cao hơn.
    (Bản cũ gán ref theo ô sông loang tới đầu tiên dưới ràng buộc độ cao -> ref đổi theo h,
    có ô ngập ở 2 m mà không ngập ở 4 m.)
    """
    river = np.asarray(river, bool)
    if ref is None:
        ref = river_reference(dem, river)
    cand = (np.asarray(dem, np.float64) - ref <= level_m) | river
    lab, n = label(cand, 8)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[river])] = True
    keep[0] = False
    return keep[lab] & ~river


def river_mask(water: np.ndarray, scl_before: np.ndarray, min_cells: int = 30) -> np.ndarray:
    """Lòng sông thường xuyên = nước NASADEM ∪ lớp nước SCL (=6) của ảnh trước bão.

    NASADEM chỉ có sông Hồng; sông Chảy hẹp hơn nên lấy thêm từ SCL. Đóng để nối đoạn đứt,
    bỏ vũng nhỏ (ao, ruộng ngập) dưới min_cells ô.
    """
    m = (np.asarray(water) > 0.5) | (scl_before == 6)
    m = closing(m, disk(1))
    return remove_small(m, min_cells)


# ---------------------------------------------------------------- kiểm chứng ngập bằng Sentinel-1
def to_db(x: np.ndarray) -> np.ndarray:
    """Đổi hệ số tán xạ ngược radar từ đơn vị tuyến tính sang dB: 10·log10(x).

    Vào: x tuyến tính (σ⁰). Ra: float32 dB. Cắt x ≥ 1e-6 để không lấy log của 0.
    """
    return (10.0 * np.log10(np.clip(np.asarray(x, np.float64), 1e-6, None))).astype(np.float32)


def median_filter3(img: np.ndarray) -> np.ndarray:
    """Lọc trung vị 3x3 (giảm nhiễu đốm radar). Mép lặp giá trị."""
    a = np.pad(np.asarray(img, np.float64), 1, mode="edge")
    rows, cols = img.shape
    stack = np.stack([a[i:i + rows, j:j + cols] for i in range(3) for j in range(3)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(stack, axis=0)


def s1_flood_change(vv_before, vv_after, slope_deg, *, drop_db=3.0, max_slope=5.0):
    """Ngập phát hiện từ radar: sau lũ tối (dưới ngưỡng Otsu) VÀ tối đi >= drop_db so với trước, chỗ bằng.

    Chỉ dùng để KIỂM CHỨNG lớp ngập theo mực nước (ảnh 12 ngày một lần, bỏ lỡ đỉnh lũ).
    """
    b = median_filter3(to_db(vv_before))
    a = median_filter3(to_db(vv_after))
    t = otsu_threshold(a)
    return (a < t) & ((b - a) >= drop_db) & (slope_deg <= max_slope), t


# ---------------------------------------------------------------- hố Sao Hỏa
def crater_mask(dem: np.ndarray, *, se_radius=12, min_depth=8.0, min_cells=20,
                min_circularity=0.5, return_steps=False):
    """Hố va chạm = chỗ trũng sâu, gần tròn.

    1. Top-hat đen với đĩa bán kính se_radius ô: độ sâu so với "nắp" đóng lại.
    2. Ngưỡng độ sâu >= min_depth (m) -> ứng viên.
    3. Gán nhãn, giữ thành phần >= min_cells ô và độ tròn >= min_circularity.
    """
    th = black_tophat(dem, disk(se_radius))
    cand = th >= min_depth
    lab, n = label(cand)
    if n == 0:
        out = np.zeros_like(cand)
    else:
        pr = region_props(lab, n)
        keep = (pr["area"] >= min_cells) & (pr["circularity"] >= min_circularity)
        keep[0] = False
        out = keep[lab]
    if return_steps:
        return out, {"tophat": th, "candidate": cand}
    return out


# ---------------------------------------------------------------- vùng đệm
def buffer(mask: np.ndarray, cell_size_m: float, radius_m: float) -> np.ndarray:
    """Vùng đệm an toàn: giãn mặt nạ nguy hiểm bằng đĩa bán kính radius_m mét."""
    r = int(round(radius_m / cell_size_m))
    m = np.asarray(mask, bool)
    return m if r <= 0 else dilate(m, disk(r))
