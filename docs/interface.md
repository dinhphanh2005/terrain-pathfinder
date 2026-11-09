# Định dạng dữ liệu chung

Mọi người làm song song theo đúng file này. Muốn đổi gì thì báo cả nhóm và sửa file này trước, rồi mới sửa code.

## 1. Quy ước lưới

- Mỗi lớp là một mảng NumPy 2 chiều, kiểu `float32`, lưu thành `.npy`.
- Mọi lớp của **cùng một vùng** có cùng kích thước `(rows, cols)` và cùng lưới (đã chiếu lại trong `scripts/prepare_data.py`).
- Tọa độ một ô luôn là `(hàng, cột)` = `(r, c)`, gốc ở góc trên bên trái. Không dùng `(x, y)` trong code lõi.
- Ô không có dữ liệu trong DEM: `np.nan`.

## 2. File meta đi kèm

Mỗi `ten_lop.npy` có một `ten_lop.json` cùng tên:

```json
{
  "region": "mars",
  "layer": "dem",
  "rows": 1000,
  "cols": 1000,
  "cell_size_m": 20.0,
  "unit": "m",
  "source": "USGS M20_JezeroCrater_CTXDEM_20m.tif",
  "crs": "chuỗi CRS gốc (nếu có)",
  "transform": [a, b, c, d, e, f]
}
```

`cell_size_m` là bắt buộc: độ dốc, khoảng cách và thời gian đi bộ đều phụ thuộc vào nó.

## 3. Đọc và ghi

Chỉ đọc, ghi lớp qua `src/grid_io.py`:

- `save_layer(path, array, meta)`
- `load_layer(path) -> (array, meta)`

## 4. Tên file đầu ra theo từng phần

Thư mục: `data/mars/` và `data/vn/`.

| Phần | Người | File | Giá trị |
| --- | --- | --- | --- |
| 1. Dữ liệu | B | `dem.npy` | độ cao (m) |
| 1. Dữ liệu | B | `ortho.npy` | ảnh nền xám 0–1 (nếu có) |
| 1. Dữ liệu | B | `s1_vv_<YYYYMMDD>.npy`, `s2_ndvi_before.npy`, `s2_ndvi_after.npy` | chỉ vùng `vn` |
| 2. Địa hình | A | `slope.npy` | độ dốc (độ) |
| 2. Địa hình | A | `aspect.npy` | hướng dốc (độ, 0 = Bắc, theo chiều kim đồng hồ) |
| 2. Địa hình | A | `roughness.npy` | độ lệch chuẩn cục bộ của độ cao (m) |
| 2. Địa hình | A | `hillshade.npy` | 0–1, chỉ để hiển thị |
| 3. Nguy hiểm | B | `hazard_flood_<YYYYMMDD>.npy`, `hazard_landslide.npy` | 0/1, chỉ vùng `vn` |
| 3. Nguy hiểm | A | `hazard_crater.npy` | 0/1, chỉ vùng `mars` |
| 3. Nguy hiểm | B | `forbidden.npy` | 0/1, hợp các lớp nguy hiểm sau khi giãn vùng đệm |
| 4. Chi phí | A | `cost.npy` | số giây để đi qua ô, ô cấm = `np.inf` |

## 5. Chữ ký hàm giữa các phần

```python
# src/cost.py (A)
build_cost(layers: dict[str, np.ndarray], meta: dict, weights: dict) -> np.ndarray

# src/pathfinding.py (dp)
find_path(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
          method: str = "astar") -> tuple[list[tuple[int, int]], float, int]
# trả về (đường đi gồm các ô (r, c), tổng chi phí, số nút đã duyệt)
# không tìm được đường: ([], np.inf, số nút đã duyệt)
```

- Lân cận 8 hướng. Đi chéo tốn chi phí trung bình hai ô × √2, đi thẳng tốn trung bình hai ô × 1.
- `cost` phải dương hoặc `np.inf`. Hàm nhận giá trị ≤ 0 hoặc NaN thì báo lỗi.

## 6. Quy tắc chung

- Thuật toán cốt lõi tự cài bằng NumPy. Hàm thư viện (`skimage`, `scipy`, `cv2`) chỉ dùng trong `tests/` để đối chiếu.
- Mỗi hàm lõi có ít nhất một test trong `tests/` so với hàm thư viện tương ứng.
- Không commit dữ liệu. Ai tải dữ liệu mới thì ghi nguồn vào `data/README.md`.
