# terrain-pathfinder

Hazard-aware A* pathfinding on multi-layer terrain maps (Mars & Vietnam), with image-processing-based slope, flood, landslide and crater layers.

Bài tập lớn môn Xử lý ảnh, Đề 26 (danh sách đề NASA): tìm lộ trình đi bộ an toàn trên bản đồ địa hình nhiều lớp, cho một vùng Sao Hỏa (hố Jezero) và một huyện miền núi Việt Nam bị lũ, sạt lở chia cắt.

## Cấu trúc

```
src/            lõi thuật toán (không import gì từ app/)
  grid_io.py      đọc/ghi lớp .npy + meta.json
  synthetic.py    sinh DEM và bản đồ chi phí giả để thử
  terrain.py      độ dốc (Sobel), độ gồ ghề, hillshade
  hazards.py      Otsu, lớp ngập, sạt lở, hố, vùng đệm
  cost.py         bản đồ chi phí, hàm Tobler
  pathfinding.py  Dijkstra, A*, các phần mở rộng
app/main.py     giao diện Streamlit, chỉ gọi hàm trong src/
scripts/        tiền xử lý dữ liệu thô thành .npy
experiments/    run_experiments.py và kết quả
tests/          so thuật toán tự cài với hàm thư viện
data/           dữ liệu (không commit), xem data/README.md
docs/           interface.md: định dạng dữ liệu chung
report/         báo cáo, slide
```

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy

Mọi lệnh chạy từ thư mục gốc của repo.

```bash
python scripts/prepare_data.py          # dữ liệu thô -> data/mars, data/vn
pytest                                  # kiểm tra thuật toán
python experiments/run_experiments.py   # tái lập bảng và hình Chương 4
streamlit run app/main.py               # giao diện demo
```

Định dạng dữ liệu và chữ ký hàm giữa các phần: [docs/interface.md](docs/interface.md).
