# Kịch bản cập nhật theo thời gian (dùng cho `src/live.py` và trang "Theo thời gian" của giao diện)

> **GIẢ LẬP – CHỈ ĐỂ MINH HỌA.** Không có dữ liệu camera ngập hay số đo mực nước thật cho khu vực này.
> Các con số dưới đây được đặt tay để trình diễn cơ chế cập nhật và tìm lại đường; không dùng làm kết quả đo đạc.

## `yagi_demo_levels.csv` — mực nước sông dâng h(t)

Cột: `time` (giờ Việt Nam), `level_m` (m, cao hơn lòng sông thường ngày). Giữa hai mốc nội suy tuyến tính.
Hình dạng giả định: lên từ 7/9, đỉnh 7 m sáng 10/9, rút dần tới 14/9 — khớp thời gian bão Yagi
(đổ bộ 7/9/2024, lũ sông Hồng ở Lào Cai, Yên Bái lên cao nhất khoảng 9–11/9), độ lớn là giả định.

## `yagi_demo_observations.csv` — quan sát điểm

| Cột | Ý nghĩa |
| --- | --- |
| `time` | thời điểm nhận báo cáo |
| `kind` | `camera_flood` (camera báo ngập), `landslide` (báo sạt lở), `blocked` (đường bị chặn) |
| `lat`, `lon` | vị trí (WGS84) |
| `value` | `camera_flood`: độ sâu nước (m); `landslide`, `blocked`: bán kính ảnh hưởng (m) |
| `ttl_h` | thời hạn hiệu lực (giờ), hết hạn thì bỏ |
| `source`, `note` | nguồn, ghi chú |

Cách đặt các điểm demo:
1. Camera ven sông Hồng ở Bảo Hà (10/9 08:00, sâu 3 m) — để thấy vùng ngập quanh camera trên bản đồ.
2. Báo sạt lở (11/9 07:30, bán kính 150 m) đặt tại khoảng 55% tuyến A* ban đầu Phố Ràng → Làng Nủ (xuất phát 11/9 06:00).
3. Camera ở suối (11/9 08:40, sâu 2 m) đặt tại ô thấp nhất ở đoạn giữa tuyến sau lần tìm lại thứ nhất.

Kết quả (`experiments/run_realtime.py`): có tìm lại đường thì tới nơi lúc 11:09 (2 lần tìm lại, 07:57 và 08:56);
giữ nguyên tuyến ban đầu thì bị kẹt lúc 08:07.

Ghi chú 11/11/2026: sau khi sửa lớp ngập theo mực nước (vùng ngập lồng nhau theo h), tuyến ban đầu
đổi nên camera thứ 3 được dời từ ô (407, 696) sang ô (493, 719), vẫn là ô trũng ở đoạn giữa tuyến
sau lần tìm lại thứ nhất. Báo sạt lở giữ nguyên.

Muốn thử dữ liệu khác: tạo CSV cùng cột rồi tải lên ở trang "Theo thời gian" của giao diện.
