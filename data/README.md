# Dữ liệu

Dữ liệu **không commit lên git** vì quá nặng. Tải về theo bảng dưới, bỏ vào `data/raw/`, rồi chạy:

```bash
python scripts/prepare_data.py
```

Script sẽ cắt, chiếu lại về cùng lưới và ghi ra `data/mars/` và `data/vn/` theo định dạng ở `docs/interface.md`.

## Sao Hỏa: hố Jezero

| File | Nguồn | Ghi chú |
| --- | --- | --- |
| `M20_JezeroCrater_CTXDEM_20m.tif` | https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/ScienceInvestigationMaps_JPL/M20_JezeroCrater_CTXDEM_20m.tif | DEM CTX 20 m/pixel (USGS Astrogeology) |

Chỉ cần đúng file DEM này: độ dốc, độ gồ ghề, hố và ảnh nền (hillshade) đều tính từ DEM.

Vùng cắt: _(điền khung tọa độ đã chọn)_

## Việt Nam: khu vực Bảo Yên, Lào Cai (sạt lở sau bão Yagi, 9/2024)

| File | Nguồn | Ghi chú |
| --- | --- | --- |
| DEM | NASADEM (NASA Earthdata, `NASADEM_HGT_n22e104`) | 30 m; thay bằng Copernicus GLO-30 nếu được cấp quyền CCM |
| Sentinel-2 trước bão | https://browser.dataspace.copernicus.eu | L2A, **2024-04-27** (không mây cả khung); băng B04, B08, SCL |
| Sentinel-2 sau bão | https://browser.dataspace.copernicus.eu | L2A, **2025-05-12** (cùng mùa với ảnh trước; vài đám mây nhỏ ở giữa khung, lọc bằng SCL); băng B04, B08, SCL |
| Sentinel-1 | https://browser.dataspace.copernicus.eu | IW, VV (Raw, gamma0 tuyến tính), quỹ đạo **Ascending**: 2024-09-03 (trước bão), 2024-09-15, 2024-09-27. Đỉnh lũ (9–11/9) rơi giữa hai lần chụp nên dùng để kiểm chứng; lớp ngập chính tính từ DEM theo mực nước giả định |
| Đường OSM | https://overpass-turbo.eu | `osm_roads.geojson`: `way["highway"]` trong khung VN, xuất GeoJSON (10/11/2026). © OpenStreetMap contributors (ODbL) |

Vùng cắt: khung 30 × 30 km, file `data/aoi_vn.geojson` (WGS84)

- Tây 104.2952, Nam 22.0555, Đông 104.5862, Bắc 22.325
- Chứa: Làng Nủ (22.1451, 104.5360), sông Chảy ở Phố Ràng (22.2354, 104.5002), sông Hồng ở Bảo Hà (22.1877, 104.3454)
- Dùng đúng khung này cho mọi lần tải (DEM, Sentinel-1, Sentinel-2) để các lớp khớp nhau

### Cấu trúc `data/raw/vn/` sau khi tải

```
data/raw/vn/
├── nasadem_n22e104/   n22e104.hgt (DEM 30 m, int16), n22e104.swb (mặt nạ sông hồ)
├── s2_20240427/       B04.tif, B08.tif, SCL_vis.tif
├── s2_20250512/       B04.tif, B08.tif, SCL_vis.tif
├── s1_20240903/, s1_20240915/, s1_20240927/   VV.tif
└── osm_roads.geojson
```

Sentinel-2 tải từ Copernicus Browser, tab Analytical: TIFF 32-bit float, UTM 48N (EPSG:32648),
độ phân giải 15 m (2002 × 1981 px), cắt theo `aoi_vn.geojson`. Hai ngày cùng một lưới.
`SCL_vis.tif` là bản tô màu của lớp phân loại cảnh (3 băng RGB), `prepare_data.py` đổi màu ngược về mã loại.

## Trích dẫn

- European Space Agency (2024). Copernicus Global Digital Elevation Model. Distributed by OpenTopography. https://doi.org/10.5069/G9028PQB
- USGS Astrogeology Science Center, Mars 2020 Science Investigation CTX DEM Mosaic.
