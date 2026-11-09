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
| `M20_JezeroCrater_CTXortho_mosaic_5m.tif` | https://astropedia.astrogeology.usgs.gov/download/Mars/Mars2020/ancillary/M20_JezeroCrater_CTXortho_mosaic_5m.tif | Ảnh trực giao CTX 5 m, làm nền hiển thị |

Vùng cắt: _(điền khung tọa độ đã chọn)_

## Việt Nam: khu vực Bảo Yên, Lào Cai (sạt lở sau bão Yagi, 9/2024)

| File | Nguồn | Ghi chú |
| --- | --- | --- |
| DEM | Copernicus GLO-30 qua https://opentopography.org | 30 m |
| Sentinel-2 trước bão | https://browser.dataspace.copernicus.eu | Tháng 8/2024, ít mây; băng B4, B8 để tính NDVI |
| Sentinel-2 sau bão | https://browser.dataspace.copernicus.eu | Cuối 9 đến 10/2024, ít mây |
| Sentinel-1 | https://browser.dataspace.copernicus.eu | GRD, phân cực VV, 2–3 ngày quanh đỉnh lũ |

Vùng cắt: _(điền khung tọa độ đã chọn, khoảng 30 × 30 km)_

## Trích dẫn

- European Space Agency (2024). Copernicus Global Digital Elevation Model. Distributed by OpenTopography. https://doi.org/10.5069/G9028PQB
- USGS Astrogeology Science Center, Mars 2020 Science Investigation CTX DEM Mosaic.
