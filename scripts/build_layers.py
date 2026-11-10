"""Tính mọi lớp dẫn xuất (địa hình, nguy hiểm) cho cả hai vùng và lưu ảnh xem nhanh.

Chạy sau prepare_data.py:   python scripts/build_layers.py [--only mars|vn] [--preview]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.layers import build_region  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["mars", "vn"])
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    for region in ([a.only] if a.only else ["mars", "vn"]):
        t = time.perf_counter()
        layers = build_region(region)
        print(f"[{region}] {len(layers)} lớp, {time.perf_counter() - t:.1f} s: {', '.join(layers)}")
        if a.preview:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            out = ROOT / "data" / "preview"
            out.mkdir(parents=True, exist_ok=True)
            for name, arr in layers.items():
                fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
                im = ax.imshow(arr, cmap="gray" if name == "hillshade" else "viridis", interpolation="nearest")
                ax.set_title(f"{region}/{name}"); ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.savefig(out / f"{region}_{name}.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
