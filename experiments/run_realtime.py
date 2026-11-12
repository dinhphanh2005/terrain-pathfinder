"""Thí nghiệm (d): cập nhật lớp nguy hiểm theo thời gian và tìm lại đường (phần phát triển thêm).

Chạy: python experiments/run_realtime.py
Kịch bản GIẢ LẬP trong data/scenarios/ (xem README ở đó). Kết quả: experiments/results/realtime/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import live  # noqa: E402
from src.layers import PLACES, load_region  # noqa: E402

SCN = ROOT / "data" / "scenarios"
OUT = ROOT / "experiments" / "results" / "realtime"
START, GOAL = PLACES["vn"]["Phố Ràng"], (665, 829)   # Làng Nủ (ô đi được gần nhất)
T0 = pd.Timestamp("2024-09-11 06:00")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L, meta = load_region("vn")
    levels = pd.read_csv(SCN / "yagi_demo_levels.csv")
    obs = live.load_observations(SCN / "yagi_demo_observations.csv", meta)
    rf = live.RiverFloodCache(L["dem"], L["river"] > 0.5)
    runs = {}
    for name, rp in (("Có tìm lại đường", True), ("Giữ tuyến ban đầu", False)):
        ev = live.simulate_walk(L, meta, START, GOAL, T0, levels, obs, update_min=30, replan=rp, river_flood=rf)
        runs[name] = ev
        live.events_table(ev).to_csv(OUT / f"events_{'replan' if rp else 'static'}.csv", index=False)

    rows = []
    for name, ev in runs.items():
        last = ev[-1]
        rep = [e for e in ev if e.action == "tìm lại đường"]
        walked = live.walked_cells(ev)
        rows.append({
            "chiến lược": name,
            "kết quả": "tới nơi" if last.action == "đến nơi" else "kẹt",
            "thời điểm kết thúc": last.time.strftime("%d/%m %H:%M"),
            "thời gian đi (giờ)": round((last.time - T0).total_seconds() / 3600, 2),
            "ước tính lúc xuất phát (giờ)": round(ev[0].remaining_h, 2),
            "số lần tìm lại": len(rep),
            "quãng đường đã đi (km)": round(sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(walked, walked[1:])) * 30 / 1000, 2),
            "thời gian tính lần đầu (s)": round(ev[0].compute_s, 2),
            "thời gian tính mỗi lần tìm lại (s)": round(float(np.mean([e.compute_s for e in rep])), 2) if rep else None,
            "nút duyệt lần đầu": ev[0].expanded,
            "nút duyệt mỗi lần tìm lại": int(np.mean([e.expanded for e in rep])) if rep else None,
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False)

    # ---- hình 1: bản đồ tuyến
    ev = runs["Có tìm lại đường"]
    t_show = [e for e in ev if e.action == "tìm lại đường"][-1].time
    snap = live.snapshot(L, meta, t_show, levels, obs, rf)
    allp = [c for e in ev for c in e.path]
    rr = np.array([c[0] for c in allp]); cc = np.array([c[1] for c in allp])
    r0, r1 = max(0, rr.min() - 40), min(L["dem"].shape[0], rr.max() + 40)
    c0, c1 = max(0, cc.min() - 40), min(L["dem"].shape[1], cc.max() + 40)
    img = np.dstack([L["hillshade"]] * 3).astype(np.float32)
    def over(mask, col, a):
        m = mask.astype(bool); img[m] = img[m] * (1 - a) + np.array(col) * a
    over(L["river"] > 0.5, (0.1, 0.25, 0.8), 0.9)
    over(snap.parts["river_flood"], (0.35, 0.6, 1.0), 0.6)
    over(L["hazard_landslide"] > 0.5, (0.95, 0.35, 0.1), 0.85)
    for k, col in (("landslide", (0.85, 0.0, 0.0)), ("camera_flood", (0.0, 0.85, 0.85)), ("blocked", (0.5, 0.0, 0.5))):
        if k in snap.parts:
            over(snap.parts[k], col, 0.9)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)
    ax.imshow(img[r0:r1, c0:c1], extent=(c0, c1, r1, r0))
    p0 = ev[0].path
    ax.plot([c for _, c in p0], [r for r, _ in p0], "--", color="white", lw=1.3, zorder=5, label="Tuyến ban đầu (06:00)")
    from matplotlib.patches import Patch
    extra = [Patch(color=(0.85, 0, 0), label="Báo sạt lở (giả lập)"), Patch(color=(0, 0.85, 0.85), label="Ngập quanh camera (giả lập)"),
             Patch(color=(0.35, 0.6, 1.0), label="Ngập do sông dâng h(t)"), Patch(color=(0.95, 0.35, 0.1), label="Sạt lở (Sentinel-2)")]
    w = live.walked_cells(runs["Giữ tuyến ban đầu"])
    ax.plot([c for _, c in w], [r for r, _ in w], color="yellow", lw=2.2, label="Giữ tuyến: kẹt lúc " + runs["Giữ tuyến ban đầu"][-1].time.strftime("%H:%M"))
    w = live.walked_cells(ev)
    ax.plot([c for _, c in w], [r for r, _ in w], color="magenta", lw=2.0, label="Có tìm lại đường: tới nơi " + ev[-1].time.strftime("%H:%M"))
    for e in ev:
        if e.action == "tìm lại đường":
            ax.plot(e.cell[1], e.cell[0], "o", ms=9, mfc="white", mec="black")
            ax.annotate(e.time.strftime("%H:%M"), (e.cell[1], e.cell[0]), xytext=(6, -6), textcoords="offset points", fontsize=8, color="white")
    ax.plot(START[1], START[0], "o", ms=9, color="lime", mec="k"); ax.plot(GOAL[1], GOAL[0], "o", ms=9, color="red", mec="k")
    ax.set_title(f"Đi bộ Phố Ràng → Làng Nủ, xuất phát {T0:%d/%m %H:%M} (kịch bản giả lập)\nNền: lớp nguy hiểm lúc {t_show:%H:%M}", fontsize=10)
    h, l = ax.get_legend_handles_labels(); ax.legend(h + extra, l + [e.get_label() for e in extra], loc="lower left", fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "realtime_routes.png", bbox_inches="tight"); plt.close(fig)

    # ---- hình 2: mực nước theo thời gian và các quan sát
    lv = levels.copy(); lv["time"] = pd.to_datetime(lv["time"])
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=130)
    ax.plot(lv["time"], lv["level_m"], "-o", ms=3, label="Mực nước dâng h(t) (giả định)")
    ax.axvspan(T0, ev[-1].time, color="magenta", alpha=0.15, label="Thời gian đi bộ")
    for o in obs.itertuples():
        ax.axvline(o.time, color={"camera_flood": "c", "landslide": "r", "blocked": "purple"}[o.kind], ls=":", lw=1.2)
    ax.set_ylabel("h (m)"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.savefig(OUT / "realtime_levels.png", bbox_inches="tight"); plt.close(fig)

    md = ["# Thí nghiệm (d): cập nhật theo thời gian (kịch bản GIẢ LẬP, xem data/scenarios/README.md)", "",
          summ.to_markdown(index=False), "", "## Nhật ký (có tìm lại đường)", "",
          live.events_table(ev).to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
