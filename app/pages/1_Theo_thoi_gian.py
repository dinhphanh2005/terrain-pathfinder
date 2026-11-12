"""Trang 2 của giao diện: lớp nguy hiểm theo thời gian, điểm ngập từ camera, tìm lại đường khi đang đi.

Streamlit tự nhận các file trong app/pages/ thành trang riêng (menu bên trái). Chỉ gọi src/live.py.
Hai chế độ (chọn ở thanh bên):
  - "Xem theo thời điểm": kéo thanh thời gian, xem lớp nguy hiểm và các quan sát đang hiệu lực.
  - "Mô phỏng đi bộ (live)": bấm bản đồ chọn điểm đi/đến, chạy mô phỏng, bấm ▶ Phát để xem từng bước.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import live  # noqa: E402
from src.layers import PLACES, load_region  # noqa: E402
from src.pathfinding import snap_to_free  # noqa: E402

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

st.set_page_config(page_title="Theo thời gian · Đề 26", layout="wide")
SCN = ROOT / "data" / "scenarios"
DISPLAY = 760
COL = {"river": (20, 60, 200), "river_flood": (90, 150, 255), "landslide_s2": (240, 100, 30),
       "camera_flood": (0, 215, 215), "landslide": (220, 0, 0), "blocked": (150, 0, 150), "forbidden": (120, 0, 120)}
LEGEND = ("Xanh nhạt: ngập do sông dâng h(t) · Xanh ngọc: ngập quanh camera · Đỏ: báo sạt lở · "
          "Tím: đường bị chặn · Cam: sạt lở từ Sentinel-2 · Chấm màu: vị trí quan sát đang hiệu lực.")


@st.cache_resource(show_spinner="Đang đọc các lớp bản đồ…")
def region_data():
    L, meta = load_region("vn")
    return L, meta, live.RiverFloodCache(L["dem"], L["river"] > 0.5)


def base_image(L, snap, show_forbidden):
    g = (np.clip(np.nan_to_num(L["hillshade"]), 0, 1) * 255).astype(np.float32)
    img = np.dstack([g, g, g])

    def over(mask, key, a):
        m = np.asarray(mask, bool)
        img[m] = img[m] * (1 - a) + np.array(COL[key], np.float32) * a
    if show_forbidden:
        over(snap.forbidden, "forbidden", 0.25)
    over(L["hazard_landslide"] > 0.5, "landslide_s2", 0.85)
    over(snap.parts["river_flood"], "river_flood", 0.6)
    over(L["river"] > 0.5, "river", 0.9)
    for k in ("camera_flood", "landslide", "blocked"):
        if k in snap.parts:
            over(snap.parts[k], k, 0.9)
    pil = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    return pil.resize((DISPLAY, int(L["dem"].shape[0] * DISPLAY / L["dem"].shape[1])), Image.LANCZOS)


def line(d, path, s, color, w=3):
    if path and len(path) > 1:
        d.line([(c * s, r * s) for r, c in path], fill=color, width=w)


def dot(d, cell, s, color, r=6):
    y, x = cell[0] * s, cell[1] * s
    d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255))


def mark_obs(d, act, s):
    for o in act.itertuples():
        dot(d, (o.row, o.col), s, COL[o.kind], 5)



@st.cache_data(show_spinner=False, max_entries=16)
def pick_base(t0, buffer_m, cam_r, show_forb, levels, obs):
    """Ảnh nền (mảng RGB) để bấm chọn điểm; cache theo tham số nên bấm chọn điểm không phải tính lại."""
    snap = live.snapshot(L, meta, t0, levels, obs, rf, buffer_m=buffer_m, camera_radius_m=cam_r)
    img = base_image(L, snap, show_forb)
    mark_obs(ImageDraw.Draw(img), snap.active, scale)
    return np.asarray(img)


def player_html(frames) -> str:
    """Trình phát các khung chạy bằng JavaScript trong trình duyệt: kéo thanh / phát không gửi gì về Python."""
    import base64
    import json
    data = [{"src": "data:image/jpeg;base64," + base64.b64encode(f["png"]).decode(),
             "t": f["time"].strftime("%d/%m %H:%M"), "a": f["action"], "h": round(f["level"], 2),
             "n": f["n_active"], "r": None if f["remaining"] != f["remaining"] else round(f["remaining"], 2)}
            for f in frames]
    return """
<style>
 body{margin:0;font-family:-apple-system,Segoe UI,Inter,sans-serif;color:#9aa0a6;font-size:14px}
 #stage{position:relative;width:100%;max-width:760px;aspect-ratio:var(--ar);border-radius:6px;overflow:hidden;background:#222}
 #stage img{position:absolute;inset:0;width:100%;height:100%;display:block;visibility:hidden}
 #stage img.on{visibility:visible}
 .bar{display:flex;gap:10px;align-items:center;margin:8px 0;max-width:760px}
 button{background:#ff4b4b;color:#fff;border:0;border-radius:6px;padding:6px 14px;font-size:14px;cursor:pointer}
 select{border-radius:6px;padding:4px}
 input[type=range]{flex:1;accent-color:#ff4b4b}
 .info{max-width:760px;display:flex;gap:18px;flex-wrap:wrap}
 .info b{color:#e8eaed}
 .warn b.act{color:#ffb020}
 @media (prefers-color-scheme: light){body{color:#5f6368} .info b{color:#1b1f24} .warn b.act{color:#d9480f}}  /* nền sáng */
</style>
<div id="stage"></div>
<div class="bar"><button id="pl">▶ Phát</button>
 <input id="sl" type="range" min="0" value="0">
 <select id="sp"><option value="300">nhanh</option><option value="700" selected>vừa</option><option value="1400">chậm</option></select></div>
<div class="info" id="info"></div>
<script>
const F = """ + json.dumps(data) + """;
const stage = document.getElementById('stage'), sl = document.getElementById('sl'), pl = document.getElementById('pl'),
      sp = document.getElementById('sp'), info = document.getElementById('info');
stage.style.setProperty('--ar', '""" + str(cols) + " / " + str(rows) + """');
// Tạo sẵn một <img> cho mỗi khung, chồng lên nhau; chuyển khung chỉ bật/tắt hiển thị
// -> không phải giải mã lại ảnh, khung không đổi kích thước nên trang không bị nhảy.
const IM = F.map(f => { const x = document.createElement('img'); x.src = f.src; stage.appendChild(x); return x; });
sl.max = F.length - 1; let k = 0, timer = null;
function show(i){ IM[k].classList.remove('on'); k = i; IM[i].classList.add('on'); const f = F[i]; sl.value = i;
  const hot = f.a.startsWith('tìm lại') || f.a.startsWith('kẹt');
  info.className = 'info' + (hot ? ' warn' : '');
  info.innerHTML = `<span>Khung <b>${i+1}/${F.length}</b></span><span>Thời điểm <b>${f.t}</b></span>`
    + `<span>Sự kiện <b class="act">${f.a}</b></span><span>Mực nước <b>${f.h} m</b></span>`
    + `<span>Quan sát hiệu lực <b>${f.n}</b></span><span>Còn lại <b>${f.r === null ? '—' : f.r + ' giờ'}</b></span>`; }
function stop(){ clearInterval(timer); timer = null; pl.textContent = '▶ Phát'; }
pl.onclick = () => { if (timer) { stop(); return; }
  if (k >= F.length - 1) show(0);
  pl.textContent = '⏸ Dừng';
  timer = setInterval(() => { if (k >= F.length - 1) { stop(); return; } show(k + 1); }, +sp.value); };
sp.onchange = () => { if (timer) { stop(); pl.click(); } };
sl.oninput = () => { stop(); show(+sl.value); };
show(0);
</script>"""


L, meta, rf = region_data()
rows, cols = L["dem"].shape
scale = DISPLAY / cols
a_km2 = float(meta["cell_size_m"]) ** 2 / 1e6

# ---------------------------------------------------------------- thanh bên
st.sidebar.title("Theo thời gian")
mode = st.sidebar.radio("Chế độ", ["Mô phỏng đi bộ (live)", "Xem theo thời điểm"])
st.sidebar.caption("Chỉ cho vùng Việt Nam. Kịch bản mẫu là **giả lập minh họa** (xem data/scenarios/README.md).")
up = st.sidebar.file_uploader("CSV quan sát khác (time, kind, lat, lon, value, ttl_h)", type="csv")
try:
    obs = live.load_observations(up if up is not None else SCN / "yagi_demo_observations.csv", meta)
except Exception as e:  # CSV người dùng sai định dạng
    st.sidebar.error(f"CSV không hợp lệ, dùng kịch bản mẫu: {e}")
    obs = live.load_observations(SCN / "yagi_demo_observations.csv", meta)
if (~obs["inside"]).any():
    st.sidebar.warning(f"{int((~obs['inside']).sum())} quan sát nằm ngoài vùng, bị bỏ qua.")
with st.sidebar.expander("Mực nước sông dâng h(t) (sửa được)"):
    levels = st.data_editor(pd.read_csv(SCN / "yagi_demo_levels.csv"), num_rows="dynamic", hide_index=True)
buffer_m = st.sidebar.slider("Bán kính vùng đệm (m)", 0, 150, 60, step=30)
cam_r = st.sidebar.slider("Bán kính tối đa vùng ngập quanh camera (m)", 150, 1500, 600, step=150)
show_forb = st.sidebar.checkbox("Tô vùng cấm (gồm vùng đệm)", False)
KW = {"buffer_m": buffer_m, "camera_radius_m": cam_r}

times = pd.concat([pd.to_datetime(levels["time"]), obs["time"]])
t_min, t_max = times.min().to_pydatetime(), times.max().to_pydatetime()

# ================================================================ chế độ 1: xem theo thời điểm
if mode == "Xem theo thời điểm":
    st.title("Lớp nguy hiểm theo thời điểm")
    t = st.slider("Thời điểm", min_value=t_min, max_value=t_max, value=dt.datetime(2024, 9, 11, 8, 0),
                  step=dt.timedelta(hours=1), format="DD/MM HH:mm")
    snap = live.snapshot(L, meta, t, levels, obs, rf, **KW)
    left, right = st.columns([3, 2])
    with left:
        img = base_image(L, snap, show_forb)
        mark_obs(ImageDraw.Draw(img), snap.active, scale)
        st.image(img)
        st.caption(LEGEND)
    with right:
        c1, c2 = st.columns(2)
        c1.metric("Mực nước dâng h(t)", f"{snap.level_m:.2f} m")
        c2.metric("Quan sát đang hiệu lực", len(snap.active))
        c1.metric("Ngập do sông", f"{snap.parts['river_flood'].sum() * a_km2:.1f} km²")
        c2.metric("Ngập quanh camera", f"{snap.parts.get('camera_flood', np.zeros(1)).sum() * a_km2:.2f} km²")
        st.subheader("Quan sát")
        show = obs[["time", "kind", "value", "ttl_h", "note"]].copy()
        show["đang hiệu lực"] = obs.index.isin(snap.active.index)
        st.dataframe(show, hide_index=True)
    st.stop()

# ================================================================ chế độ 2: mô phỏng đi bộ (live)
st.title("Mô phỏng đi bộ có cập nhật (live)")
places = {**PLACES["vn"], "Làng Nủ": (665, 829)}
if "live_pts" not in st.session_state:
    st.session_state.live_pts = [places["Phố Ràng"], places["Làng Nủ"]]

# Cú bấm bản đồ ở lần chạy trước nằm sẵn trong session_state["live_map"] -> xử lý ngay ở đầu,
# trước khi vẽ, nên chấm mới hiện luôn mà không phải chạy lại trang lần nữa (bỏ st.rerun gây giật).
_click = st.session_state.get("live_map")
if _click and _click != st.session_state.get("live_click") and "sim" not in st.session_state:
    st.session_state.live_click = _click
    sx = cols / _click["width"] if _click.get("width") else 1.0 / scale
    sy = rows / _click["height"] if _click.get("height") else 1.0 / scale
    _p = (min(rows - 1, int(_click["y"] * sy)), min(cols - 1, int(_click["x"] * sx)))
    _cur = st.session_state.live_pts
    st.session_state.live_pts = [_p] if len(_cur) != 1 else [_cur[0], _p]
PLAYER_H = int(rows * scale) + 110   # bản đồ + thanh điều khiển + dòng thông tin
MAP_H = int(rows * scale) + 8   # chiều cao cố định cho khung bản đồ -> trang không co giãn khi vẽ lại

c1, c2, c3 = st.columns(3)
day = c1.date_input("Ngày xuất phát", dt.date(2024, 9, 11))
hour = c2.time_input("Giờ xuất phát", dt.time(6, 0))
upd = c3.select_slider("Cập nhật mỗi (phút đi bộ)", [10, 20, 30, 60, 90, 120], 30)
t0 = pd.Timestamp(dt.datetime.combine(day, hour))

left, right = st.columns([3, 2])
map_slot = left.empty()
cap_slot = left.empty()

with right:
    st.markdown("**Chọn điểm:** bấm lên bản đồ (lần 1: điểm đi, lần 2: điểm đến, lần 3: chọn lại), "
                "hoặc chọn địa danh:")
    p1, p2 = st.columns(2)
    a = p1.selectbox("Điểm đi", ["—"] + list(places))
    b = p2.selectbox("Điểm đến", ["—"] + list(places))
    if st.button("Dùng hai địa danh này") and a != "—" and b != "—":
        st.session_state.live_pts = [places[a], places[b]]
        st.session_state.pop("sim", None)
        st.rerun()
    pts = st.session_state.live_pts
    st.caption("Điểm đi: " + (f"ô {pts[0]}" if len(pts) > 0 else "chưa chọn") + " · Điểm đến: "
               + (f"ô {pts[1]}" if len(pts) > 1 else "chưa chọn"))
    run = st.button("Chạy mô phỏng", type="primary", disabled=len(pts) < 2)
    status = st.empty()
    table_slot = st.empty()

if run:
    snap0 = live.snapshot(L, meta, t0, levels, obs, rf, **KW)
    s = snap_to_free(snap0.cost, pts[0]); g = snap_to_free(snap0.cost, pts[1])
    if s is None or g is None:
        status.error("Điểm chọn nằm sâu trong vùng cấm ở thời điểm xuất phát.")
    else:
        res = {}
        with st.spinner("Đang mô phỏng…"):
            for name, rp in (("Có tìm lại đường", True), ("Giữ tuyến ban đầu", False)):
                res[name] = live.simulate_walk(L, meta, s, g, t0, levels, obs, update_min=upd, replan=rp,
                                               river_flood=rf, **KW)
        st.session_state.sim = {"t0": t0, "res": res, "goal": g}
        st.session_state.frame = len(res["Có tìm lại đường"]) - 1
        st.rerun()

sim = st.session_state.get("sim")


def frame_image(k: int):
    """Ảnh bản đồ tại sự kiện thứ k của chiến lược 'Có tìm lại đường'."""
    ev = sim["res"]["Có tìm lại đường"]
    e = ev[k]
    snap = live.snapshot(L, meta, e.time, levels, obs, rf, **KW)
    img = base_image(L, snap, show_forb)
    d = ImageDraw.Draw(img)
    line(d, ev[0].path, scale, (255, 255, 255), 1)                       # tuyến ban đầu
    walked = live.walked_cells(ev[:k + 1])
    plan = next((x.path for x in reversed(ev[:k + 1]) if x.path), [])     # tuyến đang theo
    if plan and e.cell in plan:
        line(d, plan[plan.index(e.cell):], scale, (255, 120, 230), 2)     # phần còn lại dự kiến
    line(d, walked, scale, (255, 0, 200), 4)                             # đã đi
    for x in ev[:k + 1]:
        if x.action == "tìm lại đường":
            dot(d, x.cell, scale, (255, 255, 255), 5)
    mark_obs(d, snap.active, scale)
    dot(d, ev[0].cell, scale, (0, 200, 0)); dot(d, sim["goal"], scale, (220, 0, 0))
    dot(d, e.cell, scale, (255, 230, 0), 8)                              # vị trí hiện tại
    return img, e, snap


def build_frames():
    """Dựng sẵn mọi khung (ảnh + số liệu) một lần sau khi mô phỏng, để kéo thanh / phát không phải tính lại."""
    import io
    ev = sim["res"]["Có tìm lại đường"]
    out = []
    bar = st.progress(0.0, text="Đang dựng các khung…")
    for k in range(len(ev)):
        img, e, snap = frame_image(k)
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=88)  # JPEG nhẹ hơn PNG ~5 lần, gửi lên trình duyệt nhanh hơn
        out.append({"png": buf.getvalue(), "time": e.time, "action": e.action, "level": snap.level_m,
                    "n_active": len(snap.active), "remaining": e.remaining_h})
        bar.progress((k + 1) / len(ev), text=f"Đang dựng khung {k + 1}/{len(ev)}…")
    bar.empty()
    return out


if sim is None:
    # chưa mô phỏng: hiện bản đồ để bấm chọn điểm (ảnh nền được cache nên mỗi lần bấm chỉ vẽ lại 2 chấm)
    base = pick_base(t0, buffer_m, cam_r, show_forb, levels, obs)
    img = Image.fromarray(base.copy())
    d = ImageDraw.Draw(img)
    for i, p in enumerate(st.session_state.live_pts):
        dot(d, p, scale, (0, 200, 0) if i == 0 else (220, 0, 0))
    with map_slot.container(height=MAP_H, border=False):
        if streamlit_image_coordinates is not None:
            streamlit_image_coordinates(img, key="live_map", use_column_width="always")  # cú bấm xử lý ở đầu lần chạy sau
        else:
            st.image(img)
    cap_slot.caption(f"Lớp nguy hiểm lúc xuất phát {t0:%d/%m %H:%M}. " + LEGEND)
    status.info("Chọn điểm rồi bấm **Chạy mô phỏng**.")
else:
    ev = sim["res"]["Có tìm lại đường"]
    if ev[0].action == "kẹt":
        status.warning("Không có đường ngay lúc xuất phát: vùng cấm chia cắt điểm đi và điểm đến ở thời điểm này.")
    else:
        key = (show_forb, buffer_m, cam_r)
        if sim.get("frames_key") != key:
            sim["frames"] = build_frames()
            sim["frames_key"] = key
        with map_slot.container(height=PLAYER_H, border=False):
            html = player_html(sim["frames"])
            if hasattr(st, "iframe"):          # Streamlit mới: st.iframe thay cho components.html
                st.iframe(html, height=PLAYER_H - 8)
            else:
                components.html(html, height=PLAYER_H - 8, scrolling=False)
        cap_slot.caption("Vàng: vị trí hiện tại · Hồng đậm: đã đi · Hồng nhạt: phần còn lại dự kiến · "
                         "Trắng mảnh: tuyến ban đầu · Chấm trắng: chỗ tìm lại đường. " + LEGEND)
        status.info("Bấm **▶ Phát** hoặc kéo thanh ngay dưới bản đồ để xem từng khung.")
        with table_slot.container():
            for name, e in sim["res"].items():
                last = e[-1]
                st.metric(name, ("Tới nơi " if last.action == "đến nơi" else "Kẹt ") + last.time.strftime("%d/%m %H:%M"),
                          f"{(last.time - sim['t0']).total_seconds() / 3600:.2f} giờ đi bộ", delta_color="off")
            st.dataframe(live.events_table(ev), hide_index=True, height=300)
            if st.button("Chọn lại điểm"):
                st.session_state.pop("sim", None); st.session_state.live_pts = []
                st.rerun()
