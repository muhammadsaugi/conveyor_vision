# ============================================================
# CONVEYOR VISION SYSTEM — Streamlit App
# Stack: streamlit==1.44.0, streamlit-webrtc==0.64.6,
#        aiortc==1.14.0, av==16.1.0, Python 3.14
# ============================================================

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ.setdefault("DISPLAY", "")

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from threading import Lock

import streamlit as st

st.set_page_config(
    page_title="Conveyor Vision System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Safe imports ─────────────────────────────────────────────
try:
    import cv2
except ImportError as e:
    st.error(f"❌ OpenCV error: {e}")
    st.stop()

try:
    import numpy as np
except ImportError as e:
    st.error(f"❌ NumPy error: {e}")
    st.stop()

try:
    import pandas as pd
except ImportError as e:
    st.error(f"❌ Pandas error: {e}")
    st.stop()

try:
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# =============================================================
# KONFIGURASI
# =============================================================
MODEL_PATH  = Path("best.pt")
CLASS_NAMES = ["Good Day", "Nutrisari"]
LOG_DIR     = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

COLORS_BGR = {
    "Good Day"       : (34,  139,  34),
    "Nutrisari"      : (205,  90,  55),
    "Tidak Dikenali" : (0,     0, 200),
}
COLORS_HEX = {
    "Good Day"       : "#228B22",
    "Nutrisari"      : "#CD5A37",
    "Tidak Dikenali" : "#C80000",
}

# =============================================================
# SESSION STATE
# =============================================================
def init_state():
    defaults = {
        "session_id"   : datetime.now().strftime("%Y%m%d_%H%M%S"),
        "count"        : {"Good Day": 0, "Nutrisari": 0, "Tidak Dikenali": 0},
        "counted_ids"  : set(),
        "prev_cy"      : {},
        "detections"   : [],
        "alert_active" : False,
        "alert_until"  : 0.0,
        "frame_count"  : 0,
        "_conf_val"    : 0.50,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================================
# LOAD MODEL
# =============================================================
@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_model(path: str):
    if not Path(path).exists():
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

# =============================================================
# KLASIFIKASI 3 KELAS
# =============================================================
def classify(cls_id: int, conf: float, thresh: float) -> str:
    if conf < thresh:
        return "Tidak Dikenali"
    if 0 <= cls_id < len(CLASS_NAMES):
        return CLASS_NAMES[cls_id]
    return "Tidak Dikenali"

# =============================================================
# ANNOTASI FRAME
# =============================================================
def annotate_frame(frame, results, thresh, line_y, state):
    h, w = frame.shape[:2]
    now  = time.time()

    cv2.line(frame, (0, line_y), (w, line_y), (0, 220, 255), 2)
    cv2.putText(frame, "COUNTING LINE", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    if state["alert_active"] and now < state["alert_until"]:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
        txt = "! TIDAK DIKENALI !"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.1, 3)
        cv2.putText(frame, txt, ((w-tw)//2, h//2+th//2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        state["alert_active"] = False

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            conf   = float(box.conf.item())
            cls_id = int(box.cls.item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            tid = int(box.id.item()) if box.id is not None else None

            label = classify(cls_id, conf, thresh)
            color = COLORS_BGR.get(label, (128,128,128))

            if tid is not None and tid in state["prev_cy"]:
                prev = state["prev_cy"][tid]
                if prev < line_y <= cy and tid not in state["counted_ids"]:
                    state["counted_ids"].add(tid)
                    state["count"][label] += 1
                    state["detections"].append({
                        "id"        : len(state["detections"]) + 1,
                        "track_id"  : tid,
                        "label"     : label,
                        "confidence": round(conf, 4),
                        "timestamp" : datetime.now().isoformat(),
                        "bbox"      : [x1,y1,x2,y2],
                    })
                    if label == "Tidak Dikenali":
                        state["alert_active"] = True
                        state["alert_until"]  = now + 2.5

            if tid is not None:
                state["prev_cy"][tid] = cy

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3 if label=="Tidak Dikenali" else 2)
            txt = f"ID:{tid} {label} {conf:.0%}" if tid else f"{label} {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1,y1-th-10), (x1+tw+6,y1), color, -1)
            cv2.putText(frame, txt, (x1+3,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx,cy), 5, color, -1)

    ov2 = frame.copy()
    cv2.rectangle(ov2, (0,0), (250,100), (0,0,0), -1)
    cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
    hud = [("Good Day",state["count"]["Good Day"],(100,220,100)),
           ("Nutrisari",state["count"]["Nutrisari"],(220,150,80)),
           ("Tdk Dikenali",state["count"]["Tidak Dikenali"],(80,80,220))]
    for i,(lbl,cnt,clr) in enumerate(hud):
        cv2.putText(frame, f"{lbl:<14}: {cnt:>3}", (8,22+i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, clr, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{'Total':<14}: {sum(state['count'].values()):>3}",
                (8,22+3*26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1, cv2.LINE_AA)

    state["frame_count"] += 1
    return frame

# =============================================================
# WEBRTC PROCESSOR
# =============================================================
class VideoProcessor:
    def __init__(self):
        self.lock        = Lock()
        self.model       = None
        self.conf_thresh = 0.50
        self.line_ratio  = 0.55
        self.tracking_on = True

    def recv(self, frame):
        img    = frame.to_ndarray(format="bgr24")
        h, w   = img.shape[:2]
        line_y = int(h * self.line_ratio)

        with self.lock:
            model    = self.model
            thresh   = self.conf_thresh
            track_on = self.tracking_on

        if model is None:
            cv2.putText(img, "Model belum dimuat...", (20, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,140,255), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            if track_on:
                results = model.track(img, persist=True, conf=max(0.15,thresh-0.15),
                                      iou=0.45, tracker="bytetrack.yaml", verbose=False)
            else:
                results = model(img, conf=thresh, iou=0.45, verbose=False)
            out = annotate_frame(img, results, thresh, line_y, st.session_state)
        except Exception as e:
            cv2.putText(img, f"Err:{str(e)[:40]}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            out = img

        return av.VideoFrame.from_ndarray(out, format="bgr24")

# =============================================================
# HELPERS
# =============================================================
def save_log():
    sid = st.session_state["session_id"]
    lp  = LOG_DIR / f"detection_{sid}.json"
    sp  = LOG_DIR / f"summary_{sid}.json"
    with open(lp, "w", encoding="utf-8") as f:
        json.dump({"session_id":sid,"saved_at":datetime.now().isoformat(),
                   "class_names":CLASS_NAMES,"detections":st.session_state["detections"]},
                  f, indent=2, ensure_ascii=False)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"session_id":sid,"last_updated":datetime.now().isoformat(),
                   "counters":{"good_day":st.session_state["count"]["Good Day"],
                               "nutrisari":st.session_state["count"]["Nutrisari"],
                               "tidak_dikenali":st.session_state["count"]["Tidak Dikenali"]},
                   "total":sum(st.session_state["count"].values()),
                   "frames":st.session_state["frame_count"]},
                  f, indent=2, ensure_ascii=False)
    return lp, sp

def reset_counters():
    st.session_state.update({
        "session_id":"" , "count":{"Good Day":0,"Nutrisari":0,"Tidak Dikenali":0},
        "counted_ids":set(), "prev_cy":{}, "detections":[],
        "alert_active":False, "alert_until":0.0, "frame_count":0,
    })
    st.session_state["session_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.title("⚙️ Pengaturan")
    st.divider()

    st.subheader("📦 Status Model")
    if not YOLO_AVAILABLE:
        st.error("❌ ultralytics tidak terinstall")
    elif MODEL_PATH.exists():
        sz = MODEL_PATH.stat().st_size/1024/1024
        st.success(f"✅ best.pt ({sz:.1f} MB)")
    else:
        st.error("❌ best.pt tidak ditemukan!")
        st.info("Letakkan `best.pt` di root folder yang sama dengan `app.py`")

    if not WEBRTC_AVAILABLE:
        st.warning("⚠️ WebRTC tidak tersedia")

    st.divider()
    st.subheader("🎯 Confidence Threshold")
    conf_val = st.slider("Threshold", 0.10, 0.90, 0.50, 0.05)
    st.session_state["_conf_val"] = conf_val
    st.caption(f"≥ **{conf_val:.0%}** → Good Day / Nutrisari  \n< **{conf_val:.0%}** → Tidak Dikenali")

    st.divider()
    st.subheader("📏 Counting Line")
    line_ratio = st.slider("Posisi garis (%)", 0.30, 0.80, 0.55, 0.05)

    st.divider()
    st.subheader("🔍 Tracking")
    tracking_on = st.toggle("Aktifkan ByteTrack", value=True)

    st.divider()
    st.subheader("🎛️ Kontrol")
    if st.button("🔄 Reset Counter", use_container_width=True, type="primary"):
        reset_counters()
        st.success("Counter direset!")
        st.rerun()

    if st.button("💾 Simpan Log", use_container_width=True):
        if st.session_state["detections"]:
            lp, _ = save_log()
            st.success(f"✅ {lp.name}")
        else:
            st.info("Belum ada deteksi.")

    st.divider()
    st.subheader("📥 Download")
    if st.session_state["detections"]:
        df_dl = pd.DataFrame(st.session_state["detections"])
        st.download_button("⬇️ Download CSV", data=df_dl.to_csv(index=False),
                           file_name=f"detection_{st.session_state['session_id']}.csv",
                           mime="text/csv", use_container_width=True)
    else:
        st.caption("Belum ada data.")

    st.divider()
    st.caption(f"**Session:** `{st.session_state['session_id']}`  \n"
               f"**Frames:** {st.session_state['frame_count']:,}  \n"
               f"**Stack:** st 1.57 · webrtc 0.64 · av 16")

# =============================================================
# MAIN UI
# =============================================================
st.markdown("<h2 style='margin-bottom:2px'>🏭 Conveyor Vision System</h2>"
            "<p style='color:gray;margin-top:0'>Deteksi real-time Good Day &amp; Nutrisari</p>",
            unsafe_allow_html=True)
st.divider()

cnt   = st.session_state["count"]
total = sum(cnt.values())
alert_now = (st.session_state["alert_active"] and time.time() < st.session_state["alert_until"])

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("☕ Good Day",      cnt["Good Day"])
c2.metric("🍊 Nutrisari",     cnt["Nutrisari"])
c3.metric("⚠️ Tdk Dikenali", cnt["Tidak Dikenali"])
c4.metric("📦 Total",         total)
with c5:
    if alert_now:              st.error("🚨 ALERT!")
    elif cnt["Tidak Dikenali"]: st.warning(f"⚠️ {cnt['Tidak Dikenali']} tdk dikenali")
    else:                       st.success("✅ Normal")

st.divider()
st.subheader("📹 Video Real-Time")

if not WEBRTC_AVAILABLE:
    st.error("❌ streamlit-webrtc tidak tersedia. Cek requirements.txt")
elif not YOLO_AVAILABLE:
    st.error("❌ ultralytics tidak tersedia.")
else:
    model_obj = load_model(str(MODEL_PATH))
    processor = VideoProcessor()
    if model_obj:
        with processor.lock:
            processor.model       = model_obj
            processor.conf_thresh = conf_val
            processor.line_ratio  = line_ratio
            processor.tracking_on = tracking_on

    RTC_CONFIG = RTCConfiguration({"iceServers":[
        {"urls":["stun:stun.l.google.com:19302"]},
        {"urls":["stun:stun1.l.google.com:19302"]},
    ]})

    ctx = webrtc_streamer(
        key="conveyor-vision",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: processor,
        media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},
                                            "frameRate":{"ideal":20,"max":30}},"audio":False},
        async_processing=True,
    )

    if ctx.video_processor:
        with ctx.video_processor.lock:
            ctx.video_processor.conf_thresh  = conf_val
            ctx.video_processor.line_ratio   = line_ratio
            ctx.video_processor.tracking_on  = tracking_on
            if model_obj:
                ctx.video_processor.model    = model_obj

    if not ctx.state.playing:
        st.info("👆 Klik **START** → izinkan kamera → arahkan ke objek di conveyor.  \nGunakan **Chrome** atau **Edge**.")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Statistik", "📋 Log Deteksi", "❓ Panduan"])

with tab1:
    if total > 0:
        cl, cr = st.columns(2)
        with cl:
            st.markdown("#### Distribusi Kelas")
            df_d = pd.DataFrame({"Kelas":list(cnt.keys()),"Jumlah":list(cnt.values()),
                                  "Persen":[f"{v/total*100:.1f}%" for v in cnt.values()]})
            st.dataframe(df_d, hide_index=True, use_container_width=True)
            for lbl,val in cnt.items():
                pct = val/total if total>0 else 0
                clr = COLORS_HEX[lbl]
                st.markdown(f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                            f"<span style='width:110px;font-size:13px'>{lbl}</span>"
                            f"<div style='flex:1;background:#333;border-radius:4px;height:14px'>"
                            f"<div style='width:{pct*100:.1f}%;background:{clr};border-radius:4px;height:14px'>"
                            f"</div></div><span style='font-size:13px'>{val}</span></div>",
                            unsafe_allow_html=True)
        with cr:
            st.markdown("#### 10 Deteksi Terakhir")
            if st.session_state["detections"]:
                df_r = pd.DataFrame(st.session_state["detections"]).tail(10)
                st.dataframe(df_r[["id","label","confidence","timestamp"]],
                             hide_index=True, use_container_width=True)
    else:
        st.info("Belum ada deteksi. Aktifkan kamera dan arahkan ke objek.")

with tab2:
    if st.session_state["detections"]:
        df_log = pd.DataFrame(st.session_state["detections"])
        fa, fb = st.columns([2,1])
        with fa:
            f_labels = st.multiselect("Filter kelas",
                ["Good Day","Nutrisari","Tidak Dikenali"],
                default=["Good Day","Nutrisari","Tidak Dikenali"])
        with fb:
            n_show = st.selectbox("Tampilkan",[10,25,50,100,"Semua"],index=1)
        df_f = df_log[df_log["label"].isin(f_labels)]
        if n_show != "Semua":
            df_f = df_f.tail(int(n_show))
        st.dataframe(df_f[["id","track_id","label","confidence","timestamp"]],
                     hide_index=True, use_container_width=True,
                     column_config={"confidence":st.column_config.ProgressColumn(
                         "Confidence",min_value=0,max_value=1,format="%.2f")})
        st.caption(f"Total {len(df_log)} event tercatat.")
    else:
        st.info("Belum ada deteksi tercatat.")

with tab3:
    st.markdown("""
### 📌 Cara Menggunakan
1. Pastikan **`best.pt`** ada di folder yang sama dengan `app.py`
2. Klik **▶ START** di panel video
3. Izinkan akses kamera saat browser meminta
4. Arahkan kamera ke objek di conveyor
5. Objek melewati **garis kuning** → dihitung otomatis

### 🎯 Logika 3 Kelas
| Kondisi | Label | Warna |
|---|---|---|
| conf ≥ threshold & class 0 | **Good Day** | 🟢 Hijau |
| conf ≥ threshold & class 1 | **Nutrisari** | 🟠 Oranye |
| conf < threshold | **Tidak Dikenali** | 🔴 Merah |

### ⚙️ Tips
- **FPS lambat?** → Matikan ByteTrack di sidebar
- **Banyak Tidak Dikenali?** → Turunkan threshold ke 0.35
- **Kamera tidak muncul?** → Gunakan Chrome/Edge
    """)