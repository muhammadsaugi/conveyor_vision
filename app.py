# ============================================================
# CONVEYOR VISION SYSTEM — Streamlit App
# Arsitektur: Dual Mode
#   Mode 1 (LIVE): WebRTC dengan STUN+TURN server
#   Mode 2 (SNAPSHOT): Upload/capture foto → deteksi instant
# Stack: streamlit==1.57.0, Python 3.14
# ============================================================

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ.setdefault("DISPLAY", "")

import sys, json, time
from pathlib import Path
from datetime import datetime
from threading import Lock
import base64
import io

import streamlit as st

st.set_page_config(
    page_title="Conveyor Vision System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports dengan error handling ────────────────────────────
try:
    import cv2
    CV2_OK = True
except Exception as e:
    CV2_OK = False
    st.error(f"❌ OpenCV: {e}")
    st.stop()

import numpy as np
import pandas as pd
from PIL import Image

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_OK = True
except Exception as e:
    WEBRTC_OK = False
    _WEBRTC_ERR = str(e)

try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception as e:
    YOLO_OK = False
    _YOLO_ERR = str(e)

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

# TURN server publik gratis (fallback jika STUN saja tidak cukup)
RTC_CONFIG = RTCConfiguration({"iceServers": [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
    {"urls": ["stun:stun2.l.google.com:19302"]},
    {"urls": ["stun:stun3.l.google.com:19302"]},
    # TURN server publik — membantu di jaringan yang blokir STUN
    {
        "urls": ["turn:openrelay.metered.ca:80"],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
    {
        "urls": ["turn:openrelay.metered.ca:443"],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
    {
        "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
]}) if WEBRTC_OK else None

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
        "snap_results" : [],   # hasil snapshot mode
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================================
# LOAD MODEL
# =============================================================
@st.cache_resource(show_spinner="⏳ Memuat model YOLOv8...")
def load_model(path: str):
    if not Path(path).exists():
        return None
    try:
        return YOLO(path)
    except Exception as e:
        return None

# =============================================================
# KLASIFIKASI 3 KELAS
# =============================================================
def classify(cls_id: int, conf: float, thresh: float) -> str:
    if conf < thresh:
        return "Tidak Dikenali"
    return CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "Tidak Dikenali"

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
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,255), 3, cv2.LINE_AA)
    else:
        state["alert_active"] = False

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            conf   = float(box.conf.item())
            cls_id = int(box.cls.item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            tid = int(box.id.item()) if box.id is not None else None

            label = classify(cls_id, conf, thresh)
            color = COLORS_BGR.get(label, (128,128,128))

            if tid is not None and tid in state["prev_cy"]:
                prev = state["prev_cy"][tid]
                if prev < line_y <= cy and tid not in state["counted_ids"]:
                    state["counted_ids"].add(tid)
                    state["count"][label] += 1
                    state["detections"].append({
                        "id": len(state["detections"])+1,
                        "track_id": tid, "label": label,
                        "confidence": round(conf,4),
                        "timestamp": datetime.now().isoformat(),
                        "bbox": [x1,y1,x2,y2],
                    })
                    if label == "Tidak Dikenali":
                        state["alert_active"] = True
                        state["alert_until"]  = now + 2.5

            if tid is not None:
                state["prev_cy"][tid] = cy

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3 if label=="Tidak Dikenali" else 2)
            txt = f"ID:{tid} {label} {conf:.0%}" if tid else f"{label} {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+6,y1),color,-1)
            cv2.putText(frame,txt,(x1+3,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),5,color,-1)

    hud = [("Good Day",state["count"]["Good Day"],(100,220,100)),
           ("Nutrisari",state["count"]["Nutrisari"],(220,150,80)),
           ("Tdk Dikenali",state["count"]["Tidak Dikenali"],(80,80,220))]
    ov2 = frame.copy()
    cv2.rectangle(ov2,(0,0),(250,100),(0,0,0),-1)
    cv2.addWeighted(ov2,0.55,frame,0.45,0,frame)
    for i,(lbl,cnt,clr) in enumerate(hud):
        cv2.putText(frame,f"{lbl:<14}: {cnt:>3}",(8,22+i*26),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,clr,1,cv2.LINE_AA)
    cv2.putText(frame,f"{'Total':<14}: {sum(state['count'].values()):>3}",
                (8,22+3*26),cv2.FONT_HERSHEY_SIMPLEX,0.52,(200,200,200),1,cv2.LINE_AA)
    state["frame_count"] += 1
    return frame

# =============================================================
# WEBRTC PROCESSOR
# =============================================================
if WEBRTC_OK:
    class VideoProcessor:
        def __init__(self):
            self.lock = Lock()
            self.model = None
            self.conf_thresh = 0.50
            self.line_ratio  = 0.55
            self.tracking_on = True

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
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
                    results = model.track(img, persist=True,
                                          conf=max(0.15, thresh-0.15),
                                          iou=0.45, tracker="bytetrack.yaml",
                                          verbose=False)
                else:
                    results = model(img, conf=thresh, iou=0.45, verbose=False)
                out = annotate_frame(img, results, thresh, line_y, st.session_state)
            except Exception as e:
                cv2.putText(img, f"Err:{str(e)[:40]}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                out = img
            return av.VideoFrame.from_ndarray(out, format="bgr24")

# =============================================================
# SNAPSHOT DETECTION (Mode 2 — tanpa WebRTC)
# =============================================================
def detect_snapshot(img_pil, model, thresh):
    """Deteksi pada gambar statis dari upload/capture."""
    img_np  = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (640, 640))
    h, w    = img_bgr.shape[:2]

    results = model(img_bgr, conf=thresh, iou=0.45, verbose=False)

    detections = []
    for box in results[0].boxes:
        conf   = float(box.conf.item())
        cls_id = int(box.cls.item())
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        label  = classify(cls_id, conf, thresh)
        color  = COLORS_BGR.get(label, (128,128,128))

        cv2.rectangle(img_bgr,(x1,y1),(x2,y2),color,3)
        txt = f"{label} {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        cv2.rectangle(img_bgr,(x1,y1-th-12),(x1+tw+6,y1),color,-1)
        cv2.putText(img_bgr,txt,(x1+3,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
        detections.append({"label": label, "confidence": round(conf,4),
                            "bbox": [x1,y1,x2,y2]})

    img_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_out, detections

# =============================================================
# HELPERS
# =============================================================
def reset_counters():
    st.session_state.update({
        "session_id":"", "count":{"Good Day":0,"Nutrisari":0,"Tidak Dikenali":0},
        "counted_ids":set(),"prev_cy":{},"detections":[],
        "alert_active":False,"alert_until":0.0,"frame_count":0,"snap_results":[],
    })
    st.session_state["session_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.title("⚙️ Pengaturan")
    st.divider()

    st.subheader("📦 Status Model")
    if not YOLO_OK:
        st.error(f"❌ YOLO: {_YOLO_ERR}")
    elif MODEL_PATH.exists():
        sz = MODEL_PATH.stat().st_size/1024/1024
        st.success(f"✅ best.pt ({sz:.1f} MB)")
    else:
        st.error("❌ best.pt tidak ditemukan!")

    webrtc_status = "✅ Tersedia" if WEBRTC_OK else f"⚠️ {_WEBRTC_ERR if not WEBRTC_OK else ''}"
    st.caption(f"WebRTC: {webrtc_status}")

    st.divider()
    st.subheader("🎯 Confidence Threshold")
    conf_val = st.slider("Threshold", 0.10, 0.90, 0.40, 0.05)
    st.caption(f"≥ **{conf_val:.0%}** → Good Day / Nutrisari  \n< **{conf_val:.0%}** → Tidak Dikenali")

    st.divider()
    st.subheader("📏 Counting Line (Mode LIVE)")
    line_ratio = st.slider("Posisi garis (%)", 0.30, 0.80, 0.55, 0.05)

    st.divider()
    st.subheader("🔍 Tracking")
    tracking_on = st.toggle("ByteTrack", value=False,
                            help="Matikan untuk performa lebih baik di cloud")

    st.divider()
    st.subheader("🎛️ Kontrol")
    if st.button("🔄 Reset Counter", use_container_width=True, type="primary"):
        reset_counters()
        st.success("Reset!")
        st.rerun()

    st.divider()
    cnt = st.session_state["count"]
    if st.session_state["detections"]:
        df_dl = pd.DataFrame(st.session_state["detections"])
        st.download_button("⬇️ Download CSV",
                           data=df_dl.to_csv(index=False),
                           file_name=f"detection_{st.session_state['session_id']}.csv",
                           mime="text/csv", use_container_width=True)

# =============================================================
# HEADER
# =============================================================
st.markdown(
    "<h2 style='margin-bottom:2px'>🏭 Conveyor Vision System</h2>"
    "<p style='color:gray;margin-top:0'>Deteksi Good Day &amp; Nutrisari — 3 Kelas</p>",
    unsafe_allow_html=True)
st.divider()

# Metric cards
total = sum(cnt.values())
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("☕ Good Day",      cnt["Good Day"])
c2.metric("🍊 Nutrisari",     cnt["Nutrisari"])
c3.metric("⚠️ Tdk Dikenali", cnt["Tidak Dikenali"])
c4.metric("📦 Total",         total)
with c5:
    alert_now = st.session_state["alert_active"] and time.time() < st.session_state["alert_until"]
    if alert_now: st.error("🚨 ALERT!")
    elif cnt["Tidak Dikenali"]: st.warning(f"⚠️ {cnt['Tidak Dikenali']}")
    else: st.success("✅ Normal")

st.divider()

# =============================================================
# MODE SELECTOR
# =============================================================
if not YOLO_OK:
    st.error(f"❌ Model tidak tersedia.")
else:
    model_obj = load_model(str(MODEL_PATH))

    mode = st.radio(
        "**Pilih Mode Kamera:**",
        ["📸 Snapshot (Stabil, cocok HP/Laptop)",
         "📹 Live Stream (WebRTC, butuh jaringan stabil)"],
        horizontal=True,
        help="Snapshot lebih ringan dan stabil. Live Stream butuh koneksi WebRTC yang baik."
    )

    st.divider()

    # ===========================================================
    # MODE 1: SNAPSHOT
    # ===========================================================
    if "Snapshot" in mode:
        st.subheader("📸 Mode Snapshot")
        st.info(
            "**Cara pakai:**  \n"
            "1. Arahkan kamera HP ke objek  \n"
            "2. Ambil foto / upload gambar  \n"
            "3. Sistem langsung mendeteksi objek  \n"
            "✅ **Mode ini lebih stabil dan tidak butuh WebRTC**"
        )

        col_cam, col_result = st.columns([1, 1])

        with col_cam:
            # Camera input — langsung dari browser tanpa WebRTC
            img_file = st.camera_input(
                "📷 Ambil foto objek",
                help="Klik untuk mengaktifkan kamera dan ambil foto"
            )

            # Atau upload gambar
            st.caption("— atau upload gambar —")
            uploaded = st.file_uploader(
                "Upload gambar",
                type=["jpg","jpeg","png","webp"],
                label_visibility="collapsed"
            )

        with col_result:
            source_img = img_file or uploaded
            if source_img is not None and model_obj is not None:
                with st.spinner("Mendeteksi objek..."):
                    img_pil = Image.open(source_img)
                    img_out, dets = detect_snapshot(img_pil, model_obj, conf_val)

                st.image(img_out, caption="Hasil Deteksi", use_container_width=True)

                if dets:
                    for d in dets:
                        color = COLORS_HEX.get(d["label"], "#888")
                        st.markdown(
                            f"<div style='padding:6px 12px;border-radius:6px;"
                            f"background:{color}22;border-left:3px solid {color};"
                            f"margin:4px 0;font-size:14px'>"
                            f"<b>{d['label']}</b> — {d['confidence']:.0%} confidence</div>",
                            unsafe_allow_html=True
                        )
                        # Catat ke counter
                        label = d["label"]
                        st.session_state["count"][label] += 1
                        st.session_state["detections"].append({
                            "id": len(st.session_state["detections"])+1,
                            "track_id": None,
                            "label": label,
                            "confidence": d["confidence"],
                            "timestamp": datetime.now().isoformat(),
                            "bbox": d["bbox"],
                        })
                    st.rerun()
                else:
                    st.warning("Tidak ada objek terdeteksi. Coba turunkan threshold di sidebar.")
            elif source_img is None:
                st.info("Ambil foto atau upload gambar untuk memulai deteksi.")

    # ===========================================================
    # MODE 2: LIVE STREAM (WebRTC)
    # ===========================================================
    else:
        st.subheader("📹 Mode Live Stream")

        if not WEBRTC_OK:
            st.error(f"❌ WebRTC tidak tersedia: `{_WEBRTC_ERR}`")
        else:
            # Warning tentang koneksi
            st.warning(
                "⚠️ **Catatan koneksi:**  \n"
                "Mode Live membutuhkan koneksi WebRTC. Jika muncul pesan "
                "'Connection taking longer' — coba:  \n"
                "1. Gunakan **WiFi** bukan data seluler  \n"
                "2. Atau gunakan **Mode Snapshot** di atas  \n"
                "3. Pada beberapa jaringan operator Indonesia, WebRTC diblokir"
            )

            proc = VideoProcessor()
            if model_obj:
                with proc.lock:
                    proc.model       = model_obj
                    proc.conf_thresh = conf_val
                    proc.line_ratio  = line_ratio
                    proc.tracking_on = tracking_on

            ctx = webrtc_streamer(
                key="conveyor-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_processor_factory=lambda: proc,
                media_stream_constraints={
                    "video": {
                        "width" : {"ideal": 320, "max": 480},   # Resolusi lebih kecil = lebih ringan
                        "height": {"ideal": 240, "max": 360},
                        "frameRate": {"ideal": 10, "max": 15},  # FPS lebih rendah = lebih stabil
                    },
                    "audio": False,
                },
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
                st.info(
                    "👆 Klik **START** → izinkan kamera → tunggu koneksi.  \n"
                    "💡 Jika koneksi gagal setelah 30 detik → gunakan **Mode Snapshot**."
                )

# =============================================================
# TABS: LOG & STATISTIK
# =============================================================
st.divider()
tab1, tab2 = st.tabs(["📊 Statistik", "📋 Log Deteksi"])

with tab1:
    if total > 0:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Distribusi Kelas")
            df_d = pd.DataFrame({
                "Kelas": list(cnt.keys()),
                "Jumlah": list(cnt.values()),
                "Persen": [f"{v/total*100:.1f}%" for v in cnt.values()],
            })
            st.dataframe(df_d, hide_index=True, use_container_width=True)
            for lbl, val in cnt.items():
                pct = val/total if total > 0 else 0
                clr = COLORS_HEX[lbl]
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
                    f"<span style='width:110px;font-size:13px'>{lbl}</span>"
                    f"<div style='flex:1;background:#333;border-radius:4px;height:14px'>"
                    f"<div style='width:{pct*100:.1f}%;background:{clr};"
                    f"border-radius:4px;height:14px'></div></div>"
                    f"<span style='font-size:13px'>{val}</span></div>",
                    unsafe_allow_html=True)
    else:
        st.info("Belum ada deteksi.")

with tab2:
    if st.session_state["detections"]:
        df_log = pd.DataFrame(st.session_state["detections"])
        st.dataframe(
            df_log[["id","label","confidence","timestamp"]].tail(25),
            hide_index=True, use_container_width=True,
            column_config={"confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=1, format="%.2f")})
        st.caption(f"Total {len(df_log)} event tercatat.")
    else:
        st.info("Belum ada deteksi.")