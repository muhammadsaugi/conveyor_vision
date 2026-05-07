# 🏭 Conveyor Vision System

Sistem deteksi real-time berbasis computer vision untuk simulasi pemilahan sampah di conveyor belt.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🎯 Fitur

- **Deteksi 3 kelas** real-time:
  - ✅ **Good Day** — bungkus kopi (berbagai rasa/warna)
  - ✅ **Nutrisari** — bungkus minuman (berbagai rasa/warna)
  - ⚠️ **Tidak Dikenali** — objek lain / confidence rendah
- **Tracking unik** per objek (ByteTrack) → tidak ada double counting
- **Line crossing counting** — hanya dihitung saat melewati garis virtual
- **Alert visual** saat objek Tidak Dikenali terdeteksi
- **Log JSON** otomatis + download CSV
- **Dashboard** counter real-time

---

## 📁 Struktur Project

```
conveyor-vision/
├── app.py                          ← Streamlit app utama
├── best.pt                         ← Model YOLOv8 (taruh di sini)
├── requirements.txt
├── .streamlit/
│   └── config.toml                 ← Tema dark
├── logs/                           ← Log deteksi JSON (auto-generated)
└── Conveyor_Vision_Training.ipynb  ← Notebook Google Colab untuk training
```

---

## 🚀 Cara Menjalankan

### 1. Clone repo

```bash
git clone https://github.com/USERNAME/conveyor-vision.git
cd conveyor-vision
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Letakkan model

```
Salin best.pt (hasil training Google Colab) ke folder ini.
```

### 4. Jalankan app

```bash
streamlit run app.py
```

---

## 🧠 Training Model

Gunakan notebook **`Conveyor_Vision_Training.ipynb`** di Google Colab:

1. Buka [Google Colab](https://colab.research.google.com)
2. Upload notebook
3. Aktifkan GPU: `Runtime → Change runtime type → T4 GPU`
4. Jalankan cell 1–16 secara berurutan
5. Download `best.pt` dari Cell 14
6. Salin `best.pt` ke folder project ini

### Dataset yang digunakan

| Split    | Good Day | Nutrisari | Total |
|----------|----------|-----------|-------|
| Training | 88       | 112       | 200   |
| Validation | 22     | 28        | 50    |

---

## ⚙️ Konfigurasi

Edit parameter di sidebar Streamlit atau langsung di `app.py`:

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `CONF_THRESHOLD` | 0.50 | Confidence minimum untuk kelas terlatih |
| `LINE_RATIO` | 0.55 | Posisi counting line (55% tinggi frame) |
| `CLASS_NAMES` | `["Good Day", "Nutrisari"]` | Kelas yang dilatih |

---

## 📊 Logika Klasifikasi

```
confidence >= threshold  AND  class_id == 0  →  Good Day
confidence >= threshold  AND  class_id == 1  →  Nutrisari
confidence <  threshold  (apapun class-nya)  →  Tidak Dikenali
class_id di luar range                       →  Tidak Dikenali
```

---

## 🛠️ Stack Teknologi

- **Model**: YOLOv8n (Ultralytics) — fine-tuned
- **Tracker**: ByteTrack (built-in Ultralytics)
- **Web**: Streamlit + streamlit-webrtc
- **CV**: OpenCV
- **Training**: Google Colab T4 GPU

---

## 📦 Deploy ke Streamlit Cloud

1. Push semua file ke GitHub (termasuk `best.pt` jika < 100 MB)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. **New app** → pilih repo → pilih `app.py`
4. Klik **Deploy**

> **Catatan**: `best.pt` YOLOv8n ≈ 6 MB — aman di-push ke GitHub.

---

## 📄 Lisensi

MIT License — bebas digunakan untuk keperluan riset dan edukasi.