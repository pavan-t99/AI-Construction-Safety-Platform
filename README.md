# 🏗️ AI-Powered Construction Safety Monitoring Platform

> An intelligent digital safety officer for construction sites — real-time PPE violation detection, persistent worker risk profiling, hybrid RAG-powered incident analysis, and a live Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-78.9%25_mAP50-00B86B?style=flat)
![SQLite](https://img.shields.io/badge/SQLite-WAL_Mode-003B57?style=flat&logo=sqlite)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Hybrid_RAG-009688?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 What This System Does

This platform does not just detect violations — it reasons about them using a multi-source regulatory knowledge base and a production-grade risk scoring engine.

| Capability | Implementation |
|---|---|
| **See** | YOLOv8s custom model — 10 PPE classes, 78.9% mAP50 |
| **Track** | ByteTrack — persistent worker identity across frames |
| **Score** | Multi-factor risk engine: violation type + duration + machinery proximity |
| **Analyse** | Groq LLM (llama-3.3-70b) — structured incident report per violation |
| **Regulate** | Hybrid RAG — 30 curated OSHA/Indian regulations + PDF-derived FAISS index |
| **Alert** | 3-level escalating alerts: Alert → Reminder → Escalation |
| **Store** | SQLite WAL mode — concurrent pipeline writes + dashboard reads |
| **Display** | Streamlit dashboard — 9 pages: live feed, analytics, AI chat, worker profiles |
| **Log** | Structured per-camera daily log files with severity levels |

---

## 📊 Model Performance

Trained on Kaggle (Tesla T4) using the Construction Site Safety Image Dataset (Roboflow).
Architecture: **YOLOv8s** — 11.1M parameters, 28.5 GFLOPs

| Metric | Score |
|---|---|
| **mAP50 (overall)** | **78.89%** |
| **Precision** | **89.73%** |
| **Recall** | **73.84%** |
| mAP50-95 | 49.40% |
| Inference speed | 10.8ms/frame (T4 GPU) |

### Per-Class mAP50

| Class | mAP50 | |
|---|---|---|
| Hardhat | 92.6% | ✅ |
| Safety Vest | 90.5% | ✅ |
| machinery | 89.1% | ✅ |
| Person | 86.4% | ✅ |
| NO-Mask | 84.0% | ✅ |
| NO-Safety Vest | 82.9% | ✅ |
| vehicle | 80.8% | ✅ |
| Mask | 75.5% | ✅ |
| NO-Hardhat | 56.8% | ⚠️ Targeted fine-tuning planned |
| Safety Cone | 50.2% | ⚠️ Limited training samples |

> Model weights: [kaggle.com/pavankurman/trained-ppe-yolo-models](https://kaggle.com/pavankurman/trained-ppe-yolo-models)

---

## 🏗️ System Architecture

```
Camera / Video File
        │
        ▼
   OpenCV Capture (1280×720)
        │
        ▼
  YOLOv8s Inference ──────────── ppe_yolov8s_best.pt (10 classes)
        │
        ▼
  ByteTrack Tracking ─────────── Persistent WORKER_{id} across frames
        │
        ▼
  Violation↔Person Association ── Center-point containment
        │
        ▼
  Duration Timer ─────────────── Min 5s before incident confirmed
        │
        ▼
  Multi-Factor Risk Scorer
  ├── Base score (violation type: 15–30 pts)
  ├── +20 if near machinery (<200px)
  ├── +20 if multiple simultaneous violations
  └── +10 per 30 seconds duration
        │
        ▼
  GroqWorker (background thread) ← never blocks detection loop
  ├── Hybrid RAG retrieval
  │     ├── Source A: 30 curated OSHA + Indian regulations (FAISS in-memory)
  │     └── Source B: PDF-derived FAISS index (faiss_construction_index/)
  ├── Groq LLM report (llama-3.3-70b-versatile)
  └── SQLite write (incidents + workers tables)
        │
        ▼
  SQLite WAL Mode ─────────────── Concurrent read/write safe
  ├── incidents table
  ├── workers table
  └── alerts table
        │
        ▼
  Streamlit Dashboard ─────────── 9 pages, live feed, AI chat, analytics
        │
        ▼
  Telegram / Email ────────────── Escalating 3-level alerts
```

---

## 🔑 Key Technical Features

### 1. Multi-Factor Real-Time Risk Scoring
Violations are not binary flags. Every incident gets a dynamic continuous risk score:

```python
risk_score = base_points[violation_type]        # 15–30 pts
risk_score += 20   # if worker near machinery (<200px)
risk_score += 20   # if multiple simultaneous violations
risk_score += (duration_seconds // 30) * 10    # escalates over time

risk_level = "LOW" if risk_score < 30 else "MEDIUM" if risk_score < 70 else "HIGH"
```

A 5-second helmet violation scores 30. The same violation near active machinery for 2 minutes scores 90 (CRITICAL). Static detection systems cannot do this.

### 2. Hybrid RAG — Dual-Source Regulatory Retrieval

```
Query: "HELMET VIOLATION"
        │
        ├── Source A: 30 curated OSHA + IS regulations (FAISS IndexFlatIP, always available)
        └── Source B: PDF-derived FAISS index (faiss_construction_index/, optional)
                │
                ▼
        Merge → Deduplicate → Top-k by cosine similarity
                │
                ▼
        Injected into Groq prompt with exact regulation citations
```

If the PDF index is missing, Source A serves as silent fallback — no crash.

### 3. Groq LLM in Background Thread
The detection loop never waits for LLM inference. A `GroqWorker` daemon thread with a bounded queue (`maxsize=50`) picks up confirmed incidents and writes to SQLite asynchronously.

### 4. Persistent Worker Risk Profiling Without Biometrics
ByteTrack assigns visual track IDs that persist across frames. These map to stable `WORKER_{id}` identifiers with cumulative risk histories — no RFID, no face recognition, no biometric data.

### 5. 3-Level Escalating Alert System
```
Level 1 — ALERT      : Immediate on first confirmed violation (5s+)
Level 2 — REMINDER   : 30 seconds later if unresolved
Level 3 — ESCALATION : 5 minutes later if still unresolved
```
All levels → SQLite alerts table + per-camera log + Telegram/Email.

### 6. Concurrent-Safe Storage (WAL Mode)
The pipeline writes incidents while Streamlit reads from the same SQLite database simultaneously. WAL mode eliminates read-write locking — a critical production requirement.

### 7. Text-to-SQL AI Chat (Ask AI page)
```
User question
      ↓
Stage 1: Groq converts natural language → SQLite SQL → executes query
Stage 2: Pull structured context (stats, workers, violation summary)
Stage 2b: RAG retrieves relevant OSHA/IS regulations
Stage 3: Groq synthesises all three → professional answer with citations
```

---

## 🗄️ Database Schema

```sql
CREATE TABLE incidents (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id         TEXT NOT NULL,
    person_id         TEXT NOT NULL,
    violation_type    TEXT NOT NULL,
    severity          TEXT NOT NULL,       -- LOW / MEDIUM / HIGH
    risk_score        INTEGER NOT NULL,
    risk_level        TEXT NOT NULL,
    start_time        TEXT NOT NULL,
    end_time          TEXT NOT NULL,
    duration_seconds  REAL NOT NULL,
    confidence        REAL NOT NULL,
    image_path        TEXT,
    near_machinery    INTEGER DEFAULT 0,
    groq_analysis     TEXT,               -- Full LLM report
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE workers (
    camera_id               TEXT NOT NULL,
    person_id               TEXT NOT NULL,
    total_risk_score        INTEGER DEFAULT 0,
    risk_level              TEXT DEFAULT 'LOW',
    unique_violation_count  INTEGER DEFAULT 0,
    total_incidents         INTEGER DEFAULT 0,
    violations_json         TEXT DEFAULT '[]',
    last_seen               TEXT,
    UNIQUE(camera_id, person_id)
);

CREATE TABLE alerts (
    camera_id       TEXT NOT NULL,
    person_id       TEXT NOT NULL,
    alert_level     INTEGER NOT NULL,     -- 1 / 2 / 3
    violation_type  TEXT NOT NULL,
    message         TEXT NOT NULL,
    timestamp       TEXT NOT NULL
);
```

---

## 📁 Project Structure

```
AI-Construction-Safety-Platform/
│
├── safety_pipeline_v4.py        ← Core AI pipeline (inference + tracking + risk scoring)
├── app.py                       ← Streamlit dashboard (9 pages)
├── database.py                  ← SQLite WAL layer (read/write functions)
├── Incident_Analysis_v2.py      ← Groq LLM + Hybrid RAG engine
├── rag_knowledge_base.py        ← 30 curated OSHA + Indian construction regulations
├── logger_setup.py              ← Per-camera structured daily logging
├── cameras.json                 ← Camera registry (editable at runtime)
├── requirements.txt
├── .gitignore
├── README.md
│
├── faiss_construction_index/    ← PDF-derived FAISS index (Source B RAG)
│   ├── index.faiss
│   └── index.pkl
│
└── data/
    ├── safety_platform.db       ← SQLite database
    └── CAM_01/
        ├── live_frame.jpg       ← Current annotated frame for dashboard
        ├── Site_Safety.json     ← Live site metrics (2s throttle)
        ├── violation_log.csv    ← Raw event log
        ├── evidence/            ← Cropped violation images
        └── logs/
            └── CAM_01_YYYY-MM-DD.log
```

---

## 🖥️ Dashboard Pages

| Page | What It Shows |
|---|---|
| **Dashboard** | Live annotated camera feed + real-time site risk score |
| **Executive Summary** | KPIs, violation trend charts, hourly heatmap, top-risk workers |
| **Ask AI** | Natural language → SQL → RAG → Groq synthesised answer |
| **Incidents** | Full violation log table from SQLite |
| **Workers** | Per-worker risk profiles, cumulative scores, violation history |
| **Site Safety** | Live site risk matrix from Site_Safety.json |
| **Reports** | Groq-generated incident reports with evidence images |
| **Messages** | Escalating alert history (Level 1 / 2 / 3) |
| **Camera Management** | Add, edit, delete cameras — RTSP URL support |

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Webcam or video file for local testing

### Install

```bash
git clone https://github.com/pavan-t99/AI-Construction-Safety-Platform.git
cd AI-Construction-Safety-Platform
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
GROQ_API_AI_SAFETY_REPORT=your_groq_api_key_here
HEADLESS=0
TELEGRAM_BOT_TOKEN=          # optional
TELEGRAM_CHAT_ID=            # optional
```

### Download Model Weights

Download `ppe_yolov8s_best.pt` from Kaggle and place in the project root:

```
https://kaggle.com/pavankurman/trained-ppe-yolo-models
```

### Run

```bash
# Terminal 1 — Start dashboard
streamlit run app.py

# Terminal 2 — Start pipeline (webcam)
python safety_pipeline_v4.py 0 CAM_01

# Or test with a video file
python safety_pipeline_v4.py path/to/video.mp4 CAM_01
```

Open browser at `http://localhost:8501`

---

## 📦 Requirements

```
ultralytics>=8.0.0
opencv-python-headless
streamlit>=1.28.0
streamlit-autorefresh
groq
python-dotenv
faiss-cpu
sentence-transformers
pandas
plotly
psutil
requests
```

---

## 🌐 Cloud Deployment (Zero Cost)

| Component | Platform | Notes |
|---|---|---|
| **Dashboard** | HuggingFace Spaces (Streamlit SDK) | 2 vCPU, 16GB RAM, always-on, public URL |
| **Pipeline inference** | Kaggle (Tesla T4) | Free GPU, run pipeline notebook |
| **Local dev** | Docker container | Packaging only, not hosting |

> Live demo: [huggingface.co/spaces/KURMANPAVANKUMAR/AI-Construction-Safety-Platform](https://huggingface.co/spaces/KURMANPAVANKUMAR/AI-Construction-Safety-Platform)

**Note:** On the HuggingFace demo, live webcam is unavailable (cloud sandbox). Use the video upload feature in the sidebar — upload any construction site video and the full pipeline runs: detections, risk scoring, Groq reports, SQLite population, all 9 dashboard pages.

---

## 🔬 Research & Patent Potential

### Novel Contribution 1 — Multi-Factor Real-Time Risk Scoring
Existing PPE systems output binary violation flags. This system computes a continuous dynamic risk score incorporating violation type, duration, machinery proximity, and simultaneous violation count. This enables intelligent triage that no static detection pipeline provides.

### Novel Contribution 2 — Persistent Worker Risk Profiling via Visual Tracking Alone
The system builds longitudinal risk histories per worker from CCTV footage alone — no RFID, no biometrics. Visual track IDs are mapped to cumulative incident records across sessions.

Both are novel in the construction safety domain and represent patentable engineering innovations.

---

## 🗺️ Roadmap

- [x] Custom YOLOv8s PPE model (78.9% mAP50, 10 classes)
- [x] ByteTrack persistent worker tracking
- [x] Multi-factor risk scoring engine
- [x] Groq LLM incident report generation (llama-3.3-70b-versatile)
- [x] Hybrid RAG — OSHA + Indian regulations + PDF FAISS index
- [x] SQLite WAL mode (concurrent-safe)
- [x] 3-level escalating alert system
- [x] Per-camera structured daily logging
- [x] Streamlit 9-page dashboard
- [x] Text-to-SQL AI chat (Ask AI page)
- [x] HuggingFace Spaces deployment
- [ ] Telegram bot activation
- [ ] Docker packaging
- [ ] NO-Hardhat targeted fine-tuning (v2.0 model)
- [ ] Worker ReID across sessions (face recognition / RFID — v2.0)
- [ ] Multi-camera parallel processing
- [ ] Edge deployment (Jetson Nano)

---

## 👤 Author

**T.Pavan Kumar** 
final year of BSCS student
AI & Computer Vision Engineer  
pavankumar255281"gmail.com 
(https://www.linkedin.com/in/pavan-kumar-740586399/)  
https://www.kaggle.com/pavankurman

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built to demonstrate production-quality AI engineering: custom model training, real-time inference, persistent tracking, LLM reasoning, concurrent-safe storage, and structured observability.*
