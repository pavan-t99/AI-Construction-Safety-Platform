# 🏗️ AI-Powered Construction Safety Monitoring Platform

> Real-time PPE violation detection, persistent worker risk profiling, and AI-generated incident analysis — built as an intelligent digital safety officer for construction sites.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-78.9%25_mAP50-00B86B?style=flat)
![SQLite](https://img.shields.io/badge/SQLite-WAL_Mode-003B57?style=flat&logo=sqlite)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLM_Analysis-F55036?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 What This System Does

This platform behaves like an **intelligent digital safety officer**. It does not just detect violations — it reasons about them.

| Capability | Implementation |
|---|---|
| **See** | YOLOv8s custom model, 10 PPE classes, 78.9% mAP50 |
| **Track** | BotSort persistent worker identity across frames |
| **Reason** | Multi-factor risk scoring: violation type + duration + machinery proximity |
| **Analyse** | Groq LLM generates structured incident reports per violation |
| **Alert** | 3-level escalating Telegram notifications (Alert → Reminder → Escalation) |
| **Store** | SQLite with WAL mode — concurrent pipeline writes + dashboard reads |
| **Display** | Live Streamlit dashboard with real-time feed, analytics, worker history |
| **Log** | Structured per-camera daily log files with severity levels |

---

## 📊 Model Performance

Trained on Kaggle (Tesla T4) using the Construction Site Safety dataset (Roboflow).  
Model: **YOLOv8s** — 11.1M parameters, 28.5 GFLOPs

| Metric | Score |
|---|---|
| **mAP50 (overall)** | **78.89%** |
| **Precision** | **89.73%** |
| **Recall** | **73.84%** |
| mAP50-95 | 49.40% |
| Inference speed | 10.8ms/frame (T4 GPU) |

### Per-Class mAP50

| Class | mAP50 | Notes |
|---|---|---|
| Hardhat | 92.6% | ✅ Strong |
| Safety Vest | 90.5% | ✅ Strong |
| machinery | 89.1% | ✅ Strong |
| Person | 86.4% | ✅ Strong |
| NO-Mask | 84.0% | ✅ Strong |
| NO-Safety Vest | 82.9% | ✅ Strong |
| vehicle | 80.8% | ✅ Strong |
| Mask | 75.5% | ✅ Good |
| NO-Hardhat | 56.8% | ⚠️ Improvement planned |
| Safety Cone | 50.2% | ⚠️ Limited training samples |

> Model weights available on Kaggle: [pavankurman/trained-ppe-yolo-models](https://kaggle.com/pavankurman/trained-ppe-yolo-models)

---

## 🏗️ System Architecture

```
Camera / Video File
        │
        ▼
   OpenCV Capture (1280×720)
        │
        ▼
  YOLOv8s Inference ──────────── ppe_yolov8s_best.pt
  (resize to 640×640)             78.9% mAP50, 10 classes
        │
        ▼
  BotSort Tracking ───────────── Persistent worker IDs across frames
        │
        ▼
 Violation↔Person Association ── Center-point containment logic
        │
        ▼
  Duration Timer ─────────────── Min 5s before incident is confirmed
        │
        ▼
  Multi-Factor Risk Scorer
  ├── Base score (violation type)
  ├── +20 if near machinery (<200px)
  ├── +20 if multiple simultaneous violations
  └── +10 per 30 seconds of duration
        │
        ▼
  Groq LLM ───────────────────── Structured incident report generation
        │
        ▼
  SQLite (WAL mode)
  ├── incidents table
  ├── workers table
  └── alerts table
        │
        ▼
  Streamlit Dashboard ─────────── Live feed + 6 analytics pages
        │
        ▼
  Telegram Bot ────────────────── Escalating 3-level alert system
```

---

## 🔑 Key Technical Features

### 1. Multi-Factor Risk Scoring (Novel)
Violations are not binary. Each incident receives a dynamic risk score:
```python
risk_score = base_points[violation_type]   # 15–30 points
risk_score += 20  # if worker near machinery
risk_score += 20  # if multiple simultaneous violations  
risk_score += (duration_seconds // 30) * 10  # escalates over time
```
This enables triage: a 5-second helmet violation scores 30, but the same violation near active machinery for 2 minutes scores 90 (CRITICAL).

### 2. Persistent Worker Identity Without Biometrics
BotSort assigns track IDs that persist across frames. The system maps these to stable `WORKER_{id}` identifiers and accumulates a risk history per worker across the entire session — without RFID, face recognition, or any biometric data.

### 3. 3-Level Escalating Alert System
```
Level 1 — ALERT      : Immediate on first confirmed violation
Level 2 — REMINDER   : 30 seconds later if unresolved  
Level 3 — ESCALATION : 5 minutes later if still unresolved
```
Each level triggers a Telegram notification to the site supervisor.

### 4. Concurrent-Safe Storage (WAL Mode)
The pipeline writes incidents to SQLite while Streamlit reads from the same database simultaneously. WAL (Write-Ahead Logging) mode eliminates read-write locking — a critical production requirement for real-time systems.

### 5. Structured Per-Camera Logging
```
data/
  CAM_01/
    logs/
      CAM_01_2026-06-19.log   ← daily rotation, severity levels
    evidence/                  ← cropped violation images
    Site_Safety.json           ← live dashboard feed (2s throttle)
    live_frame.jpg             ← current frame for Streamlit
    violation_log.csv          ← raw event log
safety_platform.db             ← SQLite: incidents, workers, alerts
```

---

## 🗄️ Database Schema

```sql
-- Closed incidents with full metadata
CREATE TABLE incidents (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id         TEXT NOT NULL,
    person_id         TEXT NOT NULL,
    violation_type    TEXT NOT NULL,
    severity          TEXT NOT NULL,
    risk_score        INTEGER NOT NULL,
    risk_level        TEXT NOT NULL,
    start_time        TEXT NOT NULL,
    end_time          TEXT NOT NULL,
    duration_seconds  REAL NOT NULL,
    confidence        REAL NOT NULL,
    image_path        TEXT,
    near_machinery    INTEGER DEFAULT 0,
    groq_analysis     TEXT,
    created_at        TEXT DEFAULT (datetime('now'))
);

-- One row per worker — upserted on every incident
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

-- Full alert history with escalation level
CREATE TABLE alerts (
    camera_id       TEXT NOT NULL,
    person_id       TEXT NOT NULL,
    alert_level     INTEGER NOT NULL,
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
├── s_p.py                    # Core AI pipeline (inference + tracking + incidents)
├── app.py                    # Streamlit dashboard (7 pages)
├── database.py               # SQLite layer (WAL mode, read/write functions)
├── logger_setup.py           # Structured per-camera logging
├── Incident_Analysis.py      # Groq LLM report generation
├── cameras.json              # Camera registry (editable)
├── requirements.txt          # Pinned dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10+
- Webcam or video file for testing
- Groq API key (free at console.groq.com)

### Install
```bash
git clone https://github.com/pavan-t99/AI-Construction-Safety-Platform.git
cd AI-Construction-Safety-Platform
pip install -r requirements.txt
```

### Configure
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
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
python s_p.py 0 CAM_01

# Or test with a video file
python s_p.py path/to/video.mp4 CAM_01
```

Open browser at `http://localhost:8501`

---

## 📦 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
streamlit>=1.28.0
streamlit-autorefresh
pandas
groq
python-dotenv
psutil
```

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| **Dashboard** | Live camera feed + real-time site risk metrics |
| **Incidents** | Full violation log (CSV) with timestamps |
| **Workers** | Per-worker risk profiles, violation history, scores |
| **Site Safety** | Current site risk level and active violation count |
| **Reports** | AI-generated Groq incident reports with evidence images |
| **Messages** | Escalating alert history (Level 1/2/3) |
| **Camera Management** | Add/edit/remove cameras, RTSP support |

---

## 🔬 Research & Patent Potential

This system demonstrates two novel contributions:

**1. Multi-Factor Real-Time Risk Scoring for Construction Safety**  
Existing PPE detection systems output binary violation flags. This system computes a continuous risk score incorporating violation type, duration, proximity to machinery, and simultaneous violation count. This approach enables intelligent triage that static detection cannot provide.

**2. Persistent Worker Risk Profiling via Visual Tracking Alone**  
The system links visual track IDs to cumulative risk histories across sessions without biometrics, RFID, or any physical identifier. This enables longitudinal safety analytics from CCTV footage alone.

Both contributions are novel in the construction safety domain and represent patentable innovations.

---

## 🗺️ Roadmap

- [x] Custom YOLOv8s PPE model (78.9% mAP50)
- [x] BotSort persistent tracking
- [x] Multi-factor risk scoring engine
- [x] Groq LLM incident report generation
- [x] SQLite with WAL mode (concurrent-safe)
- [x] 3-level escalating alert system
- [x] Per-camera structured logging
- [x] Streamlit multi-page dashboard
- [ ] RAG pipeline for OSHA/IS regulation lookup
- [ ] Telegram bot integration
- [ ] Docker deployment
- [ ] Multi-camera parallel processing
- [ ] Edge deployment (Jetson Nano)
- [ ] NO-Hardhat class improvement (targeted fine-tuning)

---

## 👤 Author

**T.Pavan Kumar**  
AI & Computer Vision Engineer  
pavankumar255281"gmail.com 
(https://www.linkedin.com/in/pavan-kumar-740586399/)  
https://www.kaggle.com/pavankurman

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built to demonstrate production-quality AI engineering: custom model training, real-time inference, persistent tracking, LLM reasoning, concurrent-safe storage, and structured observability.*