# 🎯 SmartFace — Secure AI Attendance System

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask_2.0%2B-green)
![AI](https://img.shields.io/badge/AI-InsightFace%20(ArcFace)-orange)
![AI](https://img.shields.io/badge/AI-MiniFASNet%20(ONNX)-orange)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)

A **Local-First AI Face Recognition Attendance System** built for high-security environments.  
SmartFace combines **ArcFace** deep learning for precise identity recognition with a **three-layer anti-spoofing pipeline** (Motion Analysis + MiniFASNet + FFT Deepfake Detection) and **Adaptive Embedding Learning**, ensuring only physically present, real individuals are verified while the system continuously learns and adapts to aging or lighting changes over time.

---

## 🧠 How It Works

All biometric data is processed **100% locally** — no cloud APIs, no external calls.

Authentication is a rigorous three-stage pipeline:

1. **Motion Check** *(Webcam only)*: Three frames are captured 400 ms apart. Inter-frame pixel variance detects whether the face is live (natural micro-motion) or a static photo/screen (near-zero variance).
2. **Liveness & Deepfake Check**: 
   - **MiniFASNet** runs at three crop scales (2.7×, 4.0×, 1.5×) to catch phone screen reflections and print attacks.
   - **FFT Deepfake Analysis** analyzes frequency domains to detect AI-generated or synthetic faces.
3. **Identity Match & Adaptive Update**: The verified face embedding is compared against the enrolled database using cosine similarity. If matched with high confidence, the system updates the stored embedding (EMA) to learn the user's current appearance.

All checks must pass for attendance to be marked.

---

## 🚀 Key Features

- **🛡️ Three-Layer Anti-Spoofing & Deepfake Detection**
  - **Layer 1 — Motion Analysis**: Flags perfectly static sources (printed photos, screens) by analyzing 3 consecutive frames.
  - **Layer 2 — Multi-Scale Liveness**: MiniFASNet ensemble across 3 crop scales catches spoofing textures.
  - **Layer 3 — FFT Analysis**: Fast Fourier Transform catches AI-generated and deepfake synthetic media.

- **🧠 Adaptive Embedding Learning**
  - The system gets smarter over time. As users successfully authenticate, their stored embeddings are slowly updated using an Exponential Moving Average (EMA) to account for aging, facial hair, and lighting changes.

- **👥 Group Attendance**
  - Process up to 5 faces simultaneously in a single webcam capture or photo upload. Perfect for group check-ins.

- **🔍 Precision Face Recognition**
  - Powered by **InsightFace (ArcFace / buffalo_l)** — state-of-the-art 512-dimensional face embeddings.
  - Multi-photo enrollment: up to **4 reference images averaged** into one robust embedding.

- **📊 Advanced Analytics Dashboard**
  - Interactive **Chart.js** dashboard showing 7-day attendance trends, per-subject breakdowns, and individual attendance heatmaps.

- **📱 Modern Web Interface**
  - Dark mode glassmorphism UI built with **TailwindCSS**.
  - **AI Confidence Meter** visualizes match strength, liveness score, and FFT entropy.
  - **Face Guide Overlay** assists users with optimal camera positioning.

- **📂 Smart Data Management**
  - SQLite storage for embeddings and subject-specific attendance tracking (prevents duplicate subject check-ins on the same day).

---

## 🧰 Tech Stack

| Component         | Technology               | Description                              |
| :---------------- | :----------------------- | :--------------------------------------- |
| **Core Logic**    | Python 3.10+             | Primary programming language             |
| **Web Framework** | Flask + Gunicorn         | WSGI web server (dev + production)       |
| **AI Engine**     | InsightFace (ArcFace)    | 512-dim face embeddings via buffalo_l    |
| **Anti-Spoofing** | MiniFASNet v2 + FFT      | Multi-scale liveness & deepfake detector |
| **Inference**     | ONNX Runtime             | CPU-optimized model inference            |
| **Computer Vision** | OpenCV + SciPy       | Image decoding, motion check, FFT logic  |
| **Database**      | SQLite                   | Lightweight local storage                |
| **Frontend**      | TailwindCSS + Chart.js   | Glassmorphism UI, interactive analytics  |
| **Config**        | python-dotenv            | Environment variable management          |

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/devp1866/face-recognition-ml
cd face-recognition-ml
```

### 2️⃣ Install Dependencies

> Python 3.10 is recommended. Python 3.11/3.12 may require specific ONNX Runtime wheels.

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Environment

Create a `.env` file in the project root:

```bash
# .env
SECRET_KEY=your_secret_key_here
```

Generate a strong key with:

```bash
python -c "import os; print(os.urandom(32).hex())"
```

> ⚠️ Never commit `.env` to version control. It is already listed in `.gitignore`.

### 4️⃣ Setup Models

**Face Recognition model** (`buffalo_l`) is **automatically downloaded** on first run by InsightFace into `resources/models/`.

**Anti-Spoofing model** (`minifasv2.onnx`) is **included in the repository** at `resources/models/minifasv2.onnx` — no manual download needed.

> If the anti-spoofing model is missing, the app will still run but liveness detection will be disabled (all faces treated as real).

### 5️⃣ Run the Application

```bash
python app.py
```

Access the dashboard at: **`http://localhost:5000`**

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t smartface .

# Run container
docker run -p 5000:5000 smartface
```

The container uses **Gunicorn** with 1 worker (optimized for free-tier RAM) and a 120s timeout to allow InsightFace model loading on first start.

---

## 🎮 Usage Guide

### 1. Enroll a User 👤

- Go to the **Enroll** page.
- Enter the person's full name.
- Upload **1–4 clear face photos** (different angles and lighting recommended).
- The system extracts and averages the face embeddings, then saves to the database.
- Maximum **10 users** supported (configurable via `MAX_USERS` in `app.py`).

### 2. Mark Attendance 📷

- Go to the **Attendance** page.
- **Webcam tab** *(recommended — full anti-spoofing active)*:
  - Click the capture button.
  - Hold still for ~1 second while 3 frames are captured automatically.
  - Both motion and liveness checks run before identity is matched.
- **Upload tab** *(testing only — liveness check disabled)*:
  - Upload a photo directly.
  - Identity matched without liveness verification.

**Result indicators:**
| Box Color | Meaning |
|---|---|
| 🟩 Green | Real person + recognized → Attendance marked |
| 🟥 Red | Unknown face, spoof detected, or static image |
| 🟧 Orange | Recognized face but flagged as spoof attempt |

### 3. Manage & Export 📊

- **Dashboard**: Daily stats — total users and today's attendance count.
- **Analytics (New)**: Interactive charts mapping attendance over time, by subject, and by top attendees.
- **Group Attendance (New)**: Batch process up to 5 people at once using a group photo.
- **Users**: View enrolled profiles, delete individual users.
- **Logs**: Full attendance history with name/date filters; download as CSV.
- **Reset**: Wipe all data (users, logs, results) — requires typing `RESTART` to confirm.

---

## ⚙️ Configuration Reference

Key constants in `app.py` that can be tuned:

| Constant | Default | Description |
|---|---|---|
| `MAX_USERS` | `10` | Maximum number of enrollable users |
| `SKIP_UPLOAD_LIVENESS` | `True` | Skip liveness for uploaded images (testing) |
| `MOTION_THRESHOLD` | `0.8` | Min inter-frame pixel variance to pass motion check |

Key constants in `core/recognition.py`:

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_THRESHOLD` | `0.50` | Default identity match threshold |
| `WEBCAM_THRESHOLD` | `0.42` | Looser threshold used for webcam captures |

Key constants in `core/antispoof.py`:

| Constant | Default | Description |
|---|---|---|
| `LIVENESS_THRESHOLD` | `0.50` | Minimum real-score to pass liveness check |
| `ENSEMBLE_SCALES` | `[2.7, 4.0, 1.5]` | Crop scales for multi-scale ensemble |

---

## 📁 Project Structure

```
face-recognition-ml/
├── app.py                  # Flask routes, motion check, attendance logic
├── core/
│   ├── recognition.py      # FaceEngine: detection, embedding, recognition
│   ├── antispoof.py        # AntiSpoofDet: multi-scale MiniFASNet inference
│   ├── deepfake_detector.py# FFT-based synthetic media detection
│   ├── embedding_tracker.py# Adaptive learning (EMA) logic
│   ├── quality.py          # Blur, brightness, and pose validation
│   └── storage.py          # SQLite CRUD operations
├── templates/
│   ├── layout.html         # Base layout (nav, header, reset modal)
│   ├── index.html          # Dashboard
│   ├── analytics.html      # Chart.js dashboards
│   ├── attendance.html     # Multi-frame webcam + upload + results
│   ├── group_attendance.html # Multi-face detection handler
│   ├── enroll.html         # User enrollment
│   ├── users.html          # User management
│   ├── logs.html           # Attendance history
│   └── about.html          # Project info
├── resources/
│   └── models/
│       ├── buffalo_l/          # ArcFace model (auto-downloaded on first run)
│       └── minifasv2.onnx      # Anti-spoofing model (included in repo)
├── .env                    # Secret key — NOT committed (see .gitignore)
├── requirements.txt
├── Dockerfile
└── DEPLOYMENT.md
```

---

## 👨‍💻 Author

- **Devkumar Patel**
- 📧 **Email**: devp1866@gmail.com
- 🔗 **Portfolio**: [devkumarpatel.vercel.app](https://devkumarpatel.vercel.app)
- 🔗 **LinkedIn**: [devp1866](https://www.linkedin.com/in/devp1866/)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it for personal or commercial projects.
