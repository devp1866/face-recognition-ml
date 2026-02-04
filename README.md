# 🎯 SmartFace - Secure AI Attendance System

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask_2.0%2B-green)
![AI](<https://img.shields.io/badge/AI-InsightFace%20(ArcFace)-orange>)
![AI](https://img.shields.io/badge/AI-MiniFASNet-orange)

An advanced, **Local-First AI Face Recognition Attendance System** designed for high-security environments.  
SmartFace combines **ArcFace** deep learning for precise recognition with **MiniFASNet** for robust anti-spoofing (Liveness Detection), ensuring that only real, physically present users are verified.

---

## 🧠 Overview

SmartFace is an efficient, contactless, and automated solution for attendance management. Unlike cloud-based systems, it processes all sensitive biometric data **locally** on your secure server, ensuring maximum privacy and zero latency.

The system encodes facial features into **high-dimensional embeddings** using InsightFace. During authentication, it performs a two-stage verification:

1.  **Liveness Check**: Verifies if the face is real using texture, depth, and edge analysis (prevents photo/video attacks).
2.  **Identity Match**: Compares the live face embedding against the encrypted local database.

---

## 🚀 Key Features

- **🛡️ Liveness Detection (Anti-Spoofing)**:
  - Protects against digital screens, printed photos, and replay attacks.
  - Uses **MiniFASNet** with custom "Pad-then-Crop" logic for edge stability.
- **🔍 Precision Recognition**:
  - Powered by **InsightFace (ArcFace)**, achieving state-of-the-art accuracy.
- **🔏 Privacy-First Architecture**:
  - **100% Local Processing** (No external APIs or Cloud calls).
  - Biometric data stored as secure embeddings, not raw images.
- **⚡ High Performance**:
  - Optimized for CPU inference using ONNX Runtime.
  - Real-time processing at high FPS.
- **📱 Modern Web Interface**:
  - Responsive, mobile-friendly Dashboard.
  - Built with **Flask** and **TailwindCSS** (Glassmorphism design).
- **📂 Smart Data Management**:
  - SQLite Database for reliable record keeping.
  - Multi-photo enrollment with quality constraints.

---

## 🧰 Tech Stack

| Component           | Technology            | Description                        |
| :------------------ | :-------------------- | :--------------------------------- |
| **Core Logic**      | Python 3.10+          | Primary programming language       |
| **Web Framework**   | Flask                 | Lightweight WSGI web server        |
| **AI Engine**       | InsightFace (ArcFace) | Deep learning face recognition     |
| **Anti-Spoofing**   | MiniFASNet (ONNX)     | Liveness detection model           |
| **Computer Vision** | OpenCV                | Image processing and capture       |
| **Database**        | SQLite                | Lightweight transactional database |
| **Frontend**        | TailwindCSS           | Modern, utility-first CSS styling  |

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/devp1866/face-recognition-ml
cd face-recognition-ml
```

### 2️⃣ Install Dependencies

Ensure you have Python 3.10 installed (Python 3.11/3.12 may require specific ONNX Runtime wheels).

```bash
pip install -r requirements.txt
```

### 3️⃣ Setup Models

1.  **Face Recognition**: Models (Buffalo_L) are **automatically downloaded** on first run.
2.  **Anti-Spoofing**: You must manually download the MiniFASNet model:
    - [Download minifasv2.onnx](https://github.com/suriAI/face-antispoof-onnx/blob/main/models/best/98.20/best_model.onnx)
    - Rename it to `minifasv2.onnx`.
    - Place it in: `face-recognition-ml/resources/models/`

### 4️⃣ Run the Application

```bash
python app.py
```

_Access the dashboard at:_ `http://localhost:5000`

---

## 🎮 Usage Guide

### 1. Enrollment 👨

- Navigate to the **"Enroll User"** page.
- Enter the user's Full Name.
- Upload **2-5 reference photos**.
  - _Tip:_ Use different lighting conditions and angles for best accuracy.
- The system will generate and store the unique face embedding.

### 2. Marking Attendance 📷

- Go to the **"Attendance"** page.
- The webcam will activate automatically.
- Position your face clearly in the frame.
- **Real-Time Feedback**:
  - 🟩 **Green Box**: Real Person + Recognized (Attendance Marked).
  - 🟥 **Red Box**: Unknown Person or Spoof Attempt detected.
- View your Match Score and Liveness Score instantly.

### 3. Management 📊

- **Dashboard**: View daily statistics and recent activity.
- **Users**: Manage enrolled profiles (Delete users/data).
- **Logs**: View detailed attendance history with timestamps.

---

## 👨‍💻 Author

- **Devkumar Patel**
- 📧 **Email**: devp1866@gmail.com
- 🔗 **Portfolio**: [Devkumar Patel](https://devkumarpatel.vercel.app)
- 🔗 **LinkedIn**: [devp1866](https://www.linkedin.com/in/devp1866/)

---

### 📄 License

This project is licensed under the **MIT License**. feel free to use, modify, and distribute it for personal or commercial projects.
