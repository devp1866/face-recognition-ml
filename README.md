---
title: Smart Face Attendance
emoji: 📸
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

# 🎯 Face Detection & Recognition Attendance System

An advanced **AI-powered Face Recognition Attendance System** using **InsightFace** and **OpenCV**.  
This system captures student faces, generates embeddings during enrollment, and recognizes them in real-time to mark attendance automatically with **ID and timestamp** in a CSV file.

---

## 🧠 Overview

This project provides an efficient, contactless, and automated solution to attendance management. It replaces traditional manual entry systems with a modern AI approach capable of identifying multiple faces in real time.

Each student's facial features are encoded into **high-dimensional embeddings** using InsightFace at the time of enrollment. During runtime, the system extracts facial embeddings from live video frames and matches them against the stored embeddings database using vector similarity. Once a match is confirmed, the student’s attendance is recorded in a CSV file with their ID and the current timestamp — ensuring accuracy and eliminating duplicate entries for the same day.

---

## 🚀 Key Features

- 🔍 **Face Detection & Recognition** using InsightFace + OpenCV
- 🧩 **Embeddings Generation** during enrollment
- 🗂️ **Local Storage of Face Embeddings** (serialized format for fast access)
- 🕒 **Real-Time Attendance Logging** with timestamps
- ⚡ **Optimized for Large-Scale Use** via vector search (cosine similarity)
- 🚫 **Duplicate Attendance Prevention** (marks once per person per day)
- 🧾 **Attendance CSV Export** for record keeping

---

## 🧰 Tech Stack

| Component          | Description                          |
| ------------------ | ------------------------------------ |
| **Python 3.10+**   | Programming language                 |
| **InsightFace**    | Deep learning-based face recognition |
| **OpenCV**         | Image capture and face detection     |
| **NumPy / Pandas** | Data handling and vector processing  |
| **CSV File**       | Attendance logging                   |
| **Scikit-learn**   | Similarity and vector management     |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/devp1866/face-recognition-ml
cd face-recognition-ml
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 🖥️ Running the Attendance App

```bash
python app.py
```

## 👨‍💻 Author

- 🧩 Devkumar Patel
- 📧 Email: devp1866@gmail.com
- 🔗 Portfolio: https://devp1866.framer.website
