# live_verify.py
import cv2
import json
import numpy as np
import csv
import os
from datetime import datetime
from utils import get_face_embedding, cosine_similarity

# Load enrolled embeddings
with open("db_embeddings.json", "r") as f:
    db = json.load(f)

# Convert JSON arrays back to numpy
db = {sid: np.array(proto) for sid, proto in db.items()}

# Set similarity threshold
THRESHOLD = 0.923  # tweak using calibrate_threshold.py

# Keep track of attendance (so each student only once)
marked_attendance = set()

# CSV file setup
CSV_FILE = "attendance.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Timestamp"])

def mark_attendance(student_id):
    """Append attendance to CSV if not already marked"""
    if student_id not in marked_attendance:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([student_id, timestamp])
        marked_attendance.add(student_id)
        print(f"✅ Attendance marked for {student_id} at {timestamp}")

def verify_frame(frame):
    emb = get_face_embedding(frame)
    if emb is None:
        return "No face", 0.0, None

    best_id, best_sim = None, -1
    for sid, proto in db.items():
        sim = cosine_similarity(emb, proto)
        if sim > best_sim:
            best_id, best_sim = sid, sim

    if best_sim >= THRESHOLD:
        return f"Verified: {best_id}", best_sim, best_id
    else:
        return "Unknown", best_sim, None

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Verify face in the frame
        label, sim, student_id = verify_frame(frame)

        # Mark attendance if verified
        if student_id is not None:
            mark_attendance(student_id)

        # Draw results on frame
        cv2.putText(frame, f"{label} ({sim:.2f})", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if "Verified" in label else (0, 0, 255),
                    2)

        cv2.imshow("Live Verification", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
