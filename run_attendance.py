# run_attendance.py
import cv2
import json
import numpy as np
import csv
import os
from datetime import datetime
from utils import get_embedding, cosine_similarity
from blink_liveliness import DeepFaceLivelinessDetector

# --- CONFIGURATION ---
DB_FILE = "db_embeddings.json"
CSV_FILE = "attendance.csv"
THRESHOLD = 0.425  # Cosine similarity threshold. Calibrate this for VGG-Face.
                  # Values > 0.80 are generally strong matches for VGG-Face.


print("Loading database...")
try:
    with open(DB_FILE, "r") as f:
        db = json.load(f)
    # Convert lists back to numpy arrays
    db = {sid: np.array(proto) for sid, proto in db.items()}
    print(f"✅ Database loaded with {len(db)} students.")
except FileNotFoundError:
    print("❌ Database file not found! Please enroll students first.")
    exit()


liveness_detector = DeepFaceLivelinessDetector()

marked_attendance = set()


if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Timestamp"])

# --- CORE FUNCTIONS ---
def mark_attendance(student_id):
    """Append attendance to CSV if not already marked in this session."""
    if student_id not in marked_attendance:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([student_id, timestamp])
        marked_attendance.add(student_id)
        print(f"Attendance marked for {student_id} at {timestamp}")
        return True
    return False

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🚀 Starting attendance system... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Liveness Check
        liveness_result = liveness_detector.check_liveliness(frame)
        is_live = liveness_result['is_live']
        
        label = "SPOOF"
        color = (0, 0, 255) # Red for spoof/unknown

        if is_live:
            color = (0, 255, 0) # Green for live
            
            # 2. Face Recognition (only if live)
            emb = get_embedding(frame)
            if emb is not None:
                best_id, best_sim = None, -1
                for sid, proto in db.items():
                    sim = cosine_similarity(emb, proto)
                    if sim > best_sim:
                        best_id, best_sim = sid, sim

                if best_sim >= THRESHOLD:
                    label = f"Verified: {best_id}"
                    if mark_attendance(best_id):
                        # Optional: Display a confirmation message on screen for a few seconds
                        pass 
                else:
                    label = "Unknown"
                    color = (0, 255, 255) # Yellow for unknown
            else:
                label = "No Face Detected"
        
        # Draw liveness status and recognition result on the frame
        cv2.putText(frame, "Liveness: " + ("LIVE" if is_live else "SPOOF"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Identity: {label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Live Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("System shut down.")

if __name__ == "__main__":
    main()
