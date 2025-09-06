






# live_verify.py
import cv2
import json
import numpy as np
import csv
import os
import time
from datetime import datetime
from utils import get_face_embedding, cosine_similarity
from blink_liveliness import DeepFaceLivelinessDetector


with open("db_embeddings.json", "r") as f:
    db = json.load(f)


db = {sid: np.array(proto) for sid, proto in db.items()}


THRESHOLD = 0.923  


marked_attendance = set()

# CSV file setup
CSV_FILE = "attendance.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Timestamp"])

# Initialize DeepFace liveliness detector
liveliness_detector = DeepFaceLivelinessDetector()
frame_sequence = []
max_frames = 30

def mark_attendance(student_id):
    """Append attendance to CSV if not already marked"""
    if student_id not in marked_attendance:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([student_id, timestamp])
        marked_attendance.add(student_id)
        print(f"✅ Attendance marked for {student_id} at {timestamp}")

def verify_frame(frame, check_liveliness=True):
    # Check liveliness first if enabled
    liveliness_results = None
    if check_liveliness:
        liveliness_results = liveliness_detector.check_liveliness(frame)
        if not liveliness_results['is_live']:
            return "Spoof detected", 0.0, None, liveliness_results
    
    # Get face embedding
    emb = get_face_embedding(frame)
    if emb is None:
        return "No face", 0.0, None, liveliness_results

    # Find best match
    best_id, best_sim = None, -1
    for sid, proto in db.items():
        sim = cosine_similarity(emb, proto)
        if sim > best_sim:
            best_id, best_sim = sid, sim

    if best_sim >= THRESHOLD:
        return f"Verified: {best_id}", best_sim, best_id, liveliness_results
    else:
        return "Unknown", best_sim, None, liveliness_results

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    
    print("🎥 Live Face Verification with DeepFace Anti-Spoofing")
    print("Press 'q' to quit, 'b' to toggle liveliness detection, 'r' to reset")
    
    check_liveliness = True
    last_verification_time = 0
    verification_interval = 2.0  # seconds between verifications

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add frame to sequence for liveliness analysis
        frame_sequence.append(frame.copy())
        if len(frame_sequence) > max_frames:
            frame_sequence.pop(0)

        # Perform verification at intervals
        current_time = time.time()
        if current_time - last_verification_time >= verification_interval:
            last_verification_time = current_time
            
            # Verify face in the frame
            label, sim, student_id, liveliness_results = verify_frame(frame, check_liveliness)

            # Mark attendance if verified
            if student_id is not None:
                mark_attendance(student_id)
        else:
            # Use previous results
            label, sim, student_id, liveliness_results = "Processing...", 0.0, None, None

        # Draw main status
        color = (0, 255, 0) if "Verified" in label else (0, 0, 255) if "Spoof" in label else (255, 255, 0)
        cv2.putText(frame, f"{label} ({sim:.2f})", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Draw liveliness info
        if liveliness_results:
            liveliness_status = "LIVE" if liveliness_results['is_live'] else "SPOOF"
            liveliness_color = (0, 255, 0) if liveliness_results['is_live'] else (0, 0, 255)
            
            cv2.putText(frame, f"Liveliness: {liveliness_status}", (30, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, liveliness_color, 2)
            cv2.putText(frame, f"Faces: {liveliness_results['face_count']}", (30, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Real: {liveliness_results['real_face_count']}", (30, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw instructions
        cv2.putText(frame, f"Anti-Spoofing: {'ON' if check_liveliness else 'OFF'}", (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 'b' to toggle anti-spoofing", (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Live Verification", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            check_liveliness = not check_liveliness
            print(f"Anti-spoofing: {'ON' if check_liveliness else 'OFF'}")
        elif key == ord('r'):
            marked_attendance.clear()
            frame_sequence.clear()
            print("Reset completed")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
















# # verify.py without liveness
# import cv2
# import json
# import numpy as np
# import csv
# import os
# from datetime import datetime
# from utils import get_face_embedding, cosine_similarity


# with open("db_embeddings.json", "r") as f:
#     db = json.load(f)


# db = {sid: np.array(proto) for sid, proto in db.items()}


# THRESHOLD = 0.923  


# marked_attendance = set()

# # CSV file setup
# CSV_FILE = "attendance.csv"
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["ID", "Timestamp"])

# def mark_attendance(student_id):
#     """Append attendance to CSV if not already marked"""
#     if student_id not in marked_attendance:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         with open(CSV_FILE, mode="a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([student_id, timestamp])
#         marked_attendance.add(student_id)
#         print(f"✅ Attendance marked for {student_id} at {timestamp}")

# def verify_frame(frame):
#     emb = get_face_embedding(frame)
#     if emb is None:
#         return "No face", 0.0, None

#     best_id, best_sim = None, -1
#     for sid, proto in db.items():
#         sim = cosine_similarity(emb, proto)
#         if sim > best_sim:
#             best_id, best_sim = sid, sim

#     if best_sim >= THRESHOLD:
#         return f"Verified: {best_id}", best_sim, best_id
#     else:
#         return "Unknown", best_sim, None

# def main():
#     cap = cv2.VideoCapture(0)  # 0 = default webcam

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Verify face in the frame
#         label, sim, student_id = verify_frame(frame)

#         # Mark attendance if verified
#         if student_id is not None:
#             mark_attendance(student_id)

#         # Draw results on frame
#         cv2.putText(frame, f"{label} ({sim:.2f})", (30, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (0, 255, 0) if "Verified" in label else (0, 0, 255),
#                     2)

#         cv2.imshow("Live Verification", frame)

#         # Exit on 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
