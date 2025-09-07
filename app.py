# app.py (Final Version with CSV Logging & Pre-loaded Embeddings)
import face_recognition
import cv2
import numpy as np
import os
import time
import pickle
import csv
from datetime import datetime
from scipy.spatial import distance as dist

# --- LIVENESS CHECKER CLASS ---
class LivenessChecker:
    def __init__(self, frames_to_check=10, variation_thresh=0.03):
        self.frames_to_check = frames_to_check
        self.variation_thresh = variation_thresh 
        self.ear_history = []
        self.is_live = False

    def _eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        if C == 0: return 0.3 # Avoid division by zero
        return (A + B) / (2.0 * C)

    def check(self, face_landmarks):
        if 'left_eye' not in face_landmarks or 'right_eye' not in face_landmarks:
            return self.is_live
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        left_ear = self._eye_aspect_ratio(np.array(left_eye))
        right_ear = self._eye_aspect_ratio(np.array(right_eye))
        ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(ear)
        if len(self.ear_history) > self.frames_to_check:
            self.ear_history.pop(0)
            ear_std_dev = np.std(self.ear_history)
            if ear_std_dev > self.variation_thresh:
                self.is_live = True
        return self.is_live

    def reset(self):
        self.ear_history.clear()
        self.is_live = False

# --- CONFIGURATION ---
EMBEDDINGS_FILE = "known_embeddings.pkl"
CSV_FILE = "attendance.csv"
DISTANCE_THRESHOLD = 0.5
RESIZE_WIDTH = 480

# --- GLOBAL VARIABLES ---
known_face_embeddings = []
known_face_ids = []
marked_this_session = set()
liveness_checkers = {}

def load_known_faces():
    """Loads pre-generated embeddings from a file."""
    global known_face_embeddings, known_face_ids
    print("Loading known faces from embeddings file...")
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            known_face_embeddings = data['embeddings']
            known_face_ids = data['ids']
        print(f"✅ Loaded {len(known_face_ids)} enrolled students.")
    except FileNotFoundError:
        print(f"❌ Embeddings file '{EMBEDDINGS_FILE}' not found. Please run enroll.py first.")
        exit()

def log_attendance(student_id):
    """Logs attendance to a CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Check if CSV exists, if not, write header
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['ID', 'Timestamp'])
        writer.writerow([student_id, timestamp])

def main():
    """Main function to run the attendance system."""
    load_known_faces()

    print("🚀 Starting camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ CRITICAL: Cannot open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("System ready. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        scale = RESIZE_WIDTH / frame.shape[1]
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

        current_face_locations = set(face_locations)
        for loc in list(liveness_checkers.keys()):
            if loc not in current_face_locations:
                del liveness_checkers[loc]
        
        if not face_locations:
            cv2.putText(frame, "Point camera at face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        for face_location, face_landmarks in zip(face_locations, face_landmarks_list):
            if face_location not in liveness_checkers:
                liveness_checkers[face_location] = LivenessChecker()
            
            is_live = liveness_checkers[face_location].check(face_landmarks)

            student_id = "Checking Liveness..."
            color = (0, 255, 255) # Yellow

            if is_live:
                face_embeddings = face_recognition.face_encodings(rgb_small_frame, [face_location])
                if face_embeddings:
                    face_embedding = face_embeddings[0]
                    distances = face_recognition.face_distance(known_face_embeddings, face_embedding)
                    best_match_index = np.argmin(distances)
                    min_distance = distances[best_match_index]

                    if min_distance <= DISTANCE_THRESHOLD:
                        student_id = known_face_ids[best_match_index]
                        color = (0, 255, 0) # Green

                        if student_id not in marked_this_session:
                            log_attendance(student_id)
                            print(f"✅ Attendance Marked for {student_id}")
                            marked_this_session.add(student_id)
                    else:
                        student_id = "Unknown"
                        color = (0, 0, 255) # Red

            # Display Logic
            top, right, bottom, left = [int(c / scale) for c in face_location]
            label = student_id
            if student_id in marked_this_session:
                label += " (Marked)"
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Live Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()