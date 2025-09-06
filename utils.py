import numpy as np
import cv2
import face_recognition  # or whichever embedding library you are using

def get_face_embedding(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_recognition.face_encodings(rgb_frame)
    if len(results) > 0:
        emb = results[0]
        return emb / np.linalg.norm(emb)   # ✅ normalize
    return None
 # first face in frame

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two embeddings."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
