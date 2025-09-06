# utils.py
import numpy as np
import cv2
from deepface import DeepFace

MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "retinaface" 
def get_embedding(frame):

    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True, 
            align=True
        )
        
        if results:
            embedding = results[0]['embedding']
            return np.array(embedding)
            
    except Exception as e:
        return None
    
    return None

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two embeddings."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    
    return np.dot(vec1, vec2)