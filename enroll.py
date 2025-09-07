import face_recognition
import numpy as np
import os
import pickle


DATASET_FOLDER = "dataset"
EMBEDDINGS_FILE = "known_embeddings.pkl"

def generate_embeddings():
    print("Processing dataset and generating embeddings...")
    known_face_embeddings = []
    known_face_ids = []

    for student_id in os.listdir(DATASET_FOLDER):
        student_folder = os.path.join(DATASET_FOLDER, student_id)
        if os.path.isdir(student_folder):
            embeddings = []
            for filename in os.listdir(student_folder):
                image_path = os.path.join(student_folder, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        embeddings.append(face_encodings[0])
                except Exception as e:
                    print(f"Warning: Could not process image {image_path}. Error: {e}")
            
            if embeddings:
                prototype_embedding = np.mean(embeddings, axis=0)
                known_face_embeddings.append(prototype_embedding)
                known_face_ids.append(student_id)
                print(f"Enrolled student: {student_id}")

    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({'embeddings': known_face_embeddings, 'ids': known_face_ids}, f)
    
    print(f"\n✅ Embeddings for {len(known_face_ids)} students saved to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    generate_embeddings()