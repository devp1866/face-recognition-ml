import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640, 640)):
        """
        Initialize the InsightFace model.
        Args:
            model_name: 'buffalo_l' (high accuracy) or 'buffalo_s' (fast/low mem)
            ctx_id: 0 for CPU, -1 for GPU (if supported)
            det_size: Input image size for detection
        """
        print(f"🧠 Initializing FaceEngine with model: {model_name}")
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def process_image(self, image_bytes):
        """Decode image bytes and return image array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def get_faces(self, img):
        """
        Detect faces in the image.
        Returns a list of InsightFace objects (containing bbox, embedding, etc.)
        """
        if img is None:
            return []
        return self.app.get(img)

    def get_best_face_embedding(self, img):
        """
        Detect faces and return the embedding of the largest face.
        Returns: (embedding, face_object) or (None, None)
        """
        faces = self.get_faces(img)
        if not faces:
            return None, None

        # Pick largest face
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return face.embedding, face

    def recognize_faces(self, img, known_embeddings, ids, names, threshold=0.40):
        """
        Detect and recognize faces in the image against known embeddings.
        Args:
            img: Image array
            known_embeddings: Matrix of known embeddings
            ids: List of known IDs
            names: Dict of ID -> Name
            threshold: Cosine similarity threshold
        Returns:
            processed_img: Image with drawn boxes/labels
            results: List of dicts with match info
        """
        faces = self.get_faces(img)
        out_img = img.copy()
        results = []

        # Pre-normalize known embeddings if they exist
        known_norm = None
        if known_embeddings is not None and known_embeddings.size > 0:
            norms = np.linalg.norm(known_embeddings, axis=1, keepdims=True)
            known_norm = known_embeddings / norms

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            name = "Unknown"
            best_score = 0.0
            color = (0, 0, 255)  # Red for unknown

            if known_norm is not None:
                emb = face.embedding
                emb_norm = emb / np.linalg.norm(emb)

                # Cosine Similarity
                sims = np.dot(known_norm, emb_norm)
                best_idx = np.argmax(sims)
                best_score = float(sims[best_idx])

                if best_score >= threshold:
                    user_id = ids[best_idx]
                    name = names.get(user_id, "Unknown")
                    color = (0, 200, 0)  # Green for match

            label = f"{name} ({best_score:.2f})"
            results.append({"name": name, "score": best_score, "box": [x1, y1, x2, y2]})

            # Draw on image
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        return out_img, results
