import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Import AntiSpoofing (used for upload-mode only, not webcam)
from core.antispoof import AntiSpoofDet


class FaceEngine:

    DEFAULT_THRESHOLD = 0.50

    # ── Blink / EAR constants ──────────────────────────────────────────────────
    # Eye Aspect Ratio (EAR) = eye_height / eye_width, derived from the
    # 106-point facial landmarks already provided by InsightFace buffalo_l.
    #
    # Empirical ranges (using bounding-box of eye-region landmarks):
    #   Open eye  → EAR ≈ 0.22 – 0.40
    #   Blinking  → EAR ≈ 0.05 – 0.18
    #
    # A valid blink requires BOTH:
    #   (a) at least one frame where EAR < EAR_CLOSED_THRESHOLD  (eye closing)
    #   (b) at least one frame where EAR > EAR_OPEN_THRESHOLD    (eye open)
    # This prevents misfires from consistently bad landmark detection.
    EAR_CLOSED_THRESHOLD = 0.17   # below this → eye is closing / closed
    EAR_OPEN_THRESHOLD   = 0.22   # above this → eye is clearly open
    EAR_MIN_PTS          = 4      # minimum landmarks in eye region to be reliable

    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640, 640)):
        """
        Initialize the InsightFace model and Anti-Spoofing model.
        """
        print(f"Initializing FaceEngine with model: {model_name}")
        # Explicitly pass 'root' to force download/load from resources folder
        resources_path = os.path.join(os.getcwd(), "resources")
        self.app = FaceAnalysis(
            name=model_name, root=resources_path, providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # Cleanup: Remove the downloaded zip file if it exists to save space
        try:
            zip_path = os.path.join(resources_path, "models", f"{model_name}.zip")
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print(f"🧹 Cleanup: Removed temporary file {model_name}.zip")
        except Exception as e:
            print(f" Warning: Could not remove zip file: {e}")

        # Initialize Liveness Detector
        spoof_model_path = os.path.join(
            os.getcwd(), "resources", "models", "minifasv2.onnx"
        )
        self.spoof_det = None
        if os.path.exists(spoof_model_path):
            print(" Initializing Anti-Spoofing Model...")
            self.spoof_det = AntiSpoofDet(spoof_model_path)
        else:
            print(f" Warning: Anti-Spoofing model not found at {spoof_model_path}")

    def process_image(self, image_bytes):
        """Decode image bytes and return image array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def get_faces(self, img):
        """
        Detect faces in the image.
        Returns a list of InsightFace face objects (containing bbox, embedding, etc.)
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

        # Pick the largest face by bounding-box area
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return face.embedding, face

    def check_liveness(self, img, bbox):
        """
        Check if a face region is Real or Fake/Spoof using MiniFASNet.
        NOTE: Used only for non-webcam sources. Webcam liveness is handled
              exclusively via blink detection (detect_blink_in_sequence).

        Args:
            img:   Full BGR image
            bbox:  Face bounding box [x1, y1, x2, y2]

        Returns:
            (is_real: bool, score: float, label: str)
        """
        if not self.spoof_det:
            return True, 1.0, "Real (No Model)"

        label, score = self.spoof_det.predict(img, bbox)
        is_real = label == "Real"
        return is_real, score, label

    # ── Blink Detection (Active Liveness) ─────────────────────────────────────

    def _get_ear_from_face(self, face):
        """
        Compute the Eye Aspect Ratio (EAR) for a detected face using the
        106-point landmarks already provided by InsightFace buffalo_l.

        Instead of relying on hard-coded landmark indices (which vary between
        model versions), we find all landmarks that fall spatially within the
        eye region of the face bounding box, then measure their height/width
        ratio. This is robust across resolutions and model variants.

        Returns:
            float  → averaged EAR of both eyes  (0.0 – 1.0)
            None   → landmarks not available or too few points found
        """
        if face.landmark_2d_106 is None:
            return None

        lm  = face.landmark_2d_106          # shape (106, 2), pixel coords
        x1, y1, x2, y2 = face.bbox.astype(int)
        h   = y2 - y1
        mid_x = (x1 + x2) / 2.0

        # Eye band: roughly 20 %–52 % of face height from the top of the bbox.
        # This reliably covers both eyelids without including eyebrows or cheeks.
        eye_top    = y1 + h * 0.20
        eye_bottom = y1 + h * 0.52

        ears = []
        for side in ("left", "right"):
            if side == "left":
                pts = np.array([lm[i] for i in range(106)
                                if eye_top < lm[i][1] < eye_bottom
                                and lm[i][0] < mid_x])
            else:
                pts = np.array([lm[i] for i in range(106)
                                if eye_top < lm[i][1] < eye_bottom
                                and lm[i][0] >= mid_x])

            if len(pts) < self.EAR_MIN_PTS:
                continue

            eye_h = float(pts[:, 1].max() - pts[:, 1].min())
            eye_w = float(pts[:, 0].max() - pts[:, 0].min())
            if eye_w > 0:
                ears.append(eye_h / eye_w)

        return float(np.mean(ears)) if ears else None

    def detect_blink_in_sequence(self, imgs):
        """
        Detect whether a genuine blink occurs across a sequence of webcam frames.

        A real person naturally blinks within 3–5 seconds. A printed photo,
        phone screen, or video replay cannot produce a genuine, spontaneous blink
        that matches the open → closed → open EAR signature we look for.

        Logic:
            Valid blink = EAR drops below EAR_CLOSED_THRESHOLD in ≥ 1 frame
                          AND EAR exceeds EAR_OPEN_THRESHOLD  in ≥ 1 frame.
            Both conditions are required to guard against landmark detection
            failures that might produce uniformly low EAR values.

        Args:
            imgs:  Ordered list of BGR images from consecutive webcam captures.

        Returns:
            blink_detected (bool)
            ear_sequence   (list[float | None])  — one EAR value per frame
            debug_info     (dict)                 — diagnostic data for logging
        """
        ear_sequence = []

        for img in imgs:
            if img is None:
                ear_sequence.append(None)
                continue

            faces = self.get_faces(img)
            if not faces:
                ear_sequence.append(None)
                continue

            # Use the largest detected face in each frame
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            ear_sequence.append(self._get_ear_from_face(face))

        valid_ears = [e for e in ear_sequence if e is not None]

        if len(valid_ears) < 2:
            return False, ear_sequence, {
                "reason": "Too few frames with detectable landmarks",
                "valid_frames": len(valid_ears),
            }

        min_ear = min(valid_ears)
        max_ear = max(valid_ears)

        # Both conditions must hold for a confirmed blink
        eye_closed_seen = min_ear < self.EAR_CLOSED_THRESHOLD
        eye_open_seen   = max_ear > self.EAR_OPEN_THRESHOLD
        blink_detected  = eye_closed_seen and eye_open_seen

        debug = {
            "valid_frames"    : len(valid_ears),
            "total_frames"    : len(imgs),
            "min_ear"         : round(min_ear, 4),
            "max_ear"         : round(max_ear, 4),
            "closed_threshold": self.EAR_CLOSED_THRESHOLD,
            "open_threshold"  : self.EAR_OPEN_THRESHOLD,
            "eye_closed_seen" : eye_closed_seen,
            "eye_open_seen"   : eye_open_seen,
            "ear_sequence"    : [round(e, 4) if e is not None else None
                                 for e in ear_sequence],
        }

        return blink_detected, ear_sequence, debug

    def annotate_as_spoof(self, img, reason="NO BLINK"):
        """
        Detect faces in img, draw red SPOOF boxes with the given reason label,
        and return (annotated_image, results_list) in the standard format.

        Used when liveness (blink detection) fails before recognition even runs.
        """
        faces   = self.get_faces(img)
        out_img = img.copy()
        results = []

        if not faces:
            # No face found at all — put a warning on the image
            cv2.putText(
                out_img, "NO FACE DETECTED",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )
            return out_img, results

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                out_img,
                f"SPOOF: {reason}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
            )
            results.append({
                "user_id" : None,
                "name"    : f"SPOOF ({reason})",
                "score"   : 0.0,
                "box"     : [x1, y1, x2, y2],
                "is_real" : False,
                "liveness": 0.0,
            })

        return out_img, results

    def recognize_faces(
        self,
        img,
        known_embeddings,
        ids,
        names,
        threshold=None,
        skip_liveness=False,
    ):
        """
        Detect, recognize, and optionally check liveness for all faces in an image.

        Args:
            img:              BGR image (numpy array)
            known_embeddings: (N, D) array of enrolled face embeddings
            ids:              List of user IDs corresponding to known_embeddings rows
            names:            Dict mapping user_id → name
            threshold:        Cosine similarity threshold. Defaults to DEFAULT_THRESHOLD (0.50).
            skip_liveness:    If True, liveness check is skipped and all faces treated as Real.
                              Use for uploaded images where liveness is meaningless.

        Returns:
            out_img:  Annotated copy of the image with bounding boxes and labels drawn.
            results:  List of dicts, one per detected face:
                        {
                          "user_id":  int | None,
                          "name":     str,
                          "score":    float,   # cosine similarity (0–1)
                          "box":      [x1, y1, x2, y2],
                          "is_real":  bool,
                          "liveness": float,   # real probability from antispoof
                        }
        """
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLD

        faces = self.get_faces(img)
        out_img = img.copy()
        results = []

        # Pre-normalize known embeddings once (avoids repeated division in the loop)
        known_norm = None
        if known_embeddings is not None and known_embeddings.size > 0:
            norms = np.linalg.norm(known_embeddings, axis=1, keepdims=True)
            known_norm = known_embeddings / norms

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            #  Liveness check 
            if skip_liveness:
                # TESTING: Liveness check intentionally disabled for this source.
                is_real = True
                liveness_score = 1.0
                liveness_label = "Real (Skipped)"
            else:
                if self.spoof_det:
                    liveness_label, liveness_score = self.spoof_det.predict(img, box)
                    is_real = liveness_label == "Real"
                else:
                    is_real = True
                    liveness_score = 1.0
                    liveness_label = "Real (No Model)"

            # Identity recognition
            matched_user_id = None
            name = "Unknown"
            best_score = 0.0
            color = (0, 0, 255)  # Red default (unknown / spoof)

            if known_norm is not None:
                emb = face.embedding
                emb_norm = emb / np.linalg.norm(emb)

                sims = np.dot(known_norm, emb_norm)
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])

                if best_score >= threshold:
                    candidate_id = ids[best_idx]
                    name_candidate = names.get(candidate_id, "Unknown")

                    if is_real:
                        # Confirmed: real person + recognised → green
                        matched_user_id = candidate_id
                        name = name_candidate
                        color = (0, 200, 0)
                    else:
                        # Spoof attempt by a known person → orange warning
                        name = f"FAKE: {name_candidate}"
                        color = (0, 165, 255)

            # Override to red if spoofing regardless of match
            if not is_real:
                color = (0, 0, 255)

            # Build display label
            if is_real:
                label_text = f"{name} ({best_score:.2f})"
            else:
                label_text = f"SPOOF ({liveness_score:.2f})"

            # Draw on image 
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out_img,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            results.append(
                {
                    "user_id": matched_user_id,
                    "name": name,
                    "score": best_score,
                    "box": [x1, y1, x2, y2],
                    "is_real": is_real,
                    "liveness": liveness_score,
                }
            )

        return out_img, results
