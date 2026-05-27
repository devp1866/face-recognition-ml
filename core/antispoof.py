import cv2
import numpy as np
import onnxruntime as ort
import os


class AntiSpoofDet:
    # Liveness threshold: real_score must exceed this to be considered Real.
    # Raised to 0.72 (production norm) from the loose 0.60 default.
    LIVENESS_THRESHOLD = 0.72

    # Multi-scale crop factors used for ensemble prediction.
    # Running at multiple scales captures different texture frequencies,
    # making it significantly harder for phone screens to fool the model.
    ENSEMBLE_SCALES = [2.7, 4.0, 1.5]

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Anti-spoofing model not found at {model_path}")

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = tuple(self.session.get_inputs()[0].shape[2:])
        self.output_name = self.session.get_outputs()[0].name

    def _crop_face(self, image, bbox, scale):
        """
        Crop a square region around the face, scaled by `scale`.
        Uses BORDER_REPLICATE padding to avoid fake black borders
        that could bias the model.
        bbox: [x, y, w, h]
        """
        src_h, src_w = image.shape[:2]
        x, y, box_w, box_h = bbox

        # Enforce square crop using the longer side
        center_x = x + box_w / 2
        center_y = y + box_h / 2
        long_side = max(box_w, box_h)

        new_w = int(long_side * scale)
        new_h = int(long_side * scale)

        x1 = int(center_x - new_w / 2)
        y1 = int(center_y - new_h / 2)
        x2 = x1 + new_w
        y2 = y1 + new_h

        # Padding amounts (replicate border avoids misleading edge artifacts)
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - src_h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - src_w)

        # Clip to image bounds first, then pad
        y1_valid = max(0, y1)
        x1_valid = max(0, x1)
        y2_valid = min(src_h, y2)
        x2_valid = min(src_w, x2)

        raw_crop = image[y1_valid:y2_valid, x1_valid:x2_valid]

        if raw_crop.size == 0:
            # Fallback: crop the raw bbox without scaling
            return cv2.resize(
                image[y : y + box_h, x : x + box_w], self.input_size[::-1]
            )

        padded_crop = cv2.copyMakeBorder(
            raw_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )

        return cv2.resize(padded_crop, self.input_size[::-1])

    def _preprocess(self, image, bbox_xywh, scale):
        """Crop, normalize and format the face region for model input."""
        face = self._crop_face(image, bbox_xywh, scale)
        # BGR input (matching training data convention for this model)
        face = face.astype(np.float32).transpose(2, 0, 1)  # (C, H, W)
        face /= 255.0
        face = np.expand_dims(face, axis=0)
        return face

    def _run_single_scale(self, image, bbox_xywh, scale):
        """Run inference at a single scale. Returns the 'Real' probability."""
        input_tensor = self._preprocess(image, bbox_xywh, scale)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0]

        # Stable softmax
        logits_shifted = logits - np.max(logits)
        probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=1, keepdims=True)

        # Index 0 = Real, Index 1 = Spoof (MiniFASNet convention)
        real_score = float(probs[0][0])
        return real_score

    def predict(self, image, bbox_xyxy):
        """
        Predict liveness using multi-scale ensemble.

        Runs MiniFASNet at multiple crop scales (2.7x, 4.0x, 1.5x) and
        averages the 'Real' probability. This exploits the fact that
        phone-screen spoofs have different texture artifacts at different
        scales, making single-scale evasion much harder.

        Args:
            image:      Full BGR image (numpy array)
            bbox_xyxy:  Face bounding box [x1, y1, x2, y2]

        Returns:
            (label: str, score: float)
            label = "Real" or "Fake"
            score = averaged real probability (0.0 – 1.0)
        """
        x1, y1, x2, y2 = bbox_xyxy
        bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        # Collect real-probability scores across all ensemble scales
        scale_scores = []
        for scale in self.ENSEMBLE_SCALES:
            try:
                real_score = self._run_single_scale(image, bbox_xywh, scale)
                scale_scores.append(real_score)
            except Exception as e:
                print(f"⚠️ Antispoof scale {scale} failed: {e}")

        if not scale_scores:
            # If all scales failed, fail safe → treat as Fake
            return "Fake", 0.0

        avg_real_score = float(np.mean(scale_scores))

        label = "Real" if avg_real_score > self.LIVENESS_THRESHOLD else "Fake"
        return label, avg_real_score
