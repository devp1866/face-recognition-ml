import cv2
import numpy as np
import onnxruntime as ort
import os


class AntiSpoofDet:
    def __init__(self, model_path, scale=2.7):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Anti-spoofing model not found at {model_path}")

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.scale = scale
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = tuple(self.session.get_inputs()[0].shape[2:])
        self.output_name = self.session.get_outputs()[0].name

    def _crop_face(self, image, bbox):
        """
        bbox: [x, y, w, h]
        """
        src_h, src_w = image.shape[:2]
        x, y, box_w, box_h = bbox

        # 1. Enforce Square Crop (use max dimension)
        center_x = x + box_w / 2
        center_y = y + box_h / 2

        long_side = max(box_w, box_h)

        # 2. Calculate ideal square box
        new_w = int(long_side * self.scale)
        new_h = int(long_side * self.scale)

        x1 = int(center_x - new_w / 2)
        y1 = int(center_y - new_h / 2)
        x2 = x1 + new_w
        y2 = y1 + new_h

        # 3. Handle Padding (Replicate matches edge texture, avoiding 'fake' black borders)
        # Calculate required padding
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - src_h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - src_w)

        # Crop the valid area first
        y1_valid = max(0, y1)
        x1_valid = max(0, x1)
        y2_valid = min(src_h, y2)
        x2_valid = min(src_w, x2)

        raw_crop = image[y1_valid:y2_valid, x1_valid:x2_valid]

        if raw_crop.size == 0:
            return cv2.resize(
                image[y : y + box_h, x : x + box_w], self.input_size[::-1]
            )

        # Apply Replicate Border
        padded_crop = cv2.copyMakeBorder(
            raw_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )

        return cv2.resize(padded_crop, self.input_size[::-1])

    def _preprocess(self, image, bbox_xywh):
        face = self._crop_face(image, bbox_xywh)
        # 1. Colors: Attempting BGR (OpenCV Default) instead of RGB.
        # Some MiniFASNet models are trained on BGR.
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # 2. Shape: (C, H, W)
        face = face.astype(np.float32).transpose(2, 0, 1)
        # 3. Scale: 0-1
        face /= 255.0
        face = np.expand_dims(face, axis=0)
        return face

    def predict(self, image, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

        input_tensor = self._preprocess(image, bbox_xywh)

        # Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0]

        # Softmax
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        p0 = probs[0][1]
        p1 = probs[0][0]
        print(f"DEBUG_RAW: Spoof={p0:.4f}, Real={p1:.4f}]")

        real_score = p1

        if real_score > 0.40:
            label = "Real"
            score = real_score
        else:
            label = "Spoof"
            score = real_score

        return label, float(score)
