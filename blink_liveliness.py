# blink_liveliness.py - DeepFace Anti-Spoofing Integration
import cv2
import numpy as np
from deepface import DeepFace
import time

class DeepFaceLivelinessDetector:
    def __init__(self):
        """Initialize DeepFace liveliness detector."""
        self.model_name = "VGG-Face"  # or "Facenet", "OpenFace", etc.
        self.detector_backend = "retinaface"  # or "mtcnn", "retinaface", etc.
        self.anti_spoofing = True
        
    def check_liveliness(self, frame):
        """Check if the face in the frame is live using DeepFace anti-spoofing."""
        try:
            # Convert BGR to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract faces with anti-spoofing
            face_objs = DeepFace.extract_faces(
                img=rgb_frame,
                detector_backend=self.detector_backend,
                anti_spoofing=self.anti_spoofing
            )
            
            if not face_objs:
                return {
                    'is_live': False,
                    'confidence': 0.0,
                    'reasons': ['No face detected'],
                    'face_count': 0
                }
            
            # Check if any face is real
            real_faces = [face_obj for face_obj in face_objs if face_obj.get("is_real", False)]
            is_live = len(real_faces) > 0
            confidence = len(real_faces) / len(face_objs) if face_objs else 0.0
            
            reasons = []
            if is_live:
                reasons.append(f"Real face detected ({len(real_faces)}/{len(face_objs)})")
            else:
                reasons.append("Spoof detected - all faces appear fake")
            
            reasons.append(f"Total faces: {len(face_objs)}")
            
            return {
                'is_live': is_live,
                'confidence': confidence,
                'reasons': reasons,
                'face_count': len(face_objs),
                'real_face_count': len(real_faces)
            }
            
        except Exception as e:
            return {
                'is_live': False,
                'confidence': 0.0,
                'reasons': [f"Error: {str(e)}"],
                'face_count': 0,
                'real_face_count': 0
            }
    
    def verify_with_liveliness(self, frame, db_path=None, model_name="VGG-Face"):
        """Verify face identity with liveliness check."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if db_path:
                # Use DeepFace's built-in verification with anti-spoofing
                result = DeepFace.find(
                    img_path=rgb_frame,
                    db_path=db_path,
                    model_name=model_name,
                    detector_backend=self.detector_backend,
                    anti_spoofing=self.anti_spoofing
                )
                
                if result and len(result) > 0 and len(result[0]) > 0:
                    # Get the best match
                    best_match = result[0].iloc[0]
                    identity = best_match['identity']
                    distance = best_match['distance']
                    is_real = best_match.get('is_real', True)
                    
                    return {
                        'verified': True,
                        'identity': identity,
                        'distance': distance,
                        'is_live': is_real,
                        'confidence': 1.0 - distance if distance <= 1.0 else 0.0
                    }
                else:
                    return {
                        'verified': False,
                        'identity': None,
                        'distance': 1.0,
                        'is_live': False,
                        'confidence': 0.0
                    }
            else:
                # Just check liveliness without verification
                liveliness_result = self.check_liveliness(frame)
                return {
                    'verified': False,
                    'identity': None,
                    'distance': 1.0,
                    'is_live': liveliness_result['is_live'],
                    'confidence': liveliness_result['confidence']
                }
                
        except Exception as e:
            return {
                'verified': False,
                'identity': None,
                'distance': 1.0,
                'is_live': False,
                'confidence': 0.0,
                'error': str(e)
            }

def test_deepface_liveliness():
    """Test DeepFace liveliness detection with webcam."""
    detector = DeepFaceLivelinessDetector()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("🎥 DeepFace Liveliness Detection Test")
    print("Look at the camera...")
    print("Press 'q' to quit")
    
    frame_count = 0
    last_check_time = 0
    check_interval = 2.0  # Check every 2 seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Check liveliness at intervals
        if current_time - last_check_time >= check_interval:
            last_check_time = current_time
            
            # Check liveliness
            result = detector.check_liveliness(frame)
            
            # Draw results
            status = "LIVE" if result['is_live'] else "SPOOF"
            color = (0, 255, 0) if result['is_live'] else (0, 0, 255)
            
            cv2.putText(frame, f"Status: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Confidence: {result['confidence']:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Faces: {result['face_count']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show reasons
            for i, reason in enumerate(result['reasons'][:2]):
                cv2.putText(frame, reason, (10, 120 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Show previous results
            cv2.putText(frame, "Checking...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('DeepFace Liveliness Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_deepface_liveliness()