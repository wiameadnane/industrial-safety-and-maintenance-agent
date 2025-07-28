# File: detectors/fall_detector.py
from ultralytics import YOLO
import cv2

class FallDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                label = self.model.names[int(box.cls[0])]
                if 'fall' in label.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    print(f"[FallDetector] LOG: Potential fall detected with confidence {conf:.2f}")
                    detections.append(((x1, y1, x2, y2), conf))
        return detections
