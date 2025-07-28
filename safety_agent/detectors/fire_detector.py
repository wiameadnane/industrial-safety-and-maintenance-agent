from ultralytics import YOLO
import cv2

class FireDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)  # verbose=False to suppress YOLO output
        detections = []
        for r in results:
            for box in r.boxes:
                # Ensure the detected object is 'fire' before processing
                label = self.model.names[int(box.cls[0])]
                if 'fire' in label.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    print(f"[FireDetector] LOG: Potential fire detected with confidence {conf:.2f}")
                    detections.append(((x1, y1, x2, y2), conf))
        return detections
