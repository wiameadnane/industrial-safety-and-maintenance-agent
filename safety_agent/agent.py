import cv2
import yaml
import os
import time
from datetime import datetime
from .detectors.fall_detector import FallDetector
from .detectors.fire_detector import FireDetector
from .detectors.spill_detector import SpillDetector

class SafetyAgent:
    def __init__(self, config: dict):
        agent_dir = os.path.dirname(__file__)

        self.fall_threshold = config['fall_confidence']
        self.fire_threshold = config['fire_confidence']
        self.spill_threshold = config['spill_confidence']
        roboflow_api_key = config.get('roboflow_api_key')
        self.spill_detection_interval = config.get('spill_detection_interval', 10)
        
        # Configure evidence path
        self.evidence_path = os.path.join(agent_dir, config.get('evidence_path', 'data/evidence'))
        os.makedirs(self.evidence_path, exist_ok=True)

        fall_model_path = os.path.join(agent_dir, config['fall_model'])
        fire_model_path = os.path.join(agent_dir, config['fire_model'])

        self.fall_detector = FallDetector(fall_model_path)
        self.fire_detector = FireDetector(fire_model_path)
        self.spill_detector = SpillDetector(api_key=roboflow_api_key)
        self.agent_frame_counter = 0

        # Cooldown management
        self.cooldown_config = config.get('alert_cooldowns', {'fall': 10, 'fire': 10, 'spill': 10})
        self.last_detection_times = {event_type: 0 for event_type in self.cooldown_config.keys()}

    def _save_evidence(self, frame, event):
        """Saves visual evidence for a detected event."""
        x1, y1, x2, y2 = event['bbox']
        event_type = event['event_type']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{event_type}_{timestamp}.jpg"
        filepath = os.path.join(self.evidence_path, filename)

        if event_type == 'spill':
            # For spills, save the full frame with a bounding box for context
            evidence_img = frame.copy()
            color = (0, 255, 0) # Green for spill
            label = f"{event_type}: {event['confidence']:.2f}"
            cv2.rectangle(evidence_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(evidence_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # For other events (fall, fire), save a cropped close-up
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            evidence_img = frame[y1:y2, x1:x2]
        
        cv2.imwrite(filepath, evidence_img)
        return filepath

    def process_frame(self, frame):
        self.agent_frame_counter += 1
        all_events = []

        # Process fall detection
        fall_results = self.fall_detector.detect(frame)
        for box, conf in fall_results:
            if conf >= self.fall_threshold:
                all_events.append({'event_type': 'fall', 'confidence': conf, 'bbox': box})

        # Process fire detection
        fire_results = self.fire_detector.detect(frame)
        for box, conf in fire_results:
            if conf >= self.fire_threshold:
                all_events.append({'event_type': 'fire', 'confidence': conf, 'bbox': box})

        # Process spill detection (on a less frequent interval)
        if self.agent_frame_counter % self.spill_detection_interval == 0:
            spill_results = self.spill_detector.detect(frame)
            for box, conf in spill_results:
                if conf >= self.spill_threshold:
                    all_events.append({'event_type': 'spill', 'confidence': conf, 'bbox': box})

        # Post-process all detected events: save evidence and format, respecting cooldowns
        processed_events = []
        current_time = time.time()
        for event in all_events:
            event_type = event['event_type']
            cooldown_period = self.cooldown_config.get(event_type, 10)
            
            if current_time - self.last_detection_times.get(event_type, 0) > cooldown_period:
                evidence_filepath = self._save_evidence(frame, event)
                processed_event = {
                    'event_type': event['event_type'],
                    'confidence': round(event['confidence'], 2),
                    'bbox': event['bbox'],
                    'evidence_path': evidence_filepath
                }
                processed_events.append(processed_event)
                self.last_detection_times[event_type] = current_time # Update the last detection time
                print(f"[SafetyAgent] NEW event: {event_type}. Evidence saved to {evidence_filepath}")

        return processed_events
