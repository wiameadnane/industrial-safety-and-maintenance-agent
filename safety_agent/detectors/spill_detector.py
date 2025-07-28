# spill_detector.py
import cv2
from inference_sdk import InferenceHTTPClient

class SpillDetector:
    """
    A detector class that uses the Roboflow API to detect spills.
    """
    def __init__(self, api_key, model_id="spills-ax5xv/2"):
        """
        Initializes the Roboflow client.

        Args:
            api_key (str): The Roboflow API key.
            model_id (str): The Roboflow model ID to use for inference.
        """
        if not api_key:
            raise ValueError("Roboflow API key is required for SpillDetector.")
        
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )
        self.model_id = model_id
        print(f"[SpillDetector] Initialized with model: {model_id}")

    def detect(self, frame):
        """
        Detects spills in a single frame using the Roboflow API.

        Args:
            frame: The video frame (as a NumPy array) to be analyzed.

        Returns:
            A list of tuples, where each tuple contains a bounding box and a confidence score.
            Example: [((x1, y1, x2, y2), 0.92)]
        """
        detections = []
        try:
            # The Roboflow SDK expects a file path for inference.
            # We save the current frame to a temporary file.
            temp_image_path = "temp_spill_frame.jpg"
            cv2.imwrite(temp_image_path, frame)

            # Run inference on the temporary file
            result = self.client.infer(temp_image_path, model_id=self.model_id)

            for pred in result.get("predictions", []):
                x = int(pred["x"])
                y = int(pred["y"])
                w = int(pred["width"])
                h = int(pred["height"])
                
                # Convert from center (x,y) and width/height to corner points (x1,y1,x2,y2)
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                confidence = pred["confidence"]
                detections.append(((x1, y1, x2, y2), confidence))

        except Exception as e:
            # This prevents the whole application from crashing if Roboflow is down
            print(f"[SpillDetector] Error during Roboflow API call: {e}")

        return detections
