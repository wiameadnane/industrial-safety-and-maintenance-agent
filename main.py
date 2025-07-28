import cv2
import yaml
import os
import sys
import time
import subprocess
import atexit
from datetime import datetime
from dotenv import load_dotenv
from safety_agent.agent import SafetyAgent
from coordination_agent.agent import CoordinatorAgent

# Add the Maintenance-Agent directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Maintenance-Agent'))
from src.maintenance_agent import MaintenanceAgent
from src.log_simulator import run_log_simulator
from multiprocessing import Process

# Load environment variables from .env file
load_dotenv()

# --- Global list to keep track of background processes ---
background_processes = []

def start_background_services():
    """Starts the Maintenance Agent and its log simulator in the background."""
    print("[Launcher] Starting Maintenance Agent services...")
    
    with open('simulation_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    maintenance_agent = MaintenanceAgent()
    log_simulator_process = Process(target=run_log_simulator)
    log_simulator_process.start()
    background_processes.append(log_simulator_process)
    print("[Launcher] Maintenance Agent and Log Simulator are running.")

def stop_background_services():
    """Ensures all started background processes are terminated on exit."""
    print("\n[Launcher] Shutting down background services...")
    for p in background_processes:
        p.terminate()
    # Wait for all processes to terminate
    for p in background_processes:
        p.join()
    print("[Launcher] All background services stopped.")

# Register the cleanup function to be called on script exit
atexit.register(stop_background_services)

def main():
    """Main function to run the entire multi-agent simulation."""
    # Load simulation configuration
    with open('simulation_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    RTSP_URL = config['video_source']
    CAMERA_ID = config['camera_id']
    FRAME_SKIP = config.get('frame_skip', 3) # Default to 3 if not specified

    # --- Agent Initialization ---
    print("[Launcher] Initializing agents...")
    # Load the safety agent's specific config
    with open('safety_agent/config.yaml', 'r') as f:
        safety_agent_config = yaml.safe_load(f)
    # Merge the general simulation config with the agent-specific one
    # This ensures the agent gets the correct cooldowns from the main config
    full_agent_config = {**config, **safety_agent_config}

    safety_agent = SafetyAgent(config=full_agent_config)
    coordinator_agent = CoordinatorAgent(config_path='coordination_agent/config.yaml')
    print("[Launcher] All agents initialized.")

    # --- Video Stream Setup ---
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {RTSP_URL}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)
    print(f"Video stream opened. Camera ID: {CAMERA_ID}, FPS: {fps:.2f}")

    # --- Main Simulation Loop ---
    frame_count = 0
    last_known_detections = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            detected_events = safety_agent.process_frame(frame)
            if detected_events:
                last_known_detections = detected_events
                for event in detected_events:
                    coordinator_agent.process_safety_event(event, CAMERA_ID)
            else:
                # If no events are detected in this frame, clear the last known detections
                last_known_detections = []

        # Simulate a maintenance event for a different machine
        if frame_count == 300: # Simulate after 10 seconds
            print("\n--- [Simulating Maintenance Event] ---")
            maintenance_report = {
                'type': 'maintenance',
                'machine_id': 'CNC-MILL-02',
                'location': 'Sector B, Bay 5',
                'timestamp': datetime.now().isoformat(),
                'problem_detected': 'Unusual vibration patterns detected',
                'recommended_actions': 'Schedule immediate inspection of spindle bearings.',
                'evidence_filename': None
            }
            coordinator_agent.process_maintenance_report(maintenance_report)

        # Simulate a critical machine failure event
        if frame_count == 450: # Simulate after 15 seconds
            print("\n--- [Simulating Machine Failure Event] ---")
            failure_event = {
                'event_type': 'machine_failure',
                'confidence': 0.99,
                'evidence_filename': 'simulated_failure_evidence.jpg' # Example evidence
            }
            # In a real scenario, you'd save a real image. Here we'll just pass the name.
            # This event is for a camera watching a different machine.
            coordinator_agent.process_safety_event(failure_event, camera_id='CAM003')

        # Simulate a critical error from the Maintenance Agent
        if frame_count == 600: # Simulate after 20 seconds
            print("\n--- [Simulating Machine Error from Maintenance Agent] ---")
            error_report = {
                'type': 'machine_error', # New event type
                'machine_id': 'CNC-MILL-01',
                'location': 'Sector A, Bay 2',
                'timestamp': datetime.now().isoformat(),
                'problem_detected': 'CRITICAL: Hydraulic pressure loss detected. Immediate shutdown required.',
                'recommended_actions': 'Cease all operations. Manually depressurize system. Do not restart until cleared by technician.',
                'evidence_filename': None
            }
            coordinator_agent.process_maintenance_report(error_report)

        # --- Visualization ---
        if last_known_detections:
            for event in last_known_detections:
                x1, y1, x2, y2 = event['bbox']
                event_type = event['event_type']
                color = {'fall': (255, 0, 0), 'spill': (0, 255, 0)}.get(event_type, (0, 0, 255))
                label = f"{event_type}: {event['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Agenix-Eye - Unified Monitoring Interface", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Launcher] Main simulation loop finished.")

if __name__ == "__main__":
    start_background_services()
    main()
