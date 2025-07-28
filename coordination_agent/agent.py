# coordinator/agent.py
import os
import yaml
import requests
import time
from datetime import datetime, timedelta
from openai import OpenAI
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_socketio import SocketIO
import threading

class CoordinatorAgent:
    """
    The central brain of the multi-agent system.
    Orchestrates communication between agents, enriches event data,
    manages alert cooldowns, and triggers external workflows.
    """
    def __init__(self, config_path):
        """
        Initializes the CoordinatorAgent.

        Args:
            config_path (str): Path to the coordinator's configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Store config values in attributes for easy access
        self.n8n_webhook_url = self.config.get('n8n_webhook_url')
        self.n8n_assessment_webhook_url = self.config.get('n8n_assessment_webhook_url')
        self.maintenance_agent_api_url = self.config.get('maintenance_agent_api_url')
        self.camera_mapping = self.config.get('camera_to_machine_mapping', {})
        self.recent_events = [] # For event correlation
        self.event_memory_window_seconds = 300 # 5 minutes

        # --- LLM Initialization ---
        llm_config = self.config.get('llm_config', {})
        self.llm_model_name = llm_config.get('model_name')
        self.llm_client = None
        if self.llm_model_name:
            try:
                self.llm_client = OpenAI()
                self.llm_client.models.list() # Test the connection
                print(f"[CoordinatorAgent] OpenAI LLM client initialized successfully for model {self.llm_model_name}.")
            except Exception as e:
                print(f"[CoordinatorAgent] ERROR: Failed to initialize OpenAI client: {e}. LLM is disabled.")
        else:
            print("[CoordinatorAgent] Warning: No LLM model configured. LLM is disabled.")

        # Flask server for receiving maintenance reports and serving the dashboard
        self.app = Flask(__name__, template_folder='templates')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.host = os.getenv("COORDINATOR_HOST", "0.0.0.0")
        self.port = int(os.getenv("COORDINATOR_PORT", 8080))
        self._register_routes()

        self.server_thread = threading.Thread(
            target=lambda: self.socketio.run(self.app, host=self.host, port=self.port, allow_unsafe_werkzeug=True),
            daemon=True
        )
        self.server_thread.start()
        print(f"[CoordinatorAgent] Flask server with SocketIO started on http://{self.host}:{self.port}")

        self.last_alert_times = {}
        print("[CoordinatorAgent] Initialization complete.")

    def _register_routes(self):
        """Registers Flask routes."""
        @self.app.route('/report', methods=['POST'])
        def handle_report_post():
            report_data = request.json
            print(f"[CoordinatorAgent] Received maintenance report: {report_data.get('problem_detected')}")
            self.process_maintenance_report(report_data)
            return jsonify({"status": "success"}), 200

        @self.app.route('/')
        def index():
            """Serves the main dashboard page."""
            return render_template('index.html')

        @self.app.route('/reports/<path:filename>')
        def serve_report_image(filename):
            """Serves an image from the reports directory."""
            reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
            return send_from_directory(reports_dir, filename)

        @self.app.route('/evidence/<path:filename>')
        def serve_evidence_image(filename):
            """Serves an image from the safety_agent's evidence directory."""
            # Assumes the evidence directory is at ../safety_agent/data/evidence
            evidence_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'safety_agent', 'data', 'evidence'))
            return send_from_directory(evidence_dir, filename)

    def process_safety_event(self, event, camera_id):
        """Processes a safety event, enriches it, and triggers workflows."""
        event_type = event.get('event_type')
        print(f"\n[CoordinatorAgent] Received event: {event_type} from {camera_id}")

        machine_info = self.camera_mapping.get(camera_id, {})
        location = machine_info.get('location', 'Unknown')
        machine_id = machine_info.get('machine_id', 'Unknown')
        timestamp = datetime.now().isoformat()

        # Create the web-accessible URL for the evidence image
        evidence_path = event.get('evidence_path')
        evidence_url = None
        if evidence_path:
            file_name = os.path.basename(evidence_path)
            evidence_url = f'/evidence/{file_name}'

        # Prepare the data payload for the dashboard and internal processing
        event_data = {
            'eventType': event_type,
            'confidence': event.get('confidence', 'N/A'),
            'timestamp': timestamp,
            'location': location,
            'machineId': machine_id,
            'evidenceUrl': evidence_url
        }

        # Immediately broadcast the initial event to the dashboard
        self.broadcast_event('new_event', event_data)

        # Trigger the n8n voice call workflow for critical alerts
        self._trigger_n8n_workflow(self.n8n_webhook_url, event_data)

        # Store the event for context in future assessments
        self.recent_events.append(event_data)
        self._cleanup_old_events()

        # Asynchronously get and broadcast the LLM assessment
        correlated_events = self._find_correlated_events(event_data)
        assessment = self._invoke_holistic_llm_reasoning(event_data, correlated_events)
        event_data['assessment'] = assessment

        self.broadcast_event('event_update', {'timestamp': timestamp, 'assessment': assessment})
        self._trigger_n8n_workflow(self.n8n_assessment_webhook_url, event_data)

    def process_maintenance_report(self, report_data: dict):
        """Processes an incoming maintenance report."""
        print(f"--- [Processing Maintenance Report for {report_data.get('machine_id')}] ---")

        self.recent_events.append(report_data)
        self._cleanup_old_events()

        self.broadcast_event('new_event', report_data)

        correlated_events = self._find_correlated_events(report_data)
        assessment = self._invoke_holistic_llm_reasoning(report_data, correlated_events)
        report_data['assessment'] = assessment

        self.broadcast_event('event_update', {'timestamp': report_data['timestamp'], 'assessment': assessment})
        self._trigger_n8n_workflow(self.n8n_assessment_webhook_url, report_data)

    def broadcast_event(self, event_name, data):
        """Broadcasts an event to all connected dashboard clients."""
        with self.app.app_context():
            self.socketio.emit(event_name, data)
        print(f"[CoordinatorAgent] Broadcasted event '{event_name}' to dashboard.")

    def _trigger_n8n_workflow(self, webhook_url, payload):
        """Triggers a specified n8n workflow with the given payload."""
        if not webhook_url or 'YOUR' in webhook_url:
            print(f"[CoordinatorAgent] Skipping n8n trigger: Webhook URL '{webhook_url}' not configured.")
            return

        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"[CoordinatorAgent] Successfully triggered n8n workflow for webhook: {webhook_url}")
        except requests.exceptions.RequestException as e:
            print(f"[CoordinatorAgent] ERROR: Could not trigger n8n workflow at {webhook_url}: {e}")
        except TypeError as e:
            print(f"[CoordinatorAgent] ERROR: Payload is not JSON serializable: {e}")

    def _cleanup_old_events(self):
        """Removes old events from the memory window."""
        now = datetime.now()
        self.recent_events = [
            event for event in self.recent_events
            if now - datetime.fromisoformat(event['timestamp']) < timedelta(seconds=self.event_memory_window_seconds)
        ]

    def _find_correlated_events(self, current_event_data: dict) -> list:
        """Finds events in memory that are correlated to the current event (e.g., same machine)."""
        correlated_events = []
        current_machine_id = current_event_data.get('machineId') or current_event_data.get('machine_id')

        for event in self.recent_events:
            if event['timestamp'] == current_event_data['timestamp']:
                continue

            event_machine_id = event.get('machineId') or event.get('machine_id')
            if event_machine_id == current_machine_id:
                correlated_events.append(event)
        
        return correlated_events

    def _invoke_holistic_llm_reasoning(self, current_event, correlated_events):
        """
        Invokes the LLM to get a situational assessment and recommendation,
        considering both the current event and correlated past events.
        """
        if not self.llm_client:
            return "LLM assessment is disabled."

        system_prompt = "You are an AI safety and maintenance expert for an industrial facility. You will receive data about a primary event and a list of recent, correlated events from the same machine. Your task is to provide a holistic assessment that considers the full context. Analyze the situation, identify trends or escalating issues, and provide a clear, actionable recommendation."
        
        correlated_events_str = "No recent correlated events."
        if correlated_events:
            event_lines = []
            for e in correlated_events:
                event_type = e.get('eventType', 'N/A')
                timestamp = e.get('timestamp', 'N/A')
                details = e.get('problem_detected') or f"Confidence: {e.get('confidence', 'N/A')}"
                event_lines.append(f"- Type: {event_type}, Time: {timestamp}, Details: {details}")
            correlated_events_str = "\n".join(event_lines)

        user_prompt = f"""
        **Primary Event:**
        - Event Type: {current_event.get('eventType') or current_event.get('type', 'N/A')}
        - Location: {current_event.get('location', 'N/A')}
        - Machine ID: {current_event.get('machineId') or current_event.get('machine_id', 'N/A')}
        - Timestamp: {current_event.get('timestamp', 'N/A')}
        - Details: {current_event.get('problem_detected') or f"Confidence: {current_event.get('confidence', 'N/A')}"}

        **Recent Correlated Events (from the same machine):**
        {correlated_events_str}

        **Assessment Task:**
        1.  **Analyze:** Based on the primary event AND the correlated history, what is the most likely situation?
        2.  **Recommend:** What is the single most important, immediate action that should be taken? Prioritize safety.
        """

        try:
            print(f"[CoordinatorAgent] Querying LLM with {len(correlated_events)} correlated event(s)...")
            response = self.llm_client.chat.completions.create(
                model=self.llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
            )
            assessment = response.choices[0].message.content.strip()
            print(f"[CoordinatorAgent] LLM Assessment: {assessment}")
            return assessment

        except Exception as e:
            print(f"[CoordinatorAgent] ERROR: LLM invocation failed: {e}")
            return "LLM assessment failed due to an internal error."
