# Agenix-Eye: Multi-Agent Industrial Safety & Maintenance System

🏆 Finalist - Hackathon "Agentic AI for Industry" (Teal Technology Services)

Agenix-Eye is an advanced, multi-agent system designed to proactively monitor and manage industrial environments. It leverages computer vision and LLM-powered reasoning to detect safety hazards, predict maintenance needs, and orchestrate intelligent, context-aware responses, all visualized on a real-time dashboard.

## Core Features

*   **Multi-Agent Collaboration:** Specialized agents for Safety and Maintenance work in concert, orchestrated by a central Coordinator.
*   **Real-Time Dashboard:** A Flask-SocketIO web interface provides a live feed of all events, on-demand LLM assessments, and a modal pop-up for viewing visual evidence.
*   **LLM-Powered Holistic Reasoning:** The system correlates events to understand root causes and recommend prioritized action plans.
*   **Visual & Data-Driven Detection:** The `SafetyAgent` uses computer vision for physical hazards, while the `MaintenanceAgent` analyzes machine logs for anomalies.
*   **Intelligent Orchestration:** The `CoordinatorAgent` acts as the central brain, managing communication, enriching data with location/machine context, and triggering external workflows (e.g., via n8n).
*   **Diverse Event Handling:** The dashboard distinguishes between multiple event types with unique styling: `fire` (red), `fall` (blue), `spill` (yellow), `maintenance` (green), `machine_failure` (orange), and `machine_error` (dark red).

## System Architecture

The system uses a unified simulation launcher (`main.py`) that initializes and runs all agents in parallel. The `CoordinatorAgent` serves as the central hub for all communication and dashboard updates.

```
+-------------------+      (Video Stream)      +-----------------+
|      Cameras      | -----------------------> |   SafetyAgent   |
+-------------------+                          +-----------------+
        |                                               |
(Simulated Logs)                                        | (Event Data)
        |                                               v
+-------------------+                          +--------------------+
|  Machine Sensors  |                          |  CoordinatorAgent  |
+-------------------+                          | (Flask + LLM)      |
        |                                      +--------------------+
        | (API Call)                                    ^          |
        v                                               |          | (Socket.IO)
+--------------------+                                |          v
|  MaintenanceAgent  | -------------------------------+          +-----------------+
| (FastAPI + RAG)    |        (Maintenance Report)             |   Web Dashboard |
+--------------------+                                         +-----------------+
```

**Data Flow:**
1.  **Initialization:** The `main.py` script loads all configurations, initializes the `SafetyAgent`, `MaintenanceAgent`, and `CoordinatorAgent`.
2.  **Parallel Operation:** All agents run in separate threads.
3.  **Safety Events:** The `SafetyAgent` processes video streams. On detection, it calls the `CoordinatorAgent` directly, passing the raw event data.
4.  **Maintenance Events:** The `MaintenanceAgent` runs a FastAPI server. The simulation can trigger it via API calls. It analyzes the request, uses its RAG system, and sends a structured report to the `CoordinatorAgent`'s API endpoint.
5.  **Coordination & Enrichment:** The `CoordinatorAgent` receives all events. It enriches them with machine/location data, performs LLM-powered correlation and assessment, and broadcasts updates to the dashboard via Socket.IO.
6.  **Visualization:** The user opens the dashboard in a web browser, which connects to the `CoordinatorAgent`'s Socket.IO server to receive and display live events and assessments.

## Real-Time Dashboard

The dashboard is the primary user interface for monitoring the system. It features:
*   **Live Event Feed:** A real-time, color-coded stream of all detected events.
*   **Assessment Panel:** A dedicated section to view detailed, LLM-generated analysis for any event by clicking the "View Assessment" button.
*   **Evidence Modal:** For events with visual evidence, a "View Evidence" button appears. Clicking it opens the image in a clean, on-page modal pop-up.

## Configuration

1.  **`.env` file:** Create a `.env` file in the project root to store secrets.
    ```shell
    OPENAI_API_KEY="your-openai-api-key"
    ```
2.  **`coordination_agent/config.yaml`:** Configure webhooks, agent URLs, and the camera-to-machine mapping.
3.  **`simulation_config.yaml`:** Configure camera sources and other simulation parameters.

## How to Run

1.  **Setup Environment:**
    ```bash
    # Create a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install dependencies
    pip install -r requirements.txt
    ```
2.  **Prepare Maintenance Agent Data (Optional):**
    *   If you want the `MaintenanceAgent` to use its RAG capabilities, ensure your machine manuals (in Markdown format) are in the `data/machine_manuals` directory.
    *   Run the ingestion script once to build the vector database: `python maintenance_agent/rag_system.py`
3.  **Launch Simulation:**
    ```bash
    python main.py
    ```
4.  **View Dashboard:**
    *   Open your web browser and navigate to the URL provided by the Coordinator Agent upon startup (e.g., `http://127.0.0.1:5000`).
