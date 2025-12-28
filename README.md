<div align="center">
  <br />
  <img src="assets/logo.png" width="350" alt="AstraGuard AI Logo">
  <br />

# 🛰️ AstraGuard AI

**Autonomous Fault Detection & Recovery for CubeSats**
*Powered by Pathway's Streaming Engine & Biologically-Inspired Memory*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with Pathway](https://img.shields.io/badge/Built%20with-Pathway-3535EE.svg)](https://pathway.com/)
[![Hackathon](https://img.shields.io/badge/Hackathon-Synaptix%20Frontier-orange)](https://github.com/sr-857/AstraGuard)

<br />

**Explore the Project:**

🌐 **[Live Site](https://sr-857.github.io/AstraGuard/)** | 📊 **[Dashboard Prototype](https://astraguarddashboard.lovable.app/)** | 📚 **[Documentation](docs/TECHNICAL.md)** | 🎥 **[Demo Video](https://drive.google.com/file/d/1pKkZosBJdk8tvfBrqNelPeWyY265eOcI/view?usp=drive_link)** | 🗂️ **[Resources (GDrive)](https://drive.google.com/drive/folders/1j3Ra1_q6v-lEavG40eR2mRzVLcqDYYKH)**

📌 **[View Official Report](https://drive.google.com/file/d/17Vvhz9CNO_fVLpxJTH2eKSktByT3HnKt/view)**

</div>

---

## 🚀 Mission Brief

**Space is unforgiving. AstraGuard makes it manageable.**

AstraGuard AI is an intelligent autonomous system designed to protect CubeSats from catastrophic failure. Unlike traditional "alert-only" systems, AstraGuard uses **agentic reasoning** to detect anomalies in real-time streams, recall historical context using adaptive memory, and execute autonomous recovery actions—all with sub-second latency.

> **"Streaming anomalies. Self-updating memory. Decisions that trigger actions."**

---

## ⚡ System Preview

> *[Tip: Place a GIF here showing the Dashboard detecting an anomaly and the Agent auto-correcting it]*

---

## 🏆 Innovation & Differentiators

We moved beyond static RAG to build a system that *learns* from the stream.

| Feature | The Old Way (Static) | The AstraGuard Way (Adaptive) |
| :--- | :--- | :--- |
| **Data Processing** | Batch processing (slow) | **Streaming Intelligence (5Hz)** via Pathway |
| **Memory** | Static vector databases | **Evolving Memory** with decay & pruning |
| **Response** | Passive Alerts | **Agentic Action** (detect → reason → act) |
| **Explainability** | Black box "magic" | **Transparent Decision Traces** |
| **User Interface** | Terminal logs | **Frontier Dashboard** with neural viz |

---

## 🏗️ Architecture

AstraGuard relies on a modular, feedback-loop architecture. The system ingests telemetry, encodes it into vectors, and uses a biologically inspired "Dragon Hatchling" memory model to reason about the state of the satellite.

```mermaid
graph TD
    subgraph "Stream Ingestion"
    A["🛰️ Telemetry Stream"] -->|Pathway Engine| B["📊 Embedding Encoder"]
    end

    subgraph "The Brain (Agentic Core)"
    B -->|Vectors| C["🧠 Adaptive Memory Store"]
    C -->|Context Retrieval| D["🤖 Anomaly Reasoning Agent"]
    B -->|Real-time Event| D
    end

    subgraph "Action Layer"
    D -->|Decision| E["⚡ Response Orchestrator"]
    E -->|Command| F["🛰️ System Recovery"]
    end

    subgraph "Feedback Loop"
    F -->|Outcome| C
    end
🧱 Modular Designmemory_engine/: Adaptive store with temporal weighting & recurrence scoring.anomaly_agent/: The decision loop (LLM-assisted reasoning).response_orchestrator/: Maps decisions to concrete system commands.dashboard/: Real-time Streamlit visualization.🎯 Hackathon Tracks🤖 Track 1: Agentic AI (Applied GenAI)The Implementation: A reactive agent that doesn't just chat—it acts.Workflow: Live Input → Anomaly Detection → Memory Recall → Intelligent Decision → Automated ActionLatency: < 2s from detection to resolution.🧠 Track 2: The Frontier (Research & Deep Tech)The Research: Implementing the Dragon Hatchling (BDH) memory architecture.Innovation: We implemented Recurrence Resonance Scoring—a physics-inspired formula to reinforce repeated anomaly signals.Formula:$$Resonance = I_{base} \times (1 + 0.3 \times \log(1 + N_{recurrence})) \times D_{time}$$🛠️ Getting StartedPrerequisitesPython 3.9+Pip & Git⚡ Quick InstallBash# 1. Clone the repository
git clone [https://github.com/sr-857/AstraGuard.git](https://github.com/sr-857/AstraGuard.git)
cd AstraGuard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python verify_install.py
🎮 Run the System1. Start the Autonomous Agent:Bashpython examples/run_demo.py
2. Launch the Mission Control Dashboard:Bashstreamlit run dashboard/app.py
Tip: Toggle "Frontier Mode" in the dashboard to visualize neural activity!📊 Performance BenchmarkMetricTargetActual PerformanceStatusReaction Time< 2s~325ms🟢 ExceedsRetrieval Speed< 50ms~38ms🟢 ExceedsMemory UpdatesAutoReal-time🟢 ActiveFalse Positives< 5%~2.1%🟢 Stable🧪 Testing & ValidationWe maintain rigorous testing standards to ensure mission safety.Bash# Run full test suite
pytest tests/ -v

# Test memory dynamics specifically
pytest tests/test_memory_store.py -v
📄 Citation & ResearchIf you use AstraGuard or its memory architecture in your research, please cite:Code snippet@misc{AstraGuardAI2025,
  author = {Roy, Subhajit},
  title = {AstraGuard AI: Streaming Anomaly Detection with Adaptive Memory},
  year = {2025},
  publisher = {GitHub},
  journal = {Synaptix Frontier AI Hackathon @ IIT Madras},
  note = {Track 1: Agentic AI & Track 2: Frontier Tech}
}
<div align="center">Code is Liability. Intelligence is Asset.Made with ❤️ for the Synaptix Frontier AI HackathonDocumentation • Report • Video</div>
