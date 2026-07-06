# Volt Vision Copilot

Volt Vision Copilot is a patent-inspired educational prototype for
human-in-the-loop screening of recurring power-cycle behavior in CNC and other
electrically driven production equipment.

## Project Overview

This Kaggle Agents for Business capstone explores non-invasive monitoring from
electrical power-cycle data. A deterministic DTW-based monitoring engine
calibrates from known-good cycles, compares a candidate cycle against the
calibrated reference, and assigns a screening status. Copilot support is
bounded to human investigation: it explains saved structured evidence and
suggests inspection questions, while real-world action remains a human decision.

The prototype is not an industrially validated diagnosis, deployment, or
predictive-maintenance product.

## What The Prototype Demonstrates

- Known-good calibration cycles.
- Deterministic DTW comparison and calibrated threshold.
- `within_normal_band` versus `suspected_deviation` screening status.
- Explicit local event save before investigation.
- Local MCP evidence layer with read-only event and guidance evidence.
- Optional ADK Copilot guidance or deterministic fallback.
- Human-review acknowledgement before local feedback can be recorded.

## Safety And Scope Boundary

- Deterministic monitoring is the only authority for screening status.
- Copilot does not diagnose confirmed faults, root causes, or tool wear.
- No PLC, SCADA, MES, OPC-UA, Modbus, machine control, shutdown, repair, or
  automatic ticket creation is included.
- Manual inspection is recommended before any action.
- Feedback does not update calibration, thresholds, or detection status.
- The demo uses synthetic data, not external customer data.

## Architecture

See [docs/architecture_diagram.md](docs/architecture_diagram.md) for the public
Mermaid architecture diagram and component notes.

In short, deterministic monitoring builds a reference template and calibrated
threshold from known-good cycles, then compares a candidate cycle against that
reference. A saved suspected-deviation event can be investigated through a local
MCP evidence layer that exposes read-only event metrics, curated guidance, and
similar local events. The optional ADK Copilot can turn that evidence into
bounded guidance, or the workflow falls back to deterministic guidance. Human
acknowledgement gates local feedback, and feedback does not change detection.

## Demo Video

Demo video: published separately for the capstone submission.
It demonstrates normal screening, suspected-deviation save gating, bounded
guidance, and acknowledgement-gated feedback.

## Stable Demo Workflow

Verified public normal path:

```text
Synthetic normal cycle
-> within_normal_band
-> continue monitoring
```

Verified public suspected-deviation path:

```text
Synthetic changed cycle
-> suspected_deviation
-> explicit local save
-> bounded guidance
-> human acknowledgement
-> local feedback
```

## CSV And Current Public Demo Scope

CSV validation and deterministic screening are included. The stable synthetic
workflow is the public end-to-end demo. The CSV saved-event-to-Copilot
continuation is not presented as a completed public workflow.

Synthetic CSV examples are available in
[examples/csv_demo/](examples/csv_demo/).

## Quick Start

Prerequisites:

- Python 3.11
- `uv`
- PowerShell on Windows, or equivalent shell commands on another platform

Create the project-local environment from the locked project configuration:

```powershell
uv sync --extra dev
```

Run the test suite:

```powershell
uv run pytest
```

Start the Streamlit app:

```powershell
uv run streamlit run src/volt_vision/ui/app.py
```

Optional model-assisted guidance can be configured locally from the tracked
environment template. Leave local credentials out of Git. Without local model
configuration, the app uses deterministic fallback guidance.

## Evaluation And Evidence

- [docs/evaluation_scenarios.md](docs/evaluation_scenarios.md)
- [docs/demo_evidence.md](docs/demo_evidence.md)
- [docs/architecture_diagram.md](docs/architecture_diagram.md)

The project defines 10 fixed evaluation scenarios covering normal screening,
suspected-deviation save gating, fallback behavior, CSV validation, malformed or
insufficient CSV handling, and acknowledgement-gated feedback.

## Repository Structure

```text
src/volt_vision/        Application package: monitoring, UI, agent, MCP, guidance
tests/                  Pytest coverage for deterministic and workflow behavior
docs/                   Architecture, evaluation, demo evidence, and video docs
examples/csv_demo/      Synthetic CSV demo inputs
knowledge/              Local curated guidance content
scripts/                Local smoke and diagnostic scripts
data/.gitkeep           Placeholder for local demo persistence
pyproject.toml          Python package and dependency metadata
uv.lock                 Reproducible dependency lockfile
```

## Technology Summary

- Python
- Streamlit
- Deterministic DTW monitoring
- Google ADK
- Local read-only MCP-style evidence layer
- Local JSONL persistence where applicable
- pytest

## Non-Goals

- Industrial deployment.
- Real customer data.
- Autonomous maintenance decisions.
- Machine control.
- Automatic model training from feedback.
