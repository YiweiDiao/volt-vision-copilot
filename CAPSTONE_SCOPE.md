# Volt Vision Copilot — Capstone Scope

## Kaggle track
Agents for Business

## Project title
Volt Vision Copilot: A Human-in-the-Loop Agent for CNC Power-Deviation Investigation

## Problem
Small and medium manufacturing companies often lack a low-cost,
non-invasive way to understand abnormal process cycles from machine
power data. A power-signature deviation alone is difficult for an
operator to interpret and does not prove a root cause.

## Solution
A local prototype that:
1. calibrates normal CNC power cycles,
2. detects suspected deviations using deterministic DTW, duration,
   energy, and peak-power indicators,
3. invokes an ADK agent only after a suspected deviation,
4. retrieves maintenance guidance and historical event evidence,
5. produces a bounded inspection recommendation for human approval.

## Capstone concepts to demonstrate
1. Google ADK agent with tool use.
2. Local MCP server exposing read-only monitoring tools.
3. Security features:
   - secrets only in .env,
   - strict input schemas,
   - tool allowlist,
   - no automatic machine control,
   - no root-cause certainty claims,
   - human approval before any follow-up action.
4. Reproducibility / deployability:
   - pyproject.toml,
   - .venv,
   - setup instructions,
   - public GitHub repository.

## Core user
Maintenance manager, production manager, or CNC shop owner.

## Non-goals
- No PLC, SCADA, MES, OPC-UA, Modbus, or real machine control.
- No claim of confirmed tool wear or machine failure.
- No real customer production data in the demo.
- No automatic maintenance ticket creation or automatic shutdown.

## Data
Synthetic demonstration power traces and synthetic maintenance guidance.

## Model boundary
The deterministic monitoring engine is the source of truth for
suspected deviation detection. The LLM only explains structured
events and retrieves relevant guidance.

## Deliverables
- Streamlit demonstration
- Google ADK agent
- MCP server
- Tests
- README
- Architecture diagram
- <= 5-minute YouTube demo
- Kaggle Writeup