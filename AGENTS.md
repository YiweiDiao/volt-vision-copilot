# Development Rules for Volt Vision Copilot

## Goal
Build a Kaggle capstone project for the "Agents for Business" track.

## Required concepts
- Google ADK agent
- Local MCP server
- Security features
- Reproducible local setup

## Stack
- Python 3.11
- Project-local .venv
- pyproject.toml
- Streamlit
- Google ADK
- MCP
- pandas, numpy, plotly
- pydantic
- pytest
- SAIA only as an optional LLM backend

## Core boundary
Deterministic code decides whether a cycle is a suspected deviation.
The LLM must not diagnose a confirmed fault, confirmed tool wear,
or a confirmed root cause.

## Safety requirements
- No automatic machine control.
- No PLC, SCADA, MES, OPC-UA, or Modbus integration.
- No API key in source code, tests, README, screenshots, or Git.
- Use .env locally and .env.example in Git.
- Send only structured event summaries to an LLM, never raw traces.
- Use "suspected deviation" and "manual inspection recommended".

## Implementation approach
1. Implement deterministic power monitoring first.
2. Create a local read-only MCP server second.
3. Add an ADK investigation agent third.
4. Add Streamlit human approval and event feedback fourth.
5. Add SAIA only after the local workflow works.

## Non-goals
Do not add React, Docker, cloud deployment, login systems,
real industrial protocols, or multi-agent complexity in the first MVP.

## Quality
- Add tests for deterministic functions.
- Use typed Pydantic models for all event payloads.
- Keep every tool small and independently testable.
- Run tests after each completed task.