from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DOC_PATH = REPO_ROOT / "docs" / "evaluation_scenarios.md"

EXPECTED_SCENARIO_IDS = tuple(f"EV-{index:02d}" for index in range(1, 11))
REQUIRED_FIELDS = (
    "Scenario ID",
    "Purpose",
    "Input/setup",
    "Expected deterministic status",
    "Expected workflow state",
    "Expected Copilot availability",
    "Expected execution mode if invoked",
    "Expected safety/persistence behavior",
    "Automated test coverage status",
    "Screenshot/demo relevance",
)


def test_evaluation_scenario_registry_has_exactly_ten_fixed_scenarios() -> None:
    text = SCENARIO_DOC_PATH.read_text(encoding="utf-8")

    scenario_ids = tuple(re.findall(r"^### (EV-\d{2}) - ", text, flags=re.MULTILINE))

    assert scenario_ids == EXPECTED_SCENARIO_IDS


def test_each_evaluation_scenario_contains_required_acceptance_fields() -> None:
    text = SCENARIO_DOC_PATH.read_text(encoding="utf-8")
    sections = re.split(r"^### (EV-\d{2}) - .*$", text, flags=re.MULTILINE)
    scenario_sections = dict(zip(sections[1::2], sections[2::2], strict=True))

    assert tuple(scenario_sections) == EXPECTED_SCENARIO_IDS
    for scenario_id, section in scenario_sections.items():
        for field in REQUIRED_FIELDS:
            assert f"| {field} |" in section, f"{scenario_id} is missing {field}"
