"""Pipeline state types and free-text input parsing."""

from __future__ import annotations

import operator
import re
from typing import Annotated, Optional, TypedDict

from .schema import MatchPrediction, Stage

# Free-text stage input -> canonical stage. Checked in order.
_STAGE_PATTERNS: list[tuple[str, Stage]] = [
    (r"group|matchday", "group"),
    (r"32", "round_of_32"),
    (r"16|1\s*/\s*8|last\s*sixteen", "round_of_16"),
    (r"quarter|1\s*/\s*4", "quarter_final"),
    (r"semi|1\s*/\s*2", "semi_final"),
    (r"third|3rd|bronze", "third_place"),
    (r"final", "final"),
]


def parse_stage(text: str) -> Stage:
    lowered = text.lower()
    for pattern, stage in _STAGE_PATTERNS:
        if re.search(pattern, lowered):
            return stage
    raise ValueError(
        f"could not understand stage {text!r} — try e.g. 'group', 'round of 16', "
        "'quarter final', 'semi final' or 'final'"
    )


class PipelineState(TypedDict, total=False):
    # inputs
    team1_text: str
    team2_text: str
    stage: Stage
    as_of_date: Optional[str]
    # gather_data
    team1: str
    team2: str
    team1_report: str
    team2_report: str
    head_to_head: str
    # run_searches
    condense_tasks: list[dict]
    search_summary: str  # e.g. "54 searches in 6.2s" (for tracing)
    # condense (parallel fan-out; the reducer concatenates notes)
    research_notes: Annotated[list[str], operator.add]
    # predict
    prediction: MatchPrediction


class CondenseTask(TypedDict):
    """Payload sent to one parallel condense instance (one per team)."""

    team: str
    opponent: str
    stage: str
    material: str  # raw web search snippets for the whole squad + team news
