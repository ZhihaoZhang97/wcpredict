"""Structured prediction output for the agent pipeline."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Stages the user can ask about, normalized from free text like "1/4 final".
Stage = Literal[
    "group",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "third_place",
    "final",
]


class MatchPrediction(BaseModel):
    """The agent's final answer for one fixture."""

    team1: str = Field(description="Canonical name of the first team")
    team2: str = Field(description="Canonical name of the second team")
    stage: Stage

    outcome: Literal["team1_win", "team2_win", "draw"] = Field(
        description=(
            "Final match outcome. 'draw' is only valid for group-stage "
            "matches; knockout matches must name a winner even if it takes "
            "extra time or penalties."
        )
    )
    # Deliberately ordered before predicted_score: the model must commit
    # to a goal-expectation profile first, and the score follows from it.
    expected_goals_team1: float = Field(
        ge=0,
        le=6,
        description=(
            "team1's expected goals over 90 minutes, from their attacking "
            "evidence vs team2's defensive evidence"
        ),
    )
    expected_goals_team2: float = Field(
        ge=0,
        le=6,
        description="team2's expected goals over 90 minutes, same basis",
    )
    predicted_score: str = Field(
        description=(
            "Predicted score after 90 minutes from team1's perspective, "
            "e.g. '2:1' or '1:1'. This must be the most likely score "
            "CONDITIONAL on your predicted path: if decided_by is "
            "regulation, the winner must be ahead in it (with asymmetric "
            "expected goals that means an asymmetric score, e.g. 1:2 — "
            "never a hedged 1:1); a level score is only allowed when "
            "decided_by is extra_time or penalties."
        ),
        pattern=r"^\d{1,2}:\d{1,2}$",
    )
    decided_by: Optional[Literal["regulation", "extra_time", "penalties"]] = Field(
        description=(
            "How the match is decided. Required for knockout stages; null "
            "for group-stage matches."
        )
    )

    prob_team1_win: float = Field(ge=0, le=1)
    prob_draw: float = Field(
        ge=0, le=1, description="Probability of a draw after 90 minutes"
    )
    prob_team2_win: float = Field(ge=0, le=1)

    key_factors: list[str] = Field(
        description="3-6 decisive factors, each one short sentence",
        min_length=3,
        max_length=6,
    )
    reasoning: str = Field(
        description="Concise analyst's reasoning behind the prediction, one paragraph"
    )
