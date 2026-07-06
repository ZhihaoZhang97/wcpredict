"""Normalize heterogeneous raw match records into the canonical shape.

The source files encode scores in five different shapes:

    {"ft": [2,0], "ht": [1,0]}              regulation result
    {"ft": ..., "ht": ..., "et": ...}       decided in extra time
    {"ft": ..., "ht": ..., "et": ..., "p": ...}  decided on penalties
    [2, 0]                                  playoff shorthand (regulation)
    {"et": ...} / {"et": ..., "p": ...}     playoff records missing ht/ft
    null / absent                           not yet played

This module is the only place that knows about that mess. The decision
order for the winner is: penalties > extra time > 90-minute score.
"""

from __future__ import annotations

from typing import Optional

from .models import (
    DECIDED_EXTRA_TIME,
    DECIDED_PENALTIES,
    DECIDED_REGULATION,
    STAGE_FINAL,
    STAGE_GROUP,
    STAGE_QUALIFYING_PLAYOFF,
    STAGE_QUARTER_FINAL,
    STAGE_ROUND_OF_16,
    STAGE_ROUND_OF_32,
    STAGE_SEMI_FINAL,
    STAGE_THIRD_PLACE,
    Pair,
    ScoreBreakdown,
)

_KNOCKOUT_STAGES = {
    "Round of 32": STAGE_ROUND_OF_32,
    "Round of 16": STAGE_ROUND_OF_16,
    "Quarter-final": STAGE_QUARTER_FINAL,
    "Semi-final": STAGE_SEMI_FINAL,
    "Match for third place": STAGE_THIRD_PLACE,
    "Final": STAGE_FINAL,
}


def parse_stage(round_label: str, group: Optional[str], source: str) -> str:
    if source == "qualifying_playoff":
        return STAGE_QUALIFYING_PLAYOFF
    if group:
        return STAGE_GROUP
    try:
        return _KNOCKOUT_STAGES[round_label]
    except KeyError:
        raise ValueError(f"unrecognised round label: {round_label!r}") from None


def _pair(value) -> Optional[Pair]:
    if value is None:
        return None
    a, b = value
    return (int(a), int(b))


def normalize_score(raw) -> tuple[Optional[ScoreBreakdown], Optional[int], Optional[str]]:
    """Return (breakdown, winner_index, decided_by) for a raw score value.

    winner_index is 0 for team1, 1 for team2, None for a draw (or an
    unplayed match, where all three values are None).
    """
    if raw is None:
        return None, None, None

    if isinstance(raw, list):
        breakdown = ScoreBreakdown(ft_90=_pair(raw))
    elif isinstance(raw, dict):
        breakdown = ScoreBreakdown(
            ht=_pair(raw.get("ht")),
            ft_90=_pair(raw.get("ft")),
            et=_pair(raw.get("et")),
            pens=_pair(raw.get("p")),
        )
    else:
        raise ValueError(f"unrecognised score shape: {raw!r}")

    if breakdown.pens is not None:
        decider, decided_by = breakdown.pens, DECIDED_PENALTIES
    elif breakdown.et is not None:
        decider, decided_by = breakdown.et, DECIDED_EXTRA_TIME
    elif breakdown.ft_90 is not None:
        decider, decided_by = breakdown.ft_90, DECIDED_REGULATION
    else:
        raise ValueError(f"score has no decisive period: {raw!r}")

    if decider[0] > decider[1]:
        winner_index = 0
    elif decider[1] > decider[0]:
        winner_index = 1
    else:
        winner_index = None
    return breakdown, winner_index, decided_by
