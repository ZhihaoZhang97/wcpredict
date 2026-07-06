"""Canonical data model for the World Cup 2026 prediction pipeline.

Every raw JSON record is converted into one of these frozen dataclasses at
load time (see datastore.py / normalizer.py). Downstream code — features,
prompt assembly, the agent — only ever sees these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# A (team1, team2) goal pair, e.g. (2, 0).
Pair = tuple[int, int]

# Stage constants, ordered by tournament progression.
STAGE_QUALIFYING_PLAYOFF = "qualifying_playoff"
STAGE_GROUP = "group"
STAGE_ROUND_OF_32 = "round_of_32"
STAGE_ROUND_OF_16 = "round_of_16"
STAGE_QUARTER_FINAL = "quarter_final"
STAGE_SEMI_FINAL = "semi_final"
STAGE_THIRD_PLACE = "third_place"
STAGE_FINAL = "final"

# How a played match was decided.
DECIDED_REGULATION = "regulation"
DECIDED_EXTRA_TIME = "extra_time"
DECIDED_PENALTIES = "penalties"


@dataclass(frozen=True)
class Club:
    name: str
    country: str


@dataclass(frozen=True)
class Player:
    id: str  # stable: "<fifa_code>:<name-slug>:<date_of_birth>"
    name: str
    team_code: str  # FIFA code of the national team
    number: int
    position: str  # GK / DF / MF / FW
    date_of_birth: str  # ISO date
    club: Club


@dataclass(frozen=True)
class Team:
    name: str  # canonical display name, e.g. "South Korea"
    fifa_code: str
    group: str  # "A".."L"
    confederation: str
    continent: str
    alt_name: Optional[str] = None  # e.g. "Korea Republic"


@dataclass(frozen=True)
class Goal:
    scorer_raw: str  # name exactly as it appears in the source file
    minute: str  # raw, may include stoppage time e.g. "90+3"
    player_id: Optional[str]  # resolved against the squad; None if unresolved
    penalty: bool = False
    own_goal: bool = False

    @property
    def minute_sort_key(self) -> tuple[int, int]:
        base, _, extra = self.minute.partition("+")
        return (int(base), int(extra or 0))


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-period scores; a period absent from the source is None.

    ft_90 is always the score after 90 minutes. For matches that went to
    extra time, `et` is the score after 120 minutes (goals included), and
    `pens` is the shootout tally (not goals). Playoff records sometimes
    omit ht/ft_90 entirely.
    """

    ht: Optional[Pair] = None
    ft_90: Optional[Pair] = None
    et: Optional[Pair] = None
    pens: Optional[Pair] = None

    @property
    def final(self) -> Optional[Pair]:
        """Goals actually scored (excludes the penalty shootout)."""
        return self.et if self.et is not None else self.ft_90


@dataclass(frozen=True)
class Match:
    source: str  # "tournament" | "qualifying_playoff"
    stage: str  # one of the STAGE_* constants
    round_label: str  # raw round string from the source file
    date: str  # ISO date
    team1: str  # canonical team name where resolvable, else raw name
    team2: str
    score: Optional[ScoreBreakdown]  # None if not yet played
    winner: Optional[str]  # team name; None for draws and unplayed matches
    decided_by: Optional[str]  # DECIDED_* constant; None if unplayed
    goals1: tuple[Goal, ...] = ()
    goals2: tuple[Goal, ...] = ()
    group: Optional[str] = None  # "Group A" for group-stage matches
    venue: Optional[str] = None
    time: Optional[str] = None
    num: Optional[int] = None  # official match number, when present

    @property
    def played(self) -> bool:
        return self.score is not None

    @property
    def is_draw(self) -> bool:
        return self.played and self.winner is None

    def involves(self, team_name: str) -> bool:
        return team_name in (self.team1, self.team2)

    def opponent_of(self, team_name: str) -> str:
        if team_name == self.team1:
            return self.team2
        if team_name == self.team2:
            return self.team1
        raise ValueError(f"{team_name!r} did not play in this match")
