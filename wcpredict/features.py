"""Team reports: tournament history + squad, computed from the DataStore.

This is the evidence layer for the pure-LLM predictor. Given a team (free
text is fine) it produces a structured TeamReport — match-by-match 2026
history including how each game was decided, upcoming fixtures, and the
squad with per-player tournament goals — plus a plain-text rendering
suitable for a prompt or a terminal.

Every function takes as_of_date (ISO); only matches strictly before that
date are visible, so reports built for a prediction never leak the result
being predicted, and backtesting against played matches is free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from typing import Optional

from .datastore import DataStore
from .models import (
    DECIDED_EXTRA_TIME,
    DECIDED_PENALTIES,
    STAGE_FINAL,
    STAGE_GROUP,
    STAGE_QUALIFYING_PLAYOFF,
    Match,
    Player,
    Team,
)

_POSITION_ORDER = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}


@dataclass(frozen=True)
class MatchSummary:
    """One match from a single team's perspective."""

    date: str
    source: str
    stage: str
    round_label: str
    opponent: str
    goals_for: int
    goals_against: int
    result: str  # "W" / "D" / "L"
    decided_by: str
    score_line: str  # e.g. "1-1 a.e.t., 3-4 pens"
    venue: Optional[str]


@dataclass(frozen=True)
class Fixture:
    date: str
    stage: str
    round_label: str
    opponent: str
    venue: Optional[str]


@dataclass(frozen=True)
class TeamForm:
    team: str
    as_of_date: Optional[str]
    played: tuple[MatchSummary, ...]  # chronological, playoffs included
    upcoming: tuple[Fixture, ...]
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    extra_time_games: int  # games that went past 90 minutes
    shootout_record: tuple[int, int]  # (won, lost)
    status: str  # human-readable tournament status


@dataclass(frozen=True)
class PlayerStats:
    player: Player
    age: int
    goals: int
    penalty_goals: int
    goal_events: tuple[str, ...] = ()  # e.g. "67' vs Czech Republic (group)"


@dataclass(frozen=True)
class TeamReport:
    team: Team
    form: TeamForm
    squad: tuple[PlayerStats, ...]  # ordered GK, DF, MF, FW, then number


# ------------------------------------------------------------------ form


def _summarize(match: Match, team: str) -> MatchSummary:
    us = 0 if match.team1 == team else 1
    them = 1 - us
    final = match.score.final
    gf, ga = final[us], final[them]

    line = f"{gf}-{ga}"
    if match.decided_by == DECIDED_PENALTIES:
        pens = match.score.pens
        line = f"{gf}-{ga} a.e.t., {pens[us]}-{pens[them]} pens"
    elif match.decided_by == DECIDED_EXTRA_TIME:
        line = f"{gf}-{ga} a.e.t."

    if match.winner == team:
        result = "W"
    elif match.winner is None:
        result = "D"
    else:
        result = "L"

    return MatchSummary(
        date=match.date,
        source=match.source,
        stage=match.stage,
        round_label=match.round_label,
        opponent=match.opponent_of(team),
        goals_for=gf,
        goals_against=ga,
        result=result,
        decided_by=match.decided_by,
        score_line=line,
        venue=match.venue,
    )


def _status(played: list[MatchSummary], upcoming: list[Fixture]) -> str:
    for m in played:
        if m.source == "tournament" and m.stage not in (STAGE_GROUP,) and m.result == "L":
            if m.stage == STAGE_FINAL:
                return f"runner-up (lost the final to {m.opponent})"
            return f"eliminated in {m.round_label} by {m.opponent} ({m.score_line})"
    for m in played:
        if m.stage == STAGE_FINAL and m.result == "W":
            return "champion"
    if upcoming:
        nxt = upcoming[0]
        return f"still in — next: {nxt.round_label} vs {nxt.opponent} on {nxt.date}"
    tournament = [m for m in played if m.source == "tournament"]
    if tournament and all(m.stage == STAGE_GROUP for m in tournament) and len(tournament) >= 3:
        return "eliminated in the group stage"
    return "awaiting next fixture"


def team_form(store: DataStore, team_name: str, as_of_date: Optional[str] = None) -> TeamForm:
    matches = store.get_matches(team=team_name)
    played = [
        _summarize(m, team_name)
        for m in matches
        if m.played and (as_of_date is None or m.date < as_of_date)
    ]
    # Fixture visibility: without as_of_date, everything unplayed. With it,
    # only the team's next match on/after that date — later fixtures would
    # leak bracket progression (knowing a QF exists implies the R16 result),
    # and a played match on/after as_of_date reappears as a fixture, which
    # is exactly the backtesting view.
    if as_of_date is None:
        remaining = [m for m in matches if not m.played]
    else:
        remaining = [m for m in matches if m.date >= as_of_date][:1]
    upcoming = [
        Fixture(m.date, m.stage, m.round_label, m.opponent_of(team_name), m.venue)
        for m in remaining
    ]

    shootouts = [m for m in played if m.decided_by == DECIDED_PENALTIES]
    return TeamForm(
        team=team_name,
        as_of_date=as_of_date,
        played=tuple(played),
        upcoming=tuple(upcoming),
        wins=sum(m.result == "W" for m in played),
        draws=sum(m.result == "D" for m in played),
        losses=sum(m.result == "L" for m in played),
        goals_for=sum(m.goals_for for m in played),
        goals_against=sum(m.goals_against for m in played),
        extra_time_games=sum(m.decided_by != "regulation" for m in played),
        shootout_record=(
            sum(m.result == "W" for m in shootouts),
            sum(m.result == "L" for m in shootouts),
        ),
        status=_status(played, upcoming),
    )


# ----------------------------------------------------------------- squad


def _age(date_of_birth: str, on: Optional[str]) -> int:
    born = _date.fromisoformat(date_of_birth)
    ref = _date.fromisoformat(on) if on else _date.today()
    return ref.year - born.year - ((ref.month, ref.day) < (born.month, born.day))


def squad_stats(
    store: DataStore, team_name: str, as_of_date: Optional[str] = None
) -> tuple[PlayerStats, ...]:
    """Squad with per-player goal tallies from all matches before as_of_date.

    Own goals are credited to no player (they were scored by an opponent).
    goal_events records each goal with its match context — the per-player
    play history this dataset can offer.
    """
    goals: dict[str, int] = {}
    penalty_goals: dict[str, int] = {}
    goal_events: dict[str, list[str]] = {}
    for match in store.get_matches(team=team_name, as_of_date=as_of_date, played_only=True):
        team_goals = match.goals1 if match.team1 == team_name else match.goals2
        stage = _STAGE_LABEL.get(match.stage, match.round_label)
        opponent = match.opponent_of(team_name)
        for goal in team_goals:
            if goal.own_goal or goal.player_id is None:
                continue
            goals[goal.player_id] = goals.get(goal.player_id, 0) + 1
            if goal.penalty:
                penalty_goals[goal.player_id] = penalty_goals.get(goal.player_id, 0) + 1
            pen = " pen" if goal.penalty else ""
            goal_events.setdefault(goal.player_id, []).append(
                f"{goal.minute}'{pen} vs {opponent} ({stage})"
            )

    squad = sorted(
        store.get_squad(team_name),
        key=lambda p: (_POSITION_ORDER.get(p.position, 9), p.number),
    )
    return tuple(
        PlayerStats(
            player=p,
            age=_age(p.date_of_birth, as_of_date),
            goals=goals.get(p.id, 0),
            penalty_goals=penalty_goals.get(p.id, 0),
            goal_events=tuple(goal_events.get(p.id, ())),
        )
        for p in squad
    )


# ---------------------------------------------------------------- report


def team_report(store: DataStore, team_text: str, as_of_date: Optional[str] = None) -> TeamReport:
    """Full report from free-text team input ('korea', 'NED', 'Czechia')."""
    team = store.resolve_team(team_text)
    if team is None:
        raise KeyError(f"unknown team: {team_text!r}")
    return TeamReport(
        team=team,
        form=team_form(store, team.name, as_of_date),
        squad=squad_stats(store, team.name, as_of_date),
    )


_STAGE_LABEL = {
    STAGE_QUALIFYING_PLAYOFF: "playoff",
    STAGE_GROUP: "group",
}


def render_team_report(report: TeamReport) -> str:
    """Plain-text rendering, used both by the CLI and as LLM prompt evidence."""
    team, form = report.team, report.form
    lines = [
        f"{team.name} ({team.fifa_code}) — Group {team.group}, {team.confederation}",
        f"as of {form.as_of_date}" if form.as_of_date else "as of latest data",
        "",
        f"Record: {form.wins}W {form.draws}D {form.losses}L · "
        f"goals {form.goals_for}-{form.goals_against} · status: {form.status}",
    ]
    if form.extra_time_games:
        won, lost = form.shootout_record
        lines.append(
            f"Went past 90 minutes {form.extra_time_games}x"
            + (f" · shootouts won {won}, lost {lost}" if won or lost else "")
        )

    lines += ["", "Matches:"]
    for m in form.played:
        stage = _STAGE_LABEL.get(m.stage, m.round_label)
        lines.append(f"  {m.date}  {m.result}  {m.score_line:>18}  vs {m.opponent}  ({stage})")
    for f in form.upcoming:
        lines.append(f"  {f.date}  upcoming {f.round_label} vs {f.opponent} at {f.venue}")

    scorers = sorted(
        (s for s in report.squad if s.goals), key=lambda s: s.goals, reverse=True
    )
    if scorers:
        lines += ["", "Tournament scorers:"]
        for s in scorers:
            pens = f" ({s.penalty_goals} pen)" if s.penalty_goals else ""
            lines.append(f"  {s.goals}{pens}  {s.player.name} ({s.player.position})")

    lines += ["", "Squad:"]
    for s in report.squad:
        club = f"{s.player.club.name} ({s.player.club.country})"
        goals = f" · scored: {', '.join(s.goal_events)}" if s.goal_events else ""
        lines.append(
            f"  {s.player.position} #{s.player.number:>2} {s.player.name}, {s.age} — {club}{goals}"
        )
    return "\n".join(lines)
