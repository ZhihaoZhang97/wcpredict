"""Load the raw JSON files once and expose canonical, indexed data.

DataStore is the single entry point the rest of the pipeline (features,
agent tools) talks to. It owns the team registry, the player registry and
the normalized match list; nothing outside this package should read the
JSON files directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import Club, Goal, Match, Player, Team
from .normalizer import normalize_score, parse_stage
from .resolver import TeamResolver, match_scorer, slugify

_TOURNAMENT_FILE = "worldcup.json"
_SQUADS_FILE = "worldcup.squads.json"
_TEAMS_FILE = "worldcup.teams.json"
_PLAYOFFS_FILE = "worldcup.quali_playoffs.json"


class DataStore:
    def __init__(self, data_dir: str | Path):
        self._data_dir = Path(data_dir)

        self._teams = self._load_teams()
        self._teams_by_name = {t.name: t for t in self._teams}
        self._resolver = TeamResolver(self._teams)

        self._squads, self._players_by_id = self._load_squads()
        self._matches = self._load_matches(_TOURNAMENT_FILE, source="tournament")
        self._matches += self._load_matches(_PLAYOFFS_FILE, source="qualifying_playoff")
        self._matches.sort(key=lambda m: m.date)

    # ------------------------------------------------------------- loading

    def _read(self, filename: str):
        with open(self._data_dir / filename, encoding="utf-8") as f:
            return json.load(f)

    def _load_teams(self) -> list[Team]:
        return [
            Team(
                name=raw["name"],
                fifa_code=raw["fifa_code"],
                group=raw["group"],
                confederation=raw["confed"],
                continent=raw["continent"],
                alt_name=raw.get("name_normalised"),
            )
            for raw in self._read(_TEAMS_FILE)
        ]

    def _load_squads(self) -> tuple[dict[str, list[Player]], dict[str, Player]]:
        squads: dict[str, list[Player]] = {}
        by_id: dict[str, Player] = {}
        for raw_team in self._read(_SQUADS_FILE):
            team_name = self._resolver.resolve(raw_team["name"], fuzzy=False) or raw_team["name"]
            code = raw_team["fifa_code"]
            players = []
            for raw in raw_team["players"]:
                player = Player(
                    id=f"{code}:{slugify(raw['name'])}:{raw['date_of_birth']}",
                    name=raw["name"],
                    team_code=code,
                    number=raw["number"],
                    position=raw["pos"],
                    date_of_birth=raw["date_of_birth"],
                    club=Club(name=raw["club"]["name"], country=raw["club"]["country"]),
                )
                players.append(player)
                by_id[player.id] = player
            squads[team_name] = players
        return squads, by_id

    def _load_matches(self, filename: str, source: str) -> list[Match]:
        matches = []
        for raw in self._read(filename)["matches"]:
            # Source files use canonical spellings, so resolve exactly —
            # fuzzy matching would misassign playoff teams that did not
            # qualify (and bracket placeholders like "W93"); those keep
            # their raw name.
            team1 = self._resolver.resolve(raw["team1"], fuzzy=False) or raw["team1"]
            team2 = self._resolver.resolve(raw["team2"], fuzzy=False) or raw["team2"]
            score, winner_index, decided_by = normalize_score(raw.get("score"))
            matches.append(
                Match(
                    source=source,
                    stage=parse_stage(raw["round"], raw.get("group"), source),
                    round_label=raw["round"],
                    date=raw["date"],
                    team1=team1,
                    team2=team2,
                    score=score,
                    winner=(team1, team2)[winner_index] if winner_index is not None else None,
                    decided_by=decided_by,
                    goals1=self._resolve_goals(raw.get("goals1"), team1, team2),
                    goals2=self._resolve_goals(raw.get("goals2"), team2, team1),
                    group=raw.get("group"),
                    venue=raw.get("ground"),
                    time=raw.get("time"),
                    num=raw.get("num"),
                )
            )
        return matches

    def _resolve_goals(self, raw_goals, team: str, opponent: str) -> tuple[Goal, ...]:
        goals = []
        for raw in raw_goals or ():
            own_goal = bool(raw.get("owngoal"))
            # An own goal is credited to the side it counts for but was
            # scored by an opposition player — search the other squad.
            squad = self._squads.get(opponent if own_goal else team, ())
            player = match_scorer(raw["name"], squad)
            goals.append(
                Goal(
                    scorer_raw=raw["name"],
                    minute=raw["minute"],
                    player_id=player.id if player else None,
                    penalty=bool(raw.get("penalty")),
                    own_goal=own_goal,
                )
            )
        return tuple(goals)

    # ----------------------------------------------------------------- API

    @property
    def teams(self) -> list[Team]:
        return list(self._teams)

    def resolve_team(self, text: str) -> Optional[Team]:
        """Resolve free-text input ('Korea', 'USA', 'Czechia') to a Team."""
        name = self._resolver.resolve(text)
        return self._teams_by_name.get(name) if name else None

    def get_team(self, name: str) -> Team:
        return self._teams_by_name[name]

    def get_squad(self, team_name: str) -> list[Player]:
        """26-man squad for a canonical team name."""
        return list(self._squads[team_name])

    def get_player(self, player_id: str) -> Player:
        return self._players_by_id[player_id]

    def get_matches(
        self,
        team: Optional[str] = None,
        as_of_date: Optional[str] = None,
        source: Optional[str] = None,
        played_only: bool = False,
    ) -> list[Match]:
        """Matches filtered by team, source and date.

        as_of_date (ISO) keeps only matches strictly before that date, so
        features computed for a prediction never see the match being
        predicted — which also makes backtesting against played matches
        straightforward.
        """
        matches = self._matches
        if team is not None:
            matches = [m for m in matches if m.involves(team)]
        if source is not None:
            matches = [m for m in matches if m.source == source]
        if as_of_date is not None:
            matches = [m for m in matches if m.date < as_of_date]
        if played_only:
            matches = [m for m in matches if m.played]
        return list(matches)

    def head_to_head(
        self, team1: str, team2: str, as_of_date: Optional[str] = None
    ) -> list[Match]:
        return [
            m
            for m in self.get_matches(team=team1, as_of_date=as_of_date)
            if m.involves(team2)
        ]
