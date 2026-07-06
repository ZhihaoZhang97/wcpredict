"""Sanity checks for the data layer, run against the real data files.

Usage:  python3 -m wcpredict.check [data_dir]

Validates the loader, normalizer and resolvers against everything we know
about the source data (48 teams, 1248 players, 104 tournament matches,
five raw score shapes) and reports scorer-resolution coverage.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_config
from .datastore import DataStore
from .features import squad_stats, team_form, team_report
from .models import DECIDED_PENALTIES, STAGE_GROUP, STAGE_QUALIFYING_PLAYOFF

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok " if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        _FAILURES.append(label)


def main() -> int:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "data"
    store = DataStore(data_dir)

    print("config")
    cfg = load_config()
    expected_keys = {
        "llm": {"model", "request_timeout_s", "max_retries", "condense_max_tokens",
                "predict_max_tokens", "predict_effort"},
        "search": {"workers", "results_per_player", "results_team_news"},
        "graph": {"max_concurrency"},
        "sync": {"upstream", "timeout_s", "files"},
    }
    for name, keys in expected_keys.items():
        missing = keys - set(cfg.get(name) or {})
        check(f"config.yaml has {name} section", not missing, f"missing {sorted(missing)}")
    check(
        "sync.files covers the files the datastore reads",
        {"worldcup.json", "worldcup.squads.json", "worldcup.teams.json",
         "worldcup.quali_playoffs.json"} <= set(cfg["sync"]["files"]),
    )
    try:
        # The stdlib loader only speaks a YAML subset; when PyYAML is
        # around (it comes with the agent deps), confirm both read the
        # file identically so the subset never silently drifts.
        import yaml
    except ImportError:
        pass
    else:
        full = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        check("config.yaml parses identically under PyYAML", cfg == full)

    print("registry")
    check("48 teams loaded", len(store.teams) == 48, f"got {len(store.teams)}")
    squad_sizes = [len(store.get_squad(t.name)) for t in store.teams]
    check(
        "1248 players in 26-man squads",
        sum(squad_sizes) == 1248 and set(squad_sizes) == {26},
        f"total {sum(squad_sizes)}, sizes {sorted(set(squad_sizes))}",
    )

    print("team resolution")
    for text, expected in [
        ("USA", "USA"),
        ("United States", "USA"),
        ("Korea", "South Korea"),
        ("Korea Republic", "South Korea"),
        ("Czechia", "Czech Republic"),
        ("bosnia", "Bosnia & Herzegovina"),
        ("curacao", "Curaçao"),
        ("Cabo Verde", "Cape Verde"),
        ("NED", "Netherlands"),
        ("cote d'ivoire", "Ivory Coast"),
    ]:
        team = store.resolve_team(text)
        check(f"resolve {text!r}", team is not None and team.name == expected,
              f"got {team.name if team else None}")
    check("nonsense does not resolve", store.resolve_team("atlantis") is None)

    print("matches / normalizer")
    tournament = store.get_matches(source="tournament")
    played = [m for m in tournament if m.played]
    check("104 tournament matches", len(tournament) == 104, f"got {len(tournament)}")
    # The tournament progresses as data is re-synced: 91 matches were
    # played when this harness was written, and results only accumulate.
    check("at least 91 played", len(played) >= 91, f"got {len(played)}")
    knockout_played = [m for m in played if m.stage != STAGE_GROUP]
    check(
        "every played knockout match has a winner",
        all(m.winner is not None for m in knockout_played),
        f"{len(knockout_played)} matches",
    )
    check(
        "group draws have no winner but a regulation result",
        all(m.decided_by == "regulation" for m in played if m.is_draw),
    )
    pens = [m for m in played if m.decided_by == DECIDED_PENALTIES]
    check(
        "at least 3 matches decided on penalties (R32 had 3)",
        len(pens) >= 3,
        ", ".join(f"{m.winner} beat {m.opponent_of(m.winner)}" for m in pens),
    )
    germany = next(m for m in pens if m.involves("Germany"))
    check(
        "Germany–Paraguay shootout normalized correctly",
        germany.winner == "Paraguay"
        and germany.score.ft_90 == (1, 1)
        and germany.score.final == germany.score.et
        and germany.score.pens == (3, 4),
        f"ft {germany.score.ft_90}, et {germany.score.et}, pens {germany.score.pens}",
    )
    check(
        "goal tallies match final scores",
        all(
            (len(m.goals1), len(m.goals2)) == m.score.final
            for m in played
            if m.score.final is not None
        ),
    )

    playoffs = store.get_matches(source=STAGE_QUALIFYING_PLAYOFF)
    check(
        "all playoff matches played with a winner",
        len(playoffs) > 0 and all(m.played and m.winner for m in playoffs),
        f"{len(playoffs)} matches",
    )
    check(
        "no fuzzy misassignment of playoff teams (Ireland is not Korea)",
        not any(m.involves("South Korea") for m in playoffs)
        and any(m.involves("Republic of Ireland") for m in playoffs),
    )

    print("as_of_date filtering")
    before = store.get_matches(team="Germany", as_of_date="2026-06-29", played_only=True)
    check(
        "Germany before R32 (2026-06-29) = 3 group matches",
        len(before) == 3 and all(m.stage == STAGE_GROUP for m in before),
        f"got {len(before)}",
    )

    print("scorer resolution")
    goals = [g for m in played for g in (*m.goals1, *m.goals2)]
    resolved = [g for g in goals if g.player_id is not None]
    unresolved = Counter(g.scorer_raw for g in goals if g.player_id is None)
    rate = len(resolved) / len(goals) if goals else 0.0
    print(f"  tournament goals: {len(goals)}, resolved {len(resolved)} ({rate:.0%})")
    if unresolved:
        print("  unresolved scorers:", dict(unresolved))
    check("scorer resolution >= 95%", rate >= 0.95, f"{rate:.1%}")
    check(
        "resolved ids point at real players",
        all(store.get_player(g.player_id) for g in resolved),
    )

    print("features")
    germany = team_form(store, "Germany")
    check(
        "Germany: 4 tournament games, out on penalties to Paraguay",
        germany.wins + germany.draws + germany.losses == 4
        and germany.shootout_record == (0, 1)
        and "Paraguay" in germany.status,
        germany.status,
    )
    mexico = team_form(store, "Mexico", as_of_date="2026-07-04")
    check(
        "Mexico as of 2026-07-04: R16 vs England upcoming",
        any(f.opponent == "England" for f in mexico.upcoming),
        f"upcoming: {[f.opponent for f in mexico.upcoming]}",
    )
    check(
        "as_of hides later matches from form",
        len(team_form(store, "Germany", as_of_date="2026-06-12").played) <= 1,
    )
    czech = squad_stats(store, "Czech Republic")
    check(
        "squad stats cover the full 26 and credit scorers",
        len(czech) == 26 and sum(s.goals for s in czech) > 0,
        f"{sum(s.goals for s in czech)} goals credited",
    )
    report = team_report(store, "korea")  # free-text entry point
    check(
        "team_report resolves free text ('korea')",
        report.team.name == "South Korea" and len(report.squad) == 26,
    )

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED: {_FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
