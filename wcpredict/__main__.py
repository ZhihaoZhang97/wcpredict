"""Command-line entry point.

    python3 -m wcpredict team <name> [--as-of YYYY-MM-DD]
    python3 -m wcpredict h2h <team1> <team2> [--as-of YYYY-MM-DD]
    uv run python -m wcpredict predict <team1> <team2> --stage "round of 16"

Team names are free text — "korea", "NED" and "Czechia" all resolve.
The predict command calls the Claude API (needs credentials) and the
langgraph/langchain-anthropic dependencies (`uv sync`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .datastore import DataStore
from .features import _summarize, render_team_report, team_report

_PROJECT_DIR = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = _PROJECT_DIR / "data"


def _load_dotenv() -> None:
    # Pick up ANTHROPIC_API_KEY from the project's .env regardless of cwd.
    # Optional: only the predict command needs it, and dotenv may not be
    # installed when running the data-layer commands with plain python3.
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_PROJECT_DIR / ".env")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wcpredict")
    parser.add_argument("--data-dir", default=_DEFAULT_DATA_DIR, help="directory with the JSON files")
    sub = parser.add_subparsers(dest="command", required=True)

    p_team = sub.add_parser("team", help="2026 history + squad for a team")
    p_team.add_argument("name")
    p_team.add_argument("--as-of", dest="as_of", default=None, metavar="YYYY-MM-DD")

    p_h2h = sub.add_parser("h2h", help="meetings between two teams in this data")
    p_h2h.add_argument("team1")
    p_h2h.add_argument("team2")
    p_h2h.add_argument("--as-of", dest="as_of", default=None, metavar="YYYY-MM-DD")

    p_pred = sub.add_parser("predict", help="AI prediction for a fixture (calls the Claude API)")
    p_pred.add_argument("team1")
    p_pred.add_argument("team2")
    p_pred.add_argument("--stage", required=True, help="e.g. 'group', 'round of 16', '1/4 final', 'final'")
    p_pred.add_argument("--as-of", dest="as_of", default=None, metavar="YYYY-MM-DD")
    p_pred.add_argument(
        "--trace",
        action="store_true",
        help="print each graph node as it finishes and write a full trace file to traces/",
    )
    p_pred.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="cap on simultaneous research API calls (default: 6)",
    )

    args = parser.parse_args(argv)
    store = DataStore(args.data_dir)

    if args.command == "predict":
        _load_dotenv()
        return _predict(store, args)

    if args.command == "team":
        try:
            report = team_report(store, args.name, as_of_date=args.as_of)
        except KeyError:
            print(f"could not resolve team {args.name!r}", file=sys.stderr)
            return 1
        print(render_team_report(report))
        return 0

    # h2h
    teams = []
    for text in (args.team1, args.team2):
        team = store.resolve_team(text)
        if team is None:
            print(f"could not resolve team {text!r}", file=sys.stderr)
            return 1
        teams.append(team.name)
    meetings = store.head_to_head(teams[0], teams[1], as_of_date=args.as_of)
    if not meetings:
        print(f"no meetings between {teams[0]} and {teams[1]} in this data")
        return 0
    for m in meetings:
        if m.played:
            s = _summarize(m, teams[0])
            print(f"{m.date}  {m.round_label}: {teams[0]} {s.score_line} {teams[1]} ({s.result} for {teams[0]})")
        else:
            print(f"{m.date}  {m.round_label}: {teams[0]} vs {teams[1]} — not yet played")
    return 0


def _predict(store: DataStore, args) -> int:
    import os

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(
            "no Anthropic credentials found — copy .env.example to .env and "
            "set ANTHROPIC_API_KEY (or export it in your shell)",
            file=sys.stderr,
        )
        return 1
    if not os.environ.get("TAVILY_API_KEY"):
        print(
            "TAVILY_API_KEY not set — player research uses Tavily web search; "
            "get a free key at tavily.com and add it to .env",
            file=sys.stderr,
        )
        return 1

    # Lazy import: team/h2h/check must keep working without the agent deps.
    try:
        from .agent import parse_stage, run_prediction
    except ImportError as exc:
        print(f"prediction dependencies missing ({exc}); run: uv sync", file=sys.stderr)
        return 1

    try:
        parse_stage(args.stage)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"gathering data, researching players and predicting {args.team1} vs {args.team2}...")
    on_step = _make_step_tracer() if args.trace else None
    extra = {}
    if args.max_concurrency:
        extra["max_concurrency"] = args.max_concurrency
    state = run_prediction(
        store, args.team1, args.team2, args.stage,
        as_of_date=args.as_of, on_step=on_step, **extra,
    )
    if args.trace:
        print(f"trace written to {_write_trace(args, state)}")
    p = state["prediction"]

    outcome = {
        "team1_win": f"{p.team1} win",
        "team2_win": f"{p.team2} win",
        "draw": "draw",
    }[p.outcome]
    print()
    print(f"{p.team1} vs {p.team2} — {p.stage.replace('_', ' ')}")
    print(f"prediction: {outcome}, {p.predicted_score} after 90'"
          + (f" ({p.decided_by.replace('_', ' ')})" if p.decided_by else ""))
    print(f"expected goals: {p.team1} {p.expected_goals_team1:.1f} · "
          f"{p.team2} {p.expected_goals_team2:.1f}")
    print(f"probabilities: {p.team1} {p.prob_team1_win:.0%} · draw {p.prob_draw:.0%} · "
          f"{p.team2} {p.prob_team2_win:.0%}")
    print("\nkey factors:")
    for factor in p.key_factors:
        print(f"  - {factor}")
    print(f"\nreasoning: {p.reasoning}")
    return 0


def _make_step_tracer():
    """Progress printer for --trace: node name, elapsed time, output sizes."""
    import time

    started = time.monotonic()
    last = [started]

    def on_step(node_name: str, delta: dict) -> None:
        now = time.monotonic()
        fields = ", ".join(
            f"{key}={len(str(value))} chars" for key, value in delta.items()
        )
        print(f"  [{now - started:6.1f}s] {node_name} done (+{now - last[0]:.1f}s) -> {fields}")
        last[0] = now

    return on_step


def _write_trace(args, state: dict) -> Path:
    """Dump the full pipeline state to traces/ as a readable markdown file."""
    from datetime import datetime

    traces_dir = _PROJECT_DIR / "traces"
    traces_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = traces_dir / f"{stamp}-{state.get('team1', args.team1)}-vs-{state.get('team2', args.team2)}.md"

    prediction = state.get("prediction")
    sections = [
        f"# Trace: {state.get('team1')} vs {state.get('team2')} ({state.get('stage')})",
        f"run at {stamp} · as_of={state.get('as_of_date')} · "
        f"inputs: {args.team1!r} / {args.team2!r} / stage={args.stage!r}",
        "\n## Node: gather_data — team1 report\n" + state.get("team1_report", "(missing)"),
        "\n## Node: gather_data — team2 report\n" + state.get("team2_report", "(missing)"),
        "\n## Node: gather_data — head-to-head\n" + state.get("head_to_head", "(missing)"),
        "\n## Node: run_searches — " + state.get("search_summary", "(missing)") + "\n"
        + "\n".join(
            f"- {t['team']}: {len(t['material'])} chars of raw snippets"
            for t in state.get("condense_tasks", [])
        ),
        "\n## Node: condense — scout briefings (parallel)\n"
        + "\n\n".join(state.get("research_notes", ["(missing)"])),
        "\n## Node: predict — structured output\n"
        + (prediction.model_dump_json(indent=2) if prediction is not None else "(missing)"),
    ]
    path.write_text("\n".join(sections), encoding="utf-8")
    return path.relative_to(_PROJECT_DIR)


if __name__ == "__main__":
    sys.exit(main())
