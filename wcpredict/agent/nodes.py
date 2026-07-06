"""Node implementations for the prediction graph.

Each node takes its dependencies (DataStore, SearchProvider,
ChatAnthropic) as explicit first arguments; graph.py binds them with
functools.partial when wiring the graph. Nothing in this module knows
about edges or execution order.
"""

from __future__ import annotations

import time

from langchain_anthropic import ChatAnthropic
from langgraph.types import Send

from ..datastore import DataStore
from ..features import render_team_report, squad_stats, team_report
from .schema import MatchPrediction
from .search import SearchProvider, search_many
from .state import CondenseTask, PipelineState

MODEL = "claude-opus-4-8"

# Search-result snippets fetched per query.
RESULTS_PER_PLAYER = 3
RESULTS_TEAM_NEWS = 5


# Per-attempt cap. Streaming keeps bytes flowing on long condense/predict
# calls so a wedged connection fails fast (instead of hanging for the
# SDK's 10-minute default, x3 attempts).
REQUEST_TIMEOUT_S = 240


# Reasoning depth for the predict call. "xhigh" spends more thinking
# tokens for deeper analysis; "high" equals omitting the parameter.
PREDICT_EFFORT = "xhigh"


def make_llm(reasoning: bool = False) -> ChatAnthropic:
    """reasoning=True enables adaptive thinking at PREDICT_EFFORT (used by
    the predict node: without thinking the model fills the schema by
    pattern and every close knockout tie collapses to the same
    1:1-on-penalties archetype). Thinking tokens count against
    max_tokens, hence the higher cap — 128000 is the model's output
    ceiling; larger values are rejected by the API."""
    extra = (
        {"thinking": {"type": "adaptive"}, "effort": PREDICT_EFFORT}
        if reasoning
        else {}
    )
    return ChatAnthropic(
        model=MODEL,
        max_tokens=128000 if reasoning else 8000,
        streaming=True,
        default_request_timeout=REQUEST_TIMEOUT_S,
        max_retries=2,
        **extra,
    )


# ------------------------------------------------------------ gather_data


def gather_data(store: DataStore, state: PipelineState) -> PipelineState:
    """No LLM: resolve both teams, render reports and head-to-head."""
    as_of = state.get("as_of_date")
    reports = []
    names = []
    for text in (state["team1_text"], state["team2_text"]):
        report = team_report(store, text, as_of_date=as_of)
        names.append(report.team.name)
        reports.append(render_team_report(report))

    meetings = store.head_to_head(names[0], names[1], as_of_date=as_of)
    h2h_lines = []
    for m in meetings:
        if m.played:
            score = m.score.final
            idx = 0 if m.team1 == names[0] else 1
            line = (
                f"{m.date} {m.round_label}: {names[0]} {score[idx]}-{score[1 - idx]} "
                f"{names[1]}"
                + (f", {m.winner} won ({m.decided_by})" if m.winner else " (draw)")
            )
        else:
            line = f"{m.date} {m.round_label}: scheduled, not yet played"
        h2h_lines.append(line)

    return {
        "team1": names[0],
        "team2": names[1],
        "team1_report": reports[0],
        "team2_report": reports[1],
        "head_to_head": "\n".join(h2h_lines) or "No meetings in this dataset.",
    }


# ------------------------------------------------------------ run_searches


def run_searches(
    store: DataStore, provider: SearchProvider, state: PipelineState
) -> PipelineState:
    """No LLM: fire one web search per player plus team news, all in parallel.

    Queries are built deterministically from our own data; the results are
    grouped per team into raw material for the condense fan-out.
    """
    as_of = state.get("as_of_date")
    stage = state["stage"].replace("_", " ")
    pairs = [(state["team1"], state["team2"]), (state["team2"], state["team1"])]

    labelled: list[tuple[str, str]] = []  # (team, label per snippet block)
    queries: list[tuple[str, int]] = []
    for team, _opponent in pairs:
        for s in squad_stats(store, team, as_of_date=as_of):
            goals = f", scored {', '.join(s.goal_events)}" if s.goal_events else ""
            label = (
                f"{s.player.name} ({s.player.position}, {s.player.club.name}, "
                f"age {s.age}{goals})"
            )
            labelled.append((team, label))
            queries.append(
                (
                    f"{s.player.name} {team} footballer injury news form "
                    "international career stats",
                    RESULTS_PER_PLAYER,
                )
            )
        labelled.append((team, "TEAM NEWS"))
        queries.append(
            (
                f"{team} national team news injuries suspensions expected "
                f"lineup 2026 World Cup {stage}",
                RESULTS_TEAM_NEWS,
            )
        )

    started = time.monotonic()
    results = search_many(provider, queries)
    elapsed = time.monotonic() - started

    material: dict[str, list[str]] = {}
    for (team, label), snippets in zip(labelled, results):
        material.setdefault(team, []).append(f"--- {label} ---\n{snippets}")

    condense_tasks = [
        CondenseTask(
            team=team,
            opponent=opponent,
            stage=state["stage"],
            material="\n\n".join(material[team]),
        )
        for team, opponent in pairs
    ]
    return {
        "condense_tasks": condense_tasks,
        "search_summary": f"{len(queries)} searches in {elapsed:.1f}s",
    }


def fan_out_condense(state: PipelineState) -> list[Send]:
    """Conditional edge: dispatch one condense instance per team."""
    return [Send("condense", task) for task in state["condense_tasks"]]


# ---------------------------------------------------------------- condense


def condense(llm: ChatAnthropic, task: CondenseTask) -> PipelineState:
    """One LLM call per team: raw search snippets -> scout briefing."""
    fixture = (
        f"the 2026 World Cup {task['stage'].replace('_', ' ')} match "
        f"{task['team']} vs {task['opponent']}"
    )
    prompt = (
        f"You are a football scout preparing for {fixture}. Below are raw "
        f"web search snippets about every {task['team']} squad member "
        "(name, position, club, age and any 2026 World Cup goals are in "
        "each header) plus team news.\n\n"
        "Condense them into a briefing: for likely starters and key "
        "players, a few bullets each on injury/suspension status, recent "
        "form and international career; a single line for clear backups; "
        "then a team-news section (injuries, suspensions, expected "
        "lineup). Only use what the snippets support — flag anything "
        "missing or contradictory rather than guessing.\n\n"
        f"{task['material']}"
    )
    response = llm.invoke(prompt)
    if isinstance(response.content, str):
        text = response.content
    else:
        text = "\n".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return {"research_notes": [f"### {task['team']} — scout briefing\n{text}"]}


# --------------------------------------------------------------- predict


def predict(llm: ChatAnthropic, state: PipelineState) -> PipelineState:
    """Weigh reports + merged research and emit a structured MatchPrediction.

    method="json_schema" (Claude's native structured outputs) instead of
    the default forced tool call — forced tool choice is incompatible
    with thinking, and the thinking is what stops degenerate outputs.
    """
    predictor = llm.with_structured_output(MatchPrediction, method="json_schema")
    stage = state["stage"]
    if stage == "group":
        stage_rules = (
            "This is a GROUP match: a draw is a valid final outcome, and "
            "decided_by must be null."
        )
    else:
        stage_rules = (
            "This is a KNOCKOUT match: there must be a winner. If you "
            "predict a level score after 90 minutes, set decided_by to "
            "extra_time or penalties and pick the winner; otherwise set "
            "decided_by to regulation. prob_draw is the probability the "
            "game is level after 90 minutes (before extra time)."
        )
    prompt = (
        "You are a football analyst. Predict this 2026 World Cup match "
        f"({stage.replace('_', ' ')}): {state['team1']} vs {state['team2']}.\n"
        f"{stage_rules}\n"
        "Probabilities must sum to approximately 1. Weigh tournament "
        "form, how teams won (a side that needed penalties twice is "
        "fragile), squad quality, and the scout briefing.\n"
        "Ground every number in the specific evidence below — defensive "
        "records, margins, opposition quality. Do NOT default to a "
        "generic near-even split: if the evidence favors one side, the "
        "probabilities must show it.\n"
        "For the scoreline, work in three steps.\n"
        "1) Estimate each side's expected goals from the concrete "
        "evidence: goals scored/conceded per match so far, quality of "
        "opposition faced, attacking personnel in form, and whether the "
        "matchup profiles as cagey or open (two strong attacks vs "
        "stretched defenses = open).\n"
        "2) Pick the most likely PATH through the match: is one side "
        "more likely to be ahead after 90 minutes than the game is to be "
        "level? If yes, decided_by is regulation. Only when level-after-"
        "90 is the single most likely 90-minute state should you choose "
        "extra_time or penalties.\n"
        "3) Predict the most likely score CONDITIONAL on that path — the "
        "score must tell the same story as your probabilities. A "
        "regulation win needs the winner ahead (asymmetric expected "
        "goals give asymmetric scores like 1:2 or 2:1, never a hedged "
        "1:1); high expected goals need a high-scoring line. Have the "
        "courage of your estimates.\n\n"
        f"=== {state['team1']} — 2026 tournament data ===\n"
        f"{state['team1_report']}\n\n"
        f"=== {state['team2']} — 2026 tournament data ===\n"
        f"{state['team2_report']}\n\n"
        f"=== Head-to-head in this dataset ===\n{state['head_to_head']}\n\n"
        "=== Scout briefing (web research) ===\n"
        + "\n\n".join(state["research_notes"])
    )
    prediction = predictor.invoke(prompt)
    return {"prediction": prediction}
