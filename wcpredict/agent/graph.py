"""Graph wiring: connect the nodes from nodes.py and expose run_prediction.

Topology (searches are parallel HTTP, condense fans out per team):

    START -> gather_data -> run_searches ==Send==> condense (x2, parallel)
                                                       |
                                                    predict -> END

Node implementations live in nodes.py; state types in state.py; search
providers in search.py; the prediction schema in schema.py. This module
only binds dependencies and draws edges.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

from langgraph.graph import END, START, StateGraph

from ..config import section
from ..datastore import DataStore
from . import nodes
from .llm import make_llm, resolve_provider
from .search import SearchProvider, TavilyProvider
from .state import PipelineState, parse_stage

# Cap on simultaneous LLM API calls (the search fan-out is plain HTTP and
# governed separately by search.workers) — graph.max_concurrency in
# config.yaml.
DEFAULT_MAX_CONCURRENCY = section("graph")["max_concurrency"]


def build_graph(
    store: DataStore,
    search_provider: Optional[SearchProvider] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
):
    provider = search_provider if search_provider is not None else TavilyProvider()
    spec = resolve_provider(llm_provider)
    llm = make_llm(spec, model=llm_model)
    predict_llm = make_llm(spec, model=llm_model, reasoning=True)

    graph = StateGraph(PipelineState)
    graph.add_node("gather_data", partial(nodes.gather_data, store))
    graph.add_node("run_searches", partial(nodes.run_searches, store, provider))
    graph.add_node("condense", partial(nodes.condense, llm))
    graph.add_node(
        "predict",
        partial(nodes.predict, predict_llm, spec.structured_output_method),
    )

    graph.add_edge(START, "gather_data")
    graph.add_edge("gather_data", "run_searches")
    # Fan out: one condense per team, run concurrently; predict waits for
    # both to finish (superstep barrier).
    graph.add_conditional_edges("run_searches", nodes.fan_out_condense, ["condense"])
    graph.add_edge("condense", "predict")
    graph.add_edge("predict", END)
    return graph.compile()


def run_prediction(
    store: DataStore,
    team1: str,
    team2: str,
    stage_text: str,
    as_of_date: Optional[str] = None,
    on_step=None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    search_provider: Optional[SearchProvider] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> PipelineState:
    """Convenience entry point: free-text teams and stage in, final state out.

    on_step, if given, is called as on_step(node_name, state_delta) after
    each node finishes — the hook the CLI's --trace flag uses. Without it
    the graph runs as a single opaque invoke.
    """
    app = build_graph(
        store,
        search_provider=search_provider,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    config = {"max_concurrency": max_concurrency}
    inputs: PipelineState = {
        "team1_text": team1,
        "team2_text": team2,
        "stage": parse_stage(stage_text),
        "as_of_date": as_of_date,
    }
    if on_step is None:
        return app.invoke(inputs, config=config)

    state = dict(inputs)
    for update in app.stream(inputs, config=config, stream_mode="updates"):
        for node_name, delta in update.items():
            for key, value in delta.items():
                # research_notes uses an operator.add reducer: the parallel
                # condense deltas must accumulate, not overwrite.
                if key == "research_notes":
                    state.setdefault(key, []).extend(value)
                else:
                    state[key] = value
            on_step(node_name, delta)
    return state
