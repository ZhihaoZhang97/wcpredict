"""LangGraph prediction agent. Import requires langgraph + the langchain
provider integrations.

Layout: schema.py (prediction output model), state.py (graph state +
stage parsing), nodes.py (node implementations), llm.py (provider
registry), graph.py (wiring + run_prediction entry point).
"""

from .graph import build_graph, run_prediction
from .schema import MatchPrediction
from .state import parse_stage

__all__ = ["build_graph", "parse_stage", "run_prediction", "MatchPrediction"]
