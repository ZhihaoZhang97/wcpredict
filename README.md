# wcpredict — World Cup 2026 match prediction agent

An AI agent that predicts FIFA World Cup 2026 matches. Type in two teams
and a stage; it gathers each team's tournament history from local data,
researches all 52 squad players with live web search, and produces a
structured prediction — winner, scoreline, expected goals, and outcome
probabilities — with an analyst's reasoning.

Built with [LangGraph](https://langchain-ai.github.io/langgraph/) on
match data from
[openfootball/worldcup.json](https://github.com/openfootball/worldcup.json).
Runs on [Claude](https://www.anthropic.com/claude) by default, or any of
OpenAI, Gemini, Qwen, GLM, MiniMax and DeepSeek via `--provider`.

```
$ uv run python -m wcpredict predict mexico england --stage "round of 16"

Mexico vs England — round of 16
prediction: England win, 1:2 after 90' (regulation)
expected goals: Mexico 1.2 · England 1.5
probabilities: Mexico 31% · draw 28% · England 41%
...
```

## How it works

```
START → gather_data → run_searches ══Send══> condense (×2, parallel)
                                                  │
                                               predict → END
```

| Node | What it does |
|---|---|
| `gather_data` | No LLM. Resolves free-text team names ("korea", "NED", "Czechia"), renders each team's 2026 report — match history with how each game was decided, full squad with per-player goal events — plus head-to-head. |
| `run_searches` | No LLM. Fires one web search per player (all 26 per team) plus team news, ~54 queries in parallel via [Tavily](https://tavily.com). |
| `condense` | One LLM call per team (parallel): raw snippets → scout briefing. |
| `predict` | LLM with reasoning enabled (adaptive thinking at `xhigh` effort on Claude) and structured output: expected goals first, then the most likely score conditional on the predicted path. |

The data layer is independent of the agent: a canonical match model
normalizes the five raw score shapes in the source files (regulation,
extra time, penalties, playoff shorthand, unplayed), resolves goalscorer
names against squads (100% resolution), and supports `--as-of` for
leak-free backtesting against played matches.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+.

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY and TAVILY_API_KEY
```

- `ANTHROPIC_API_KEY` — https://console.anthropic.com
- `TAVILY_API_KEY` — free tier at https://tavily.com (~18 predictions/month)

### Choosing an LLM

`predict` uses Claude by default. Pick another provider with
`--provider` (or set `WCPREDICT_LLM_PROVIDER` in `.env`) and put that
provider's API key in `.env` — see `.env.example` for the variable
names. `--model` (or `WCPREDICT_LLM_MODEL`) overrides the provider's
default model.

| Provider | Default model | Key variable |
|---|---|---|
| `anthropic` | `claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-5.1` | `OPENAI_API_KEY` |
| `gemini` | `gemini-2.5-pro` | `GOOGLE_API_KEY` |
| `qwen` | `qwen3-max` | `DASHSCOPE_API_KEY` |
| `glm` | `glm-4.6` | `ZHIPUAI_API_KEY` |
| `minimax` | `MiniMax-M2` | `MINIMAX_API_KEY` |
| `deepseek` | `deepseek-chat` (`deepseek-reasoner` for the predict step) | `DEEPSEEK_API_KEY` |

```bash
uv run python -m wcpredict predict portugal spain --stage semi --provider deepseek
uv run python -m wcpredict predict portugal spain --stage semi --provider openai --model gpt-5.2
```

## Usage

```bash
# Predict a fixture (calls both APIs)
uv run python -m wcpredict predict portugal spain --stage "round of 16"

# Stage is free text: "group", "1/4 final", "semi", "final" all work.
# --trace prints per-node timing and writes a full trace to traces/
uv run python -m wcpredict predict france morocco --stage "1/4 final" --trace

# Backtest view: only data from before the given date is used
uv run python -m wcpredict predict mexico england --stage "round of 16" --as-of 2026-07-05

# Local data commands (no API keys needed)
python3 -m wcpredict team germany
python3 -m wcpredict h2h portugal spain
```

## Data

The `data/` folder carries the 2026 tournament files from
[openfootball/worldcup.json](https://github.com/openfootball/worldcup.json)
(public domain, CC0), and stays current three ways:

- **every `predict` run syncs first** (skip with `--no-sync`; a failed
  sync degrades to a warning and uses local data),
- a GitHub Action (`.github/workflows/sync-data.yml`) syncs every six
  hours and commits changes,
- manually: `python3 scripts/sync_data.py`, then
  `uv run python -m wcpredict.check` to validate.

If the requested fixture already has a result in the data, `predict`
reports the actual score instead of "predicting" a match whose answer
is in its own input — rerun with `--as-of <match date>` to backtest
the agent honestly against it.

## Development

`uv run python -m wcpredict.check` runs the data-layer sanity harness:
team/scorer resolution, score-shape normalization, as-of filtering, and
feature invariants against the real data files.

Runtime settings — the Claude model, request timeouts, token caps,
search fan-out sizes and the data-sync source — live in `config.yaml`
rather than in the Python modules; edit it and rerun, no code changes
needed. The check harness validates the file's structure.

Project layout:

```
config.yaml            central settings (model, timeouts, fan-out sizes, sync)
data/                  tournament JSON (synced from openfootball)
scripts/sync_data.py   upstream data sync
wcpredict/
  config.py            config.yaml loader (stdlib-only YAML subset)
  datastore.py         loads + indexes the JSON, canonical match model
  normalizer.py        five raw score shapes → one schema
  resolver.py          fuzzy team input, scorer-to-squad matching
  features.py          team reports, squad stats, as-of filtering
  check.py             sanity harness
  agent/
    graph.py           LangGraph wiring + run_prediction
    llm.py             LLM provider registry (anthropic, openai, ...)
    nodes.py           node implementations + prompts
    search.py          SearchProvider protocol (Tavily default)
    schema.py          MatchPrediction structured output
    state.py           pipeline state + stage parsing
```

## Contributing

`main` is protected: all changes (including the automated data syncs)
land through pull requests, and CI runs the check harness on every PR.
Fork or branch, then open a PR against `main`.

## License

Code: [MIT](LICENSE). Match data: public domain
([CC0](https://github.com/openfootball/worldcup.json/blob/master/LICENSE.md))
courtesy of the [openfootball](https://github.com/openfootball) project.
