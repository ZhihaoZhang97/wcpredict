"""LLM provider registry: one place that knows how to build a chat model.

Every provider is exposed as a LangChain chat model so nodes.py stays
provider-agnostic. Anthropic and Gemini use their native integrations;
Qwen (DashScope), GLM (Zhipu), MiniMax and DeepSeek all speak the OpenAI
wire protocol, so they share ChatOpenAI with a custom base_url.

Selection order: explicit argument (CLI --provider/--model) >
WCPREDICT_LLM_PROVIDER / WCPREDICT_LLM_MODEL env vars > anthropic with
its default model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_PROVIDER = "anthropic"

# Per-attempt cap. Streaming (where the integration supports the flag)
# keeps bytes flowing on long condense/predict calls so a wedged
# connection fails fast instead of hanging for the SDK default.
REQUEST_TIMEOUT_S = 240

# Output cap for the non-reasoning (condense) calls, any provider.
CONDENSE_MAX_TOKENS = 8000

# Reasoning depth for Anthropic's adaptive thinking on the predict call.
# "xhigh" spends more thinking tokens for deeper analysis.
PREDICT_EFFORT = "xhigh"


@dataclass(frozen=True)
class ProviderSpec:
    """Everything provider-specific the rest of the code needs."""

    name: str
    default_model: str
    api_key_env: str
    key_url: str  # where to get a key, for error messages
    # OpenAI-compatible endpoint; None for native integrations.
    base_url: Optional[str] = None
    # Model to use when reasoning=True, if the provider splits chat and
    # reasoning into separate models (DeepSeek). None = same model.
    reasoning_model: Optional[str] = None
    # Extra constructor kwargs applied only when reasoning=True.
    reasoning_kwargs: dict[str, Any] = field(default_factory=dict)
    # Output-token ceiling for the reasoning call. For providers whose
    # thinking tokens count against the output budget (Anthropic) this
    # must be generous or the answer gets truncated by the thinking.
    reasoning_max_tokens: int = 32768
    # with_structured_output method for the predict call; None keeps the
    # integration's default (usually a forced tool call).
    structured_output_method: Optional[str] = None


_SPECS: dict[str, ProviderSpec] = {
    # json_schema (native structured outputs) instead of the default
    # forced tool call — forced tool choice is incompatible with
    # thinking, and the thinking is what stops degenerate predictions.
    "anthropic": ProviderSpec(
        name="anthropic",
        default_model="claude-opus-4-8",
        api_key_env="ANTHROPIC_API_KEY",
        key_url="https://console.anthropic.com",
        reasoning_kwargs={"thinking": {"type": "adaptive"}, "effort": PREDICT_EFFORT},
        reasoning_max_tokens=128000,  # model's output ceiling
        structured_output_method="json_schema",
    ),
    "openai": ProviderSpec(
        name="openai",
        default_model="gpt-5.1",
        api_key_env="OPENAI_API_KEY",
        key_url="https://platform.openai.com",
        reasoning_kwargs={"reasoning_effort": "high"},
        reasoning_max_tokens=128000,
        structured_output_method="json_schema",
    ),
    # Gemini 2.5 Pro thinks by default; no extra reasoning kwargs needed.
    "gemini": ProviderSpec(
        name="gemini",
        default_model="gemini-2.5-pro",
        api_key_env="GOOGLE_API_KEY",
        key_url="https://aistudio.google.com/apikey",
        reasoning_max_tokens=65536,
    ),
    # DashScope's compat mode only honors thinking flags on streamed
    # calls and per-model, so we don't force them here; pick a thinking
    # model via WCPREDICT_LLM_MODEL if you want deeper reasoning.
    "qwen": ProviderSpec(
        name="qwen",
        default_model="qwen3-max",
        api_key_env="DASHSCOPE_API_KEY",
        key_url="https://dashscope.console.aliyun.com",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        structured_output_method="function_calling",
    ),
    "glm": ProviderSpec(
        name="glm",
        default_model="glm-4.6",
        api_key_env="ZHIPUAI_API_KEY",
        key_url="https://open.bigmodel.cn",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        reasoning_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
        reasoning_max_tokens=65536,
        structured_output_method="function_calling",
    ),
    # MiniMax-M2 reasons by default (interleaved thinking).
    "minimax": ProviderSpec(
        name="minimax",
        default_model="MiniMax-M2",
        api_key_env="MINIMAX_API_KEY",
        key_url="https://platform.minimax.io",
        base_url="https://api.minimax.io/v1",
        structured_output_method="function_calling",
    ),
    # json_mode, not function_calling: DeepSeek's thinking mode (the
    # reasoner alias and thinking-enabled models like deepseek-v4-pro)
    # rejects tool calls outright ("Thinking mode does not support this
    # tool_choice") but does support JSON output — see
    # https://api-docs.deepseek.com/guides/reasoning_model. json_mode
    # requires the schema in the prompt; nodes.predict appends it.
    "deepseek": ProviderSpec(
        name="deepseek",
        default_model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        key_url="https://platform.deepseek.com",
        base_url="https://api.deepseek.com/v1",
        reasoning_model="deepseek-reasoner",
        reasoning_max_tokens=65536,
        structured_output_method="json_mode",
    ),
}

PROVIDERS = tuple(sorted(_SPECS))


def resolve_provider(name: Optional[str] = None) -> ProviderSpec:
    name = (name or os.environ.get("WCPREDICT_LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    try:
        return _SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown LLM provider {name!r}; choose from: {', '.join(PROVIDERS)}"
        ) from None


def make_llm(spec: ProviderSpec, model: Optional[str] = None, reasoning: bool = False):
    """Build the chat model for one node.

    reasoning=True is used by the predict node: without thinking the
    model fills the schema by pattern and every close knockout tie
    collapses to the same 1:1-on-penalties archetype. Where a provider
    has no thinking switch we still take its larger output budget.
    """
    model = model or os.environ.get("WCPREDICT_LLM_MODEL")
    if model is None:
        model = spec.reasoning_model if (reasoning and spec.reasoning_model) else spec.default_model
    max_tokens = spec.reasoning_max_tokens if reasoning else CONDENSE_MAX_TOKENS
    extra = dict(spec.reasoning_kwargs) if reasoning else {}

    if spec.name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            max_tokens=max_tokens,
            streaming=True,
            default_request_timeout=REQUEST_TIMEOUT_S,
            max_retries=2,
            **extra,
        )

    if spec.name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            max_output_tokens=max_tokens,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=2,
            **extra,
        )

    # OpenAI itself and every OpenAI-compatible provider.
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {}
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
        # ChatOpenAI only auto-reads OPENAI_API_KEY; compat providers
        # keep their own key in their own env var.
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{spec.api_key_env} not set — get a {spec.name} key at {spec.key_url}"
            )
        kwargs["api_key"] = api_key
    return ChatOpenAI(
        model=model,
        max_tokens=max_tokens,
        streaming=True,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=2,
        **kwargs,
        **extra,
    )
