"""Load config.yaml, the project's central settings file.

Stdlib only, like sync.py: the sync script and the data-layer commands
must run with plain python3 (no installed dependencies), so instead of
requiring PyYAML this module parses the small YAML subset config.yaml
declares at its top — nested two-space-indented mappings, scalar
values, lists of scalars, and comments. Anything outside that subset
raises ConfigError rather than being guessed at.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class ConfigError(ValueError):
    """config.yaml is missing or uses syntax outside the supported subset."""


def _scalar(text: str) -> Any:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    if text in ("null", "~", ""):
        return None
    if text in ("true", "false"):
        return text == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _lines(raw: str) -> Iterator[tuple[int, int, str]]:
    """Yield (line number, indent, content) for significant lines."""
    for lineno, line in enumerate(raw.splitlines(), 1):
        # Comments: full-line, or trailing after " #". The subset has no
        # quoted strings containing " #", so a plain split is safe.
        content = line.split(" #", 1)[0].rstrip()
        if not content or content.lstrip().startswith("#"):
            continue
        stripped = content.lstrip(" ")
        indent = len(content) - len(stripped)
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise ConfigError(f"config.yaml line {lineno}: tabs are not allowed")
        if indent % 2:
            raise ConfigError(
                f"config.yaml line {lineno}: indentation must be a multiple of two spaces"
            )
        yield lineno, indent, stripped


def _parse_block(lines: list[tuple[int, int, str]], pos: int, indent: int) -> tuple[Any, int]:
    """Parse one mapping or list starting at lines[pos], all at `indent`."""
    if lines[pos][2].startswith("- "):
        items: list[Any] = []
        while pos < len(lines) and lines[pos][1] == indent and lines[pos][2].startswith("- "):
            items.append(_scalar(lines[pos][2][2:].strip()))
            pos += 1
        if pos < len(lines) and lines[pos][1] > indent:
            raise ConfigError(f"config.yaml line {lines[pos][0]}: nested lists are not supported")
        return items, pos

    mapping: dict[str, Any] = {}
    while pos < len(lines) and lines[pos][1] == indent:
        lineno, _, content = lines[pos]
        key, sep, rest = content.partition(":")
        key, rest = key.strip(), rest.strip()
        if not sep or not key or content.startswith("- "):
            raise ConfigError(f"config.yaml line {lineno}: expected 'key: value' or 'key:'")
        if key in mapping:
            raise ConfigError(f"config.yaml line {lineno}: duplicate key {key!r}")
        pos += 1
        if rest:
            mapping[key] = _scalar(rest)
        elif pos < len(lines) and lines[pos][1] > indent:
            mapping[key], pos = _parse_block(lines, pos, lines[pos][1])
        else:
            mapping[key] = None
    if pos < len(lines) and lines[pos][1] > indent:
        raise ConfigError(f"config.yaml line {lines[pos][0]}: unexpected indent")
    return mapping, pos


def parse_simple_yaml(raw: str) -> dict[str, Any]:
    lines = list(_lines(raw))
    if not lines:
        return {}
    if lines[0][1] != 0:
        raise ConfigError(f"config.yaml line {lines[0][0]}: top level must not be indented")
    parsed, pos = _parse_block(lines, 0, 0)
    if pos != len(lines):
        raise ConfigError(f"config.yaml line {lines[pos][0]}: could not parse")
    if not isinstance(parsed, dict):
        raise ConfigError("config.yaml: top level must be a mapping")
    return parsed


@lru_cache(maxsize=None)
def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Parse the config file once per path; treat the result as read-only."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    return parse_simple_yaml(raw)


def section(name: str) -> dict[str, Any]:
    """One top-level section of config.yaml (e.g. 'llm', 'sync')."""
    config = load_config()
    try:
        return config[name]
    except KeyError:
        raise ConfigError(f"config.yaml has no {name!r} section") from None
