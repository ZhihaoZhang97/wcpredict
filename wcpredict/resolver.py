"""Fuzzy name resolution: user input -> canonical team, scorer -> squad player.

All the fuzziness in the pipeline is quarantined here. Everything else
works with canonical team names and stable player ids.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches
from typing import Iterable, Optional

from .models import Player, Team


# Letters that NFKD does not decompose to an ASCII base letter.
_CHAR_MAP = str.maketrans(
    {"ı": "i", "ø": "o", "đ": "d", "ł": "l", "ß": "ss", "æ": "ae", "œ": "oe", "ð": "d", "þ": "th"}
)


def normalize_text(text: str) -> str:
    """Lowercase, strip diacritics and punctuation: 'Curaçao' -> 'curacao'."""
    text = text.lower().translate(_CHAR_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def slugify(text: str) -> str:
    return normalize_text(text).replace(" ", "-")


# Common ways users (and other datasets) refer to teams, beyond the
# name / name_normalised / fifa_code already present in teams.json.
# Keys are canonical names; entries for teams not in the file are ignored.
_EXTRA_TEAM_ALIASES = {
    "USA": ["United States", "United States of America", "US"],
    "South Korea": ["Korea", "Republic of Korea"],
    "Czech Republic": ["Czechia"],
    "Bosnia & Herzegovina": ["Bosnia", "Bosnia and Herzegovina", "BiH"],
    "Ivory Coast": ["Cote d'Ivoire", "Cote dIvoire"],
    "Cape Verde": ["Cabo Verde"],
    "Netherlands": ["Holland"],
    "Turkey": ["Turkiye"],
}


class TeamResolver:
    """Resolve free-text team input to a canonical team name."""

    def __init__(self, teams: Iterable[Team]):
        self._by_alias: dict[str, str] = {}
        for team in teams:
            aliases = {team.name, team.fifa_code, team.alt_name}
            aliases.update(_EXTRA_TEAM_ALIASES.get(team.name, ()))
            for alias in aliases:
                if alias:
                    self._by_alias[normalize_text(alias)] = team.name

    def resolve(self, text: str, fuzzy: bool = True) -> Optional[str]:
        """Resolve to a canonical team name.

        fuzzy=False restricts to exact alias matches — use it for names
        coming from data files, which are already canonically spelled.
        Fuzzy matching is for human input only: it would otherwise map
        unrelated teams onto each other (e.g. the playoff participant
        "Republic of Ireland" onto "Korea Republic").
        """
        key = normalize_text(text)
        if not key:
            return None
        if key in self._by_alias:
            return self._by_alias[key]
        if fuzzy:
            close = get_close_matches(key, self._by_alias.keys(), n=1, cutoff=0.75)
            if close:
                return self._by_alias[close[0]]
        return None


def match_scorer(raw_name: str, squad: Iterable[Player]) -> Optional[Player]:
    """Match a goalscorer name from a match record to a squad player.

    Source names come in several forms: full ("Raúl Jiménez"), surname only
    ("Džeko", "Tonali"), or initial + surname ("D. James"). A match is only
    returned when it is unambiguous within the squad.
    """
    squad = list(squad)
    key = normalize_text(raw_name)
    if not key:
        return None
    tokens = key.split()

    exact = [p for p in squad if normalize_text(p.name) == key]
    if len(exact) == 1:
        return exact[0]

    # Initial + surname: "d james" -> first name starts with "d",
    # remaining tokens close the name.
    if len(tokens) >= 2 and len(tokens[0]) == 1:
        rest = " ".join(tokens[1:])
        candidates = [
            p
            for p in squad
            if normalize_text(p.name).startswith(tokens[0])
            and _ends_with_tokens(normalize_text(p.name), rest)
        ]
        if len(candidates) == 1:
            return candidates[0]

    # All tokens of the raw name appear in the player's name (handles
    # surname-only and partial forms), token-wise to avoid substring
    # false positives like "son" in "jackson".
    candidates = [
        p for p in squad if set(tokens) <= set(normalize_text(p.name).split())
    ]
    if len(candidates) == 1:
        return candidates[0]

    # The reverse: squads sometimes list a short nickname ("Diney") while
    # the match record carries the fuller name ("Diney Borges").
    candidates = [
        p for p in squad if set(normalize_text(p.name).split()) <= set(tokens)
    ]
    if len(candidates) == 1:
        return candidates[0]

    # The player's name ends with the raw name (multi-token surnames
    # like "van dijk").
    candidates = [p for p in squad if _ends_with_tokens(normalize_text(p.name), key)]
    if len(candidates) == 1:
        return candidates[0]

    # Shortened first names: every raw token is a prefix of a distinct
    # name token ("maxi araujo" -> "maximiliano araujo").
    candidates = [p for p in squad if _tokens_are_prefixes(tokens, normalize_text(p.name))]
    if len(candidates) == 1:
        return candidates[0]

    # Last resort for spelling variants ("mousa al tamari" vs
    # "musa al taamari"): fuzzy match against full names, accepted only
    # when exactly one squad name is close.
    by_norm = {normalize_text(p.name): p for p in squad}
    close = get_close_matches(key, by_norm.keys(), n=2, cutoff=0.8)
    if len(close) == 1:
        return by_norm[close[0]]
    return None


def _tokens_are_prefixes(tokens: list[str], name: str) -> bool:
    name_tokens = name.split()
    if len(tokens) < 2:
        return False
    remaining = list(name_tokens)
    for token in tokens:
        match = next((t for t in remaining if t.startswith(token)), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def _ends_with_tokens(name: str, suffix: str) -> bool:
    return name == suffix or name.endswith(" " + suffix)
