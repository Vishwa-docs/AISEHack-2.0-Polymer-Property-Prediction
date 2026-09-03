"""Conservative primitive-repeat normalisation for a validated linear PSMILES subset.

Supported form: exactly two terminal ``*`` endpoints and an unbranched sequence of ordinary
atom symbols, e.g. ``*CCO*``, ``*OCC*`` and ``*CCOCCO*``.  The implementation intentionally
returns an explicit unsupported result for richer PSMILES rather than guessing chemistry.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

_ATOM = re.compile(r"Cl|Br|Si|[BCNOSPFI]")


@dataclass(frozen=True)
class Normalization:
    status: str
    normalized: str | None
    primitive_tokens: tuple[str, ...] = ()
    repeat_count: int | None = None
    reason: str | None = None


def _tokens(core: str) -> tuple[str, ...] | None:
    """Return atom tokens only when *all* characters form a simple linear chain."""
    parts = tuple(_ATOM.findall(core))
    return parts if parts and "".join(parts) == core else None


def _smallest_period(tokens: tuple[str, ...]) -> tuple[tuple[str, ...], int]:
    for width in range(1, len(tokens) + 1):
        if len(tokens) % width == 0 and tokens == tokens[:width] * (len(tokens) // width):
            return tokens[:width], len(tokens) // width
    return tokens, 1


def _rotation_minimum(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Choose a deterministic cut point; token-level rotation preserves atom boundaries."""
    return min(tokens[i:] + tokens[:i] for i in range(len(tokens)))


def normalize_linear_repeat(psmiles: str) -> Normalization:
    text = str(psmiles).strip().replace("[*]", "*")
    if text.count("*") != 2:
        return Normalization("unsupported", None, reason="requires exactly two star endpoints")
    if not (text.startswith("*") and text.endswith("*")):
        return Normalization("unsupported", None, reason="requires terminal star endpoints")
    tokens = _tokens(text[1:-1])
    if tokens is None:
        return Normalization("unsupported", None,
                             reason="validated strict check supports unbranched atom sequences only")
    primitive, repeats = _smallest_period(tokens)
    canonical_tokens = _rotation_minimum(primitive)
    return Normalization("strict", "*" + "".join(canonical_tokens) + "*",
                         canonical_tokens, repeats)
