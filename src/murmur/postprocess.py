"""Clean up recognised text before it is injected.

Parakeet transcribes faithfully, which means disfluencies land in the text:
"Well it seems to be working. Um I guess..." Faithful is right for a meeting
transcript and wrong for dictation, where you want what you meant rather than
what you said.

Kept deliberately conservative. Only standalone filler tokens are removed, and
only when removing them leaves something behind, so an utterance that is purely
"um" still produces text rather than silently vanishing.
"""

from __future__ import annotations

import re

# Whole-word fillers only. "ah" and "er" are included but "a", "I" and similar
# are not, because stripping real words is far worse than leaving a filler in.
FILLERS = ("um", "uh", "erm", "err", "hmm", "mhm", "uhm")

_FILLER_RE = re.compile(
    r"(?<![\w'])(?:" + "|".join(FILLERS) + r")[,.]?(?![\w'])",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def strip_fillers(text: str) -> str:
    """Remove standalone filler words, then repair the spacing and casing they
    leave behind. Returns the original text if the result would be empty."""
    if not text or not text.strip():
        return text

    cleaned = _FILLER_RE.sub("", text)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return text

    # A filler at the start of a sentence takes the capital with it when it
    # goes: "Um I guess" -> "i guess". Put it back.
    cleaned = _recapitalise(cleaned)
    return cleaned


def _recapitalise(text: str) -> str:
    """Capitalise the first letter, and the first letter after . ? or !"""
    out = list(text)
    capitalise_next = True
    for i, ch in enumerate(out):
        if capitalise_next and ch.isalpha():
            out[i] = ch.upper()
            capitalise_next = False
        elif ch in ".?!":
            capitalise_next = True
    return "".join(out)


def clean(text: str, strip_filler_words: bool = True) -> str:
    """The full post-processing pass applied to every take."""
    text = (text or "").strip()
    if strip_filler_words:
        text = strip_fillers(text)
    return text
