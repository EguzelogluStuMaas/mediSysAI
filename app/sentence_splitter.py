"""Regex sentence boundaries and preprocessing; no speaker inference.

Offsets reference the exact input string, before whitespace normalization.
Heuristics cover common abbreviations/numbers, not every ambiguity in speech.
"""

import re
from dataclasses import dataclass
from typing import Iterable

if __package__:
    from .preprocessing import PreprocessedText, preprocess_text
else:
    from preprocessing import PreprocessedText, preprocess_text


END_PATTERN = re.compile(r"[.!?…]+[\"'”’»)\]]*")
ABBREVIATION_PATTERN = re.compile(
    r"\b(?:Dr|Av|Prof|Doç|Sn|Say|örn|md|mad|m)\.", re.IGNORECASE
)
INITIAL_PATTERN = re.compile(r"\b[A-ZÇĞİÖŞÜ]\.(?=\s*[A-ZÇĞİÖŞÜ])")
ACRONYM_PATTERN = re.compile(r"\b(?:[A-ZÇĞİÖŞÜ]\.){2,}")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)+")
ENUMERATION_PATTERN = re.compile(r"(?m)^\s*\d+\.(?=\s)")
ADDRESS_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"”]+|[\w.+-]+@[\w.-]+\.\w+")


@dataclass(frozen=True)
class Sentence:
    id: str
    turn_id: str
    session_id: str
    speaker_id: str | None
    speaker: str
    index: int
    original_text: str
    start_char: int
    end_char: int
    preprocessing: PreprocessedText

    @property
    def text(self) -> str:
        """Case/punctuation-preserving input for sentiment models."""
        return self.preprocessing.model_text


def _protected_positions(text: str) -> set[int]:
    positions: set[int] = set()
    for pattern in (ABBREVIATION_PATTERN, INITIAL_PATTERN, ACRONYM_PATTERN,
                    NUMBER_PATTERN, ENUMERATION_PATTERN, ADDRESS_PATTERN):
        for match in pattern.finditer(text):
            end = match.end()
            if pattern is ADDRESS_PATTERN:
                end = match.start() + len(match.group().rstrip(".!?…)]}"))
            positions.update(range(match.start(), end))
    return positions


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return trimmed, non-overlapping spans in original input coordinates."""
    protected = _protected_positions(text)
    spans: list[tuple[int, int]] = []
    start = 0

    def append_span(end: int) -> None:
        nonlocal start
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = end

    for match in END_PATTERN.finditer(text):
        if match.start() in protected:
            continue
        end = match.end()
        if end < len(text) and not text[end].isspace():
            continue
        # Keep a quoted sentence with its attribution: “Hayır.” dedi.
        remainder = text[end:].lstrip()
        if match.group()[-1] in '\"\'”’»' and remainder and remainder[0].islower():
            continue
        append_span(end)
    append_span(len(text))
    return spans


def split_sentences(
    text: str, *, session_id: str = "", turn_id: str = "",
    speaker_id: str | None = None, speaker: str = "Unknown",
    start_index: int = 0, remove_stopwords: bool = False,
    stopwords: Iterable[str] | None = None,
) -> list[Sentence]:
    """Split before preprocessing so source locations stay accurate.

    No sentence timestamps are fabricated from a turn's start/end times.
    """
    if isinstance(stopwords, str):
        raise ValueError("stopwords must be a collection of words, not one string.")
    # Materialize once, including when a caller supplies a generator.
    words = None if stopwords is None else tuple(stopwords)
    sentences = []
    for local_index, (start, end) in enumerate(sentence_spans(text)):
        original = text[start:end]
        sentences.append(Sentence(
            id=f"{turn_id or 'text'}:sentence_{local_index:04d}",
            turn_id=turn_id, session_id=session_id, speaker_id=speaker_id,
            speaker=speaker, index=start_index + local_index,
            original_text=original, start_char=start, end_char=end,
            preprocessing=preprocess_text(original, remove_stopwords=remove_stopwords, stopwords=words),
        ))
    return sentences
