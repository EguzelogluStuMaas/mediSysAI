import re
from dataclasses import dataclass

TURN_PATTERN = re.compile(r"^(?P<speaker>[^:]+):\s*(?P<text>.+)$")


@dataclass
class Turn:
    index: int
    speaker: str
    text: str


def is_mediator(speaker: str) -> bool:
    return "mediator" in speaker.strip().lower()


def parse_transcript(raw_text: str) -> list[Turn]:
    turns: list[Turn] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = TURN_PATTERN.match(line)
        if not match:
            continue
        turns.append(
            Turn(
                index=len(turns),
                speaker=match.group("speaker").strip(),
                text=match.group("text").strip(),
            )
        )
    return turns
