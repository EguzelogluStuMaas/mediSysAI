"""Parse manual and ASR transcripts without running speech models.

parse_transcript retains the list[Turn] API. parse_transcript_document also
returns a speaker registry and recoverable TXT issues. Supply speaker_labels
to disambiguate colons within speech; automatic TXT headers retain the legacy
Speaker: text convention. All structured timestamps are milliseconds.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

Role = Literal["mediator", "party_a", "party_b", "unknown"]
ROLES = ("mediator", "party_a", "party_b", "unknown")
TURN_PATTERN = re.compile(r"^(?P<speaker>[^:\n]+):\s*(?P<text>.*)$")


def _normalize_label(label: str) -> str:
    return " ".join(label.translate(str.maketrans("Iİ", "ıi")).lower().split())


def is_mediator(speaker: str) -> bool:
    """Recognize conventional labels, not substrings or actual identities."""
    return _normalize_label(speaker) in {"mediator", "arabulucu"}


def _suggest_role(label: str) -> Role:
    roles_by_label: dict[str, Role] = {
        "mediator": "mediator", "arabulucu": "mediator",
        "party b": "party_b", "taraf b": "party_b",
    }
    return roles_by_label.get(_normalize_label(label), "unknown")




@dataclass
class Turn:
    # Preserve the original constructor and fields used by features/scoring.
    index: int
    text: str
    id: str = ""
    session_id: str = ""
    speaker_id: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    source: Literal["manual", "asr"] = "manual"
    flags: list[str] = field(default_factory=list)
    text_corrected: str | None = None
    text_original: str = field(init=False)

    def __post_init__(self) -> None:
        self.text_original = self.text
        self.apply_correction(self.text_corrected)

    def apply_correction(self, text: str | None) -> None:
        """Update text for existing analysis code; None restores the original."""
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError("Corrected text must be a non-empty string or None.")
        self.text_corrected = text
        self.text = self.text_original if text is None else text


@dataclass
class Speaker:
    id: str
    display_name: str
    role: Role = "unknown"
    role_confirmed: bool = False


@dataclass
class ParseIssue:
    code: str
    message: str
    line_number: int | None = None


@dataclass
class Transcript:
    session_id: str
    speakers: dict[str, Speaker] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)

    def assign_role(self, speaker_id: str, role: Role) -> None:
        if role not in ROLES:
            raise ValueError(f"Unsupported speaker role: {role}")
        speaker = self.speakers[speaker_id]
        speaker.role = role
        speaker.role_confirmed = True

    def select_turns(self, role: Role) -> list[Turn]:
        if role not in ROLES:
            raise ValueError(f"Unsupported speaker role: {role}")
        return [
            turn for turn in self.turns
            if (self.speakers[turn.speaker_id].role if turn.speaker_id else "unknown") == role
        ]

    def to_dict(self) -> dict:
        """JSON representation, retaining original text and parsing issues."""
        return {
            "session_id": self.session_id,
            "speakers": [
                {"id": s.id, "display_name": s.display_name,
                 "role": s.role, "role_confirmed": s.role_confirmed}
                for s in self.speakers.values()
            ],
            "turns": [
                {"id": t.id, "speaker_id": t.speaker_id,
                 "text_original": t.text_original, "text_corrected": t.text_corrected,
                 "start_ms": t.start_ms, "end_ms": t.end_ms,
                 "source": t.source, "flags": list(t.flags)}
                for t in self.turns
            ],
            "issues": [
                {"code": i.code, "message": i.message, "line_number": i.line_number}
                for i in self.issues
            ],
        }


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def validate_transcript(transcript: Transcript) -> None:
    """Validate individual intervals; overlapping turns are allowed."""
    _nonempty_string(transcript.session_id, "session_id")
    seen_ids: set[str] = set()
    for key, speaker in transcript.speakers.items():
        _nonempty_string(speaker.id, "speaker.id")
        _nonempty_string(speaker.display_name, "speaker.display_name")
        if key != speaker.id or speaker.role not in ROLES:
            raise ValueError("Invalid speaker registry entry.")
        if not isinstance(speaker.role_confirmed, bool):
            raise ValueError("role_confirmed must be a boolean.")
    for turn in transcript.turns:
        _nonempty_string(turn.id, "turn.id")
        if turn.id in seen_ids:
            raise ValueError(f"Duplicate turn ID: {turn.id}")
        seen_ids.add(turn.id)
        if turn.session_id != transcript.session_id:
            raise ValueError(f"Turn {turn.id} belongs to another session.")
        if turn.speaker_id is not None:
            _nonempty_string(turn.speaker_id, "turn.speaker_id")
            if turn.speaker_id not in transcript.speakers:
                raise ValueError(f"Turn {turn.id} references an unknown speaker.")
        _nonempty_string(turn.text_original, f"Turn {turn.id} text_original")
        _nonempty_string(turn.text, f"Turn {turn.id} text")
        if turn.source not in ("manual", "asr"):
            raise ValueError(f"Turn {turn.id} has an unsupported source.")
        if not isinstance(turn.flags, list) or any(not isinstance(f, str) for f in turn.flags):
            raise ValueError(f"Turn {turn.id} flags must be a list of strings.")
        start, end = turn.start_ms, turn.end_ms
        if start is None and end is None:
            continue
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"Turn {turn.id} timestamps must both be integer milliseconds or null.")
        if start < 0 or end <= start:
            raise ValueError(f"Turn {turn.id} requires 0 <= start_ms < end_ms.")


def _parse_txt(raw_text: str, session_id: str, speaker_labels: list[str] | None) -> Transcript:
    transcript = Transcript(session_id)
    allowed = None
    if speaker_labels is not None:
        allowed = {
            _normalize_label(_nonempty_string(label, "speaker label"))
            for label in speaker_labels
        }
        if not allowed:
            raise ValueError("speaker_labels must not be empty when supplied.")
    by_label: dict[str, str] = {}
    current_label: str | None = None
    current_lines: list[str] = []
    header_line = 0

    def finish_turn() -> None:
        if current_label is None:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            transcript.issues.append(ParseIssue(
                "empty_turn", f"Speaker {current_label!r} has no speech.", header_line
            ))
            return
        key = _normalize_label(current_label)
        if key not in by_label:
            speaker_id = f"speaker_{len(by_label):02d}"
            by_label[key] = speaker_id
            transcript.speakers[speaker_id] = Speaker(
                speaker_id, current_label, _suggest_role(current_label)
            )
        speaker_id = by_label[key]
        index = len(transcript.turns)
        transcript.turns.append(Turn(
            index=index, text=text, id=f"turn_{index:04d}", session_id=session_id,
            speaker_id=speaker_id,
        ))

    for line_number, line in enumerate(raw_text.lstrip("\ufeff").splitlines(), start=1):
        match = TURN_PATTERN.match(line.strip())
        label = match.group("speaker").strip() if match else None
        if match is not None and label and (allowed is None or _normalize_label(label) in allowed):
            finish_turn()
            current_label = label
            current_lines = [match.group("text")]
            header_line = line_number
        elif current_label is not None:
            current_lines.append(line)
        elif line.strip():
            transcript.issues.append(ParseIssue(
                "unassigned_text", "Text appears before a recognized speaker header.", line_number
            ))
    finish_turn()
    return transcript


def _parse_json(raw_text: str | dict, session_id: str | None) -> Transcript:
    try:
        data = json.loads(raw_text.lstrip("\ufeff")) if isinstance(raw_text, str) else raw_text
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid transcript JSON at line {error.lineno}: {error.msg}") from error
    if not isinstance(data, dict):
        raise ValueError("Transcript JSON must be an object.")
    file_session = data.get("session_id")
    if file_session is not None:
        _nonempty_string(file_session, "session_id")
    if session_id is not None and file_session is not None and session_id != file_session:
        raise ValueError("Supplied session_id conflicts with the JSON session_id.")
    transcript = Transcript(session_id or file_session or uuid4().hex)
    for name in ("speakers", "turns"):
        if not isinstance(data.get(name), list):
            raise ValueError(f"Transcript JSON requires a {name} list.")
    for entry in data["speakers"]:
        if not isinstance(entry, dict):
            raise ValueError("Each speaker must be an object.")
        speaker_id = _nonempty_string(entry.get("id"), "speaker.id")
        if speaker_id in transcript.speakers:
            raise ValueError(f"Duplicate speaker ID: {speaker_id}")
        transcript.speakers[speaker_id] = Speaker(
            id=speaker_id, display_name=entry.get("display_name", speaker_id),
            role=entry.get("role", "unknown"), role_confirmed=entry.get("role_confirmed", False),
        )
    for index, entry in enumerate(data["turns"]):
        if not isinstance(entry, dict):
            raise ValueError("Each turn must be an object.")
        speaker_id = entry.get("speaker_id")
        if speaker_id is not None:
            _nonempty_string(speaker_id, "turn.speaker_id")
            if speaker_id not in transcript.speakers:
                raise ValueError(f"Turn {index} references an unknown speaker: {speaker_id}")
        original = _nonempty_string(entry.get("text_original"), f"Turn {index} text_original")
        transcript.turns.append(Turn(
            index=index,
            text=original, id=entry.get("id", f"turn_{index:04d}"),
            session_id=transcript.session_id, speaker_id=speaker_id,
            start_ms=entry.get("start_ms"), end_ms=entry.get("end_ms"),
            source=entry.get("source", "manual"), flags=entry.get("flags", []),
            text_corrected=entry.get("text_corrected"),
        ))
    issues = data.get("issues", [])
    if not isinstance(issues, list):
        raise ValueError("issues must be a list.")
    for entry in issues:
        if not isinstance(entry, dict):
            raise ValueError("Each issue must be an object.")
        line_number = entry.get("line_number")
        if line_number is not None and (type(line_number) is not int or line_number < 1):
            raise ValueError("Issue line_number must be a positive integer or null.")
        transcript.issues.append(ParseIssue(
            _nonempty_string(entry.get("code"), "issue.code"),
            _nonempty_string(entry.get("message"), "issue.message"), line_number,
        ))
    return transcript


def parse_transcript_document(
    raw_text: str | dict, *, format: str = "txt", session_id: str | None = None,
    speaker_labels: list[str] | None = None,
) -> Transcript:
    """Return turns, speakers and recoverable issues; reject invalid JSON.

    Conventional TXT labels suggest roles but do not confirm them. Explicit
    speaker_labels prevent colons inside speech from becoming speaker headers.
    JSON requires speakers and turns lists; unknown assignments use null IDs.
    """
    if session_id is not None:
        _nonempty_string(session_id, "session_id")
    if format == "txt":
        if not isinstance(raw_text, str):
            raise ValueError("TXT input must be a string.")
        transcript = _parse_txt(raw_text, session_id or uuid4().hex, speaker_labels)
    elif format == "json":
        if speaker_labels is not None:
            raise ValueError("JSON uses its speaker registry, not speaker_labels.")
        transcript = _parse_json(raw_text, session_id)
    else:
        raise ValueError(f"Unsupported transcript format: {format}")
    validate_transcript(transcript)
    return transcript


def parse_transcript(
    raw_text: str | dict, *, format: str = "txt", session_id: str | None = None,
    speaker_labels: list[str] | None = None,
) -> list[Turn]:
    """Compatibility API: raise on issues instead of silently dropping text.

    Use parse_transcript_document to inspect valid turns alongside issues.
    """
    transcript = parse_transcript_document(
        raw_text, format=format, session_id=session_id, speaker_labels=speaker_labels
    )
    if transcript.issues:
        raise ValueError("; ".join(
            f"Line {issue.line_number}: {issue.message}" for issue in transcript.issues
        ))
    return transcript.turns

