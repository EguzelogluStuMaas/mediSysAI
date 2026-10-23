# Transcript parser

The parser uses standard-library dataclasses and does not load speech or NLP
models. Construct turns with `Turn(index=0, text="...", speaker="Arabulucu")`.
The first two positional arguments are index and text. `speaker_id` references
the registry, while `speaker` is a display-name snapshot for legacy analysis.
`parse_transcript(text) -> list[Turn]` remains available.

## Manual transcripts

Each turn starts with a `Speaker: text` header. Speech can continue on subsequent
lines, and a header may be followed by speech on the next line. UTF-8 BOMs are
accepted. No timestamps are invented for plain text.

```text
Arabulucu: Sizi anlıyorum.
Bu konuyu biraz daha açabilir misiniz?
Taraf A: Ödemenin gecikmesinden bahsediyorum.
```

For arbitrary names and speech containing colons, supply the known speaker
labels. Only those labels start new turns; other lines remain part of the
current turn. Without this list, the parser retains the legacy convention that
every non-empty prefix followed by a colon starts a turn. That automatic mode
cannot distinguish a person's name from a phrase such as `Ödeme: henüz yok`.

```python
from app.transcript_parser import parse_transcript_document

transcript = parse_transcript_document(
    "Ayşe: İki konu var.\nÖdeme: henüz yapılmadı.\nMehmet: Açıklayayım.",
    session_id="session_001",
    speaker_labels=["Ayşe", "Mehmet"],
)

transcript.assign_role("speaker_00", "mediator")
mediator_turns = transcript.select_turns("mediator")
```

Roles are `mediator`, `party_a`, `party_b`, or `unknown`. Conventional TXT names
such as Arabulucu and Taraf A suggest roles with `role_confirmed=False`.
`assign_role` records explicit confirmation. Role selection does not remove the
other turns from the transcript, so they remain available as conversation context.

Unassigned text before the first header and empty turns are reported in
`transcript.issues`, including line numbers. Inspect these before using or saving
partial results. The compatibility function `parse_transcript` raises `ValueError`
if there are any issues, rather than silently omitting text. Empty input returns
an empty turn list. Invalid structured input always raises `ValueError`.

## Structured transcripts

Use `parse_transcript_document(data, format="json")` with a JSON string or dict.
The `speakers` and `turns` lists are required. An absent session ID is generated;
pass or persist a session ID to reuse it on later loads. Speaker and turn IDs are
local to a session: use `(session_id, id)` as a cross-session key. Generated turn
IDs follow input order; retain IDs in JSON when editing an existing transcript.

```json
{
  "session_id": "session_001",
  "speakers": [
    {
      "id": "speaker_00",
      "display_name": "Konuşmacı 1",
      "role": "unknown",
      "role_confirmed": false
    }
  ],
  "turns": [
    {
      "id": "turn_0000",
      "speaker_id": "speaker_00",
      "text_original": "Sizi anlıyorum.",
      "text_corrected": null,
      "start_ms": 2100,
      "end_ms": 6200,
      "source": "asr",
      "flags": []
    }
  ]
}
```

Times must both be absent/null or both be integer milliseconds with
`0 <= start_ms < end_ms`. Overlapping turns are valid and are not automatically
merged or sorted. Speaker references must exist in the registry, or be null for
an unresolved assignment. Diarization labels do not automatically establish roles.
The audio pipeline must align words with speakers and convert seconds to
milliseconds before producing this format. Raw Whisper output is not this schema.

## Corrections and export

```python
import json

turn = transcript.turns[0]
turn.apply_correction("Sizi doğru anlıyor muyum?")
assert turn.text == turn.text_corrected
# turn.text_original still contains the parsed source speech.

payload = json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2)
restored = parse_transcript_document(payload, format="json")

turn.apply_correction(None)  # Restore original text.
```

Use `apply_correction` instead of assigning text fields directly. `turn.text`
contains the effective text for existing feature/scoring functions. The parser
preserves source speech separately from corrections, but does not retain the
byte-for-byte TXT file (speaker headers and surrounding whitespace are parsed).
Persist the uploaded file separately if it is needed. Invalidating model results
after a correction belongs to the analysis layer, not this parser.

The dashboard currently uploads TXT only; the JSON API is available for future
audio integration. It displays parser errors instead of a traceback.

## Sentences and preprocessing

`transcript.sentences()` splits each turn using `sentence_splitter.py` and runs
`preprocessing.py`. Results preserve conversation order, turn/session IDs,
speaker IDs and registry display names. No new speaker is inferred from sentence
boundaries. Results are generated on demand so corrected text is not cached.

```python
sentences = transcript.sentences(remove_stopwords=True)
for sentence in sentences:
    sentiment_input = sentence.text
    optional_bag_of_words = sentence.preprocessing.filtered_tokens
```

`sentence.text` preserves case, punctuation, negation and connective words; only
Unicode and whitespace are normalized. Use this input for sentiment/BERT and
use the model's own tokenizer. Stopword removal is off by default and affects
only `filtered_tokens` / `feature_text`. It never modifies model text or turns.
The default stopword list is small and configurable; negation, questions,
contrast markers and selected pronouns are protected even in custom lists.
No stemming or automatic spelling correction is applied.

`original_text` and `[start_char:end_char]` refer to the current **effective turn
text**, before normalization. After a correction they refer to corrected text,
not `Turn.text_original`. Sentence IDs follow position within the turn; analysis
results must be refreshed after edits. Sentence timestamps are not guessed from
the parent interval; use `turn_id` to retrieve the known parent timing.

The regex splitter handles common titles (Dr., Av.), initials, numbers, selected
legal abbreviations, URLs and quoted-speech attribution. It is heuristic: unusual
abbreviations and missing punctuation still need review. Plain newlines do not
imply new sentences or speakers. Document heading extraction is a separate step.

## Tests

From the repository root:

```powershell
python -B -m unittest discover -s tests -v
```

Tests cover legacy callers, multiline Turkish text, colon disambiguation, roles,
corrections, overlapping speech, invalid input, and JSON round trips. No model
downloads or external dependencies are required.
