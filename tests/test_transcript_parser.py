"""Parser tests need only the standard library, not downloaded models."""

import json
import unittest
from dataclasses import is_dataclass
from pathlib import Path

from app.transcript_parser import (
    Turn,
    is_mediator,
    parse_transcript,
    parse_transcript_document,
    validate_transcript,
)


def asr_document():
    return {
        "session_id": "session_001",
        "speakers": [
            {"id": "speaker_00", "display_name": "Konuşmacı 1"},
            {"id": "speaker_01", "display_name": "Konuşmacı 2"},
        ],
        "turns": [
            {"id": "a", "speaker_id": "speaker_00",
             "text_original": "Sizi anlıyorum.", "start_ms": 100, "end_ms": 1500,
             "source": "asr"},
            {"id": "b", "speaker_id": "speaker_01",
             "text_original": "Kabul etmiyorum.", "start_ms": 1200, "end_ms": 2500,
             "source": "asr", "flags": ["overlap"]},
        ],
    }


class ManualTranscriptTests(unittest.TestCase):
    def test_legacy_dataclass_constructor(self):
        turn = Turn(0, "Mediator", "Hello")
        self.assertTrue(is_dataclass(turn))
        self.assertEqual((turn.index, turn.speaker, turn.text), (0, "Mediator", "Hello"))
        self.assertEqual(turn.text_original, "Hello")
        turn.flags.append("test")
        self.assertEqual(Turn(1, "Party A", "Hi").flags, [])

    def test_existing_sample_and_legacy_fields(self):
        path = Path(__file__).resolve().parents[1] / "app" / "sample_transcript.txt"
        raw = path.read_text(encoding="utf-8")
        turns = parse_transcript(raw)
        expected = [line.split(":", 1) for line in raw.splitlines() if line.strip()]
        self.assertEqual(len(turns), len(expected))
        for index, (turn, (speaker, text)) in enumerate(zip(turns, expected)):
            self.assertEqual((turn.index, turn.speaker, turn.text), (index, speaker, text.strip()))

    def test_multiline_turns_and_blank_paragraphs(self):
        raw = "\ufeffArabulucu: Sizi anlıyorum.\n\nBiraz açar mısınız?\nTaraf A: Evet."
        doc = parse_transcript_document(raw, session_id="session")
        self.assertEqual(len(doc.turns), 2)
        self.assertEqual(doc.turns[0].text, "Sizi anlıyorum.\n\nBiraz açar mısınız?")
        self.assertIsNone(doc.turns[0].start_ms)
        self.assertIsNone(doc.turns[0].end_ms)
        self.assertEqual(doc.turns[0].session_id, "session")

    def test_known_labels_preserve_colons_inside_speech(self):
        raw = "Ayşe: İki konu var.\nÖdeme: henüz yapılmadı.\nMehmet: Açıklayayım."
        turns = parse_transcript(raw, speaker_labels=["Ayşe", "Mehmet"])
        self.assertEqual(len(turns), 2)
        self.assertIn("Ödeme: henüz yapılmadı.", turns[0].text)

    def test_legacy_custom_speakers(self):
        turns = parse_transcript("Ayşe: Merhaba.\nMehmet: Merhaba.")
        self.assertEqual([t.speaker for t in turns], ["Ayşe", "Mehmet"])

    def test_empty_header_can_have_multiline_speech(self):
        turns = parse_transcript("Arabulucu:\nMerhaba.\nTaraf A: İyi günler.")
        self.assertEqual(turns[0].text, "Merhaba.")

    def test_unassigned_text_and_empty_turn_are_visible(self):
        raw = "Unassigned preamble\nArabulucu:\nTaraf A: Merhaba."
        doc = parse_transcript_document(raw)
        self.assertEqual([i.code for i in doc.issues], ["unassigned_text", "empty_turn"])
        self.assertEqual([i.line_number for i in doc.issues], [1, 2])
        self.assertEqual(len(doc.turns), 1)
        with self.assertRaisesRegex(ValueError, "Line 1"):
            parse_transcript(raw)
        restored = parse_transcript_document(doc.to_dict(), format="json")
        self.assertEqual(restored.issues, doc.issues)

    def test_speakers_are_normalized_and_roles_are_unconfirmed_suggestions(self):
        doc = parse_transcript_document("ARABULUCU: Merhaba.\narabulucu: Devam edelim.")
        self.assertEqual(len(doc.speakers), 1)
        speaker = next(iter(doc.speakers.values()))
        self.assertEqual(speaker.role, "mediator")
        self.assertFalse(speaker.role_confirmed)
        self.assertEqual(len(doc.select_turns("mediator")), 2)
        self.assertTrue(is_mediator(" Arabulucu "))
        self.assertFalse(is_mediator("not a mediator"))

    def test_empty_input(self):
        self.assertEqual(parse_transcript(" \n"), [])

    def test_corrections_preserve_original_and_can_be_reset(self):
        turn = parse_transcript("Taraf A: Kabul ediyorum.")[0]
        turn.apply_correction("Kabul etmiyorum.")
        self.assertEqual(turn.text_original, "Kabul ediyorum.")
        self.assertEqual(turn.text, "Kabul etmiyorum.")
        self.assertEqual(turn.text_corrected, turn.text)
        turn.apply_correction(None)
        self.assertEqual(turn.text, turn.text_original)
        with self.assertRaises(ValueError):
            turn.apply_correction(" ")


class StructuredTranscriptTests(unittest.TestCase):
    def test_asr_overlap_and_role_assignment(self):
        doc = parse_transcript_document(asr_document(), format="json")
        self.assertEqual(doc.turns[1].start_ms, 1200)
        self.assertEqual(doc.turns[1].flags, ["overlap"])
        self.assertEqual(doc.turns[0].source, "asr")
        self.assertEqual(doc.select_turns("mediator"), [])
        doc.assign_role("speaker_00", "mediator")
        self.assertEqual(doc.select_turns("mediator"), [doc.turns[0]])
        self.assertTrue(doc.speakers["speaker_00"].role_confirmed)
        self.assertEqual(len(doc.turns), 2)  # Context is never removed.

    def test_json_round_trip_preserves_corrections_and_roles(self):
        doc = parse_transcript_document(asr_document(), format="json")
        doc.assign_role("speaker_00", "mediator")
        doc.turns[0].apply_correction("Sizi doğru anlıyor muyum?")
        restored = parse_transcript_document(json.dumps(doc.to_dict(), ensure_ascii=False), format="json")
        self.assertEqual(restored, doc)

    def test_unknown_assignment_is_not_forced(self):
        data = asr_document()
        data["turns"][0]["speaker_id"] = None
        doc = parse_transcript_document(data, format="json")
        self.assertIsNone(doc.turns[0].speaker_id)
        self.assertEqual(doc.turns[0].speaker, "Unknown")
        self.assertEqual(len(doc.select_turns("unknown")), 2)

    def test_invalid_timestamps(self):
        for start, end in [(None, 10), (1, None), (-1, 5), (10, 10), (20, 10),
                           (1.5, 10), (True, 10), ("1", 10)]:
            with self.subTest(start=start, end=end):
                data = asr_document()
                data["turns"][0].update(start_ms=start, end_ms=end)
                with self.assertRaises(ValueError):
                    parse_transcript_document(data, format="json")

    def test_missing_timestamps_are_allowed(self):
        data = asr_document()
        data["turns"][0].update(start_ms=None, end_ms=None)
        doc = parse_transcript_document(data, format="json")
        self.assertIsNone(doc.turns[0].start_ms)

    def test_invalid_turn_fields(self):
        for changes in [{"speaker_id": "missing"}, {"speaker_id": []},
                        {"text_original": " "}, {"text_corrected": 2},
                        {"source": "guess"}, {"source": []},
                        {"flags": "overlap"}, {"flags": [1]}, {"id": ""}]:
            with self.subTest(changes=changes):
                data = asr_document()
                data["turns"][0].update(changes)
                with self.assertRaises(ValueError):
                    parse_transcript_document(data, format="json")

    def test_invalid_speaker_fields(self):
        for changes in [{"id": ""}, {"display_name": ""}, {"role": "judge"},
                        {"role": []}, {"role_confirmed": "yes"}]:
            with self.subTest(changes=changes):
                data = asr_document()
                data["speakers"][0].update(changes)
                with self.assertRaises(ValueError):
                    parse_transcript_document(data, format="json")

    def test_duplicate_ids(self):
        for collection in ["speakers", "turns"]:
            with self.subTest(collection=collection):
                data = asr_document()
                data[collection].append(data[collection][0].copy())
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    parse_transcript_document(data, format="json")

    def test_session_conflict_and_session_scoping(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            parse_transcript_document(asr_document(), format="json", session_id="other")
        first = parse_transcript_document("Mediator: Hi.")
        second = parse_transcript_document("Mediator: Hi.")
        self.assertNotEqual(first.session_id, second.session_id)
        first.turns[0].session_id = second.session_id
        with self.assertRaisesRegex(ValueError, "another session"):
            validate_transcript(first)

    def test_invalid_input_shapes(self):
        for data in ["{broken", "[]", {}, {"speakers": {}, "turns": []},
                     {"speakers": [], "turns": ["bad"]}]:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    parse_transcript_document(data, format="json")
        with self.assertRaises(ValueError):
            parse_transcript_document("text", format="csv")
        with self.assertRaises(ValueError):
            parse_transcript_document({}, format="txt")
        with self.assertRaises(ValueError):
            parse_transcript_document("Mediator: Hi.", speaker_labels=[])


if __name__ == "__main__":
    unittest.main()
