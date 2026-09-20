import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

import spacy

from sentiment import analyze_sentiment
from transcript_parser import Turn, parse_transcript
# anahtar kelimeler şimdilik ingilizce için yapıldı, türkçe için NLP kütüphanesi bulunacak ve eklenecektir
DEFAULT_RESOLUTION_KEYWORDS: tuple[str, ...] = (
    "agreement",
    "agreed",
    "compromise",
    "resolved",
    "resolution",
    "deal",
)

_OPEN_QUESTION_STARTERS: tuple[str, ...] = (
    "what",
    "how",
    "why",
    "could you",
    "what do you think",
    "tell me about",
    "can you describe",
)

_EMPATHY_PHRASES: tuple[str, ...] = (
    "i understand",
    "i hear you",
    "i hear that",
    "how do you feel",
    "how does that feel",
    "that makes sense",
    "i can see that",
    "sounds like",
    "makes sense that",
)

_DISCOURSE_CONNECTORS: tuple[str, ...] = (
    "because",
    "therefore",
    "however",
    "so that",
    "as a result",
    "which means",
    "in other words",
)

_nlp_instance: spacy.language.Language | None = None

# Metni cümlelere ayırmak için temel bir spaCy NLP yapısı oluşturur.
def _nlp() -> spacy.language.Language:
    global _nlp_instance
    if _nlp_instance is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _nlp_instance = nlp
    return _nlp_instance

# Metnin hece sayısını hesaplar.
def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

# Belirli konuşmacılara ait konuşma sıralarını filtreler. Kimin sırarı ise onun konuşmasını (iterable) verir.
def _filter_turns(turns: Sequence[Turn], speakers: Iterable[str] | None) -> list[Turn]:
    if speakers is None:
        return list(turns)
    wanted = {s.lower() for s in speakers}
    return [t for t in turns if t.speaker.lower() in wanted]

# Konuşma turlarının duygu analizini hesaplar. A konuşmalarının skoru x B konusşmacısının konuşmalarının skoru y gibi. bu skorların ortalamasını verir.
def _sentiment_score(turns: Sequence[Turn]) -> float:
    if not turns:
        return 0.0
    scores = [analyze_sentiment(t.text).compound for t in turns]
    return sum(scores) / len(scores)

# Açık uçlu soruların sıklığını hesaplar. Bu açkık uçlu sorular OPEN_QUESTION_STARTERS ile başlayan ve soru işareti ile biten cümlelerdir.
def _open_ended_question_frequency(turns: Sequence[Turn]) -> float:
    if not turns:
        return 0.0
    hits = 0
    for turn in turns:
        stripped = turn.text.strip()
        if not stripped.endswith("?"):
            continue
        lowered = stripped.lower()
        if any(lowered.startswith(starter) for starter in _OPEN_QUESTION_STARTERS):
            hits += 1
    return hits / len(turns)

# Empati ifadelerinin sayısını hesaplar.
def _empathy_marker_count(joined_text: str) -> int:
    lowered = joined_text.lower()
    return sum(lowered.count(phrase) for phrase in _EMPATHY_PHRASES)

# Dilin karmaşıklığını hesaplar. Ortalama hece sayısı üzerinden bir ölçüm sağlar. Uzun cümleler için daha yüksek bir karmaşıklık değeri döner bu da toplam skordan çıkartılır

def _language_complexity(joined_text: str, nlp: spacy.language.Language) -> float:
    doc = nlp(joined_text)
    words = [tok.text for tok in doc if tok.is_alpha]
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    return syllables / len(words)

# Akıcılık hesabı. Cümle uzunluklarındaki tutarlılık ve bağlaç kullanımına göre bir ölçüm sağlar.
# Akıcılık skoru 1 - (std_dev / avg_len). (std_dev / avg_len) 0 a yakınsa Cümle uzunlukları birbirine çok yakındır ve oldukça düzenlidir.
#(std_dev / avg_len) 1 e yakınlaştıkça cümle uzunlukları arasındaki tutarsızlık artar ve akıcılık düşer.
def _syntactic_coherence(joined_text: str, nlp: spacy.language.Language) -> float:
    doc = nlp(joined_text)
    sentences = [sent for sent in doc.sents if sent.text.strip()]
    if not sentences:
        return 0.0

    lengths = [len([tok for tok in sent if tok.is_alpha]) for sent in sentences]
    avg_len = statistics.mean(lengths)
    if avg_len == 0:
        consistency = 0.0
    elif len(lengths) > 1:
        std_dev = statistics.stdev(lengths)
        consistency = 1 - min(std_dev / avg_len, 1.0)
    else:
        consistency = 1.0

    connector_hits = sum(
        1 for sent in sentences if any(c in sent.text.lower() for c in _DISCOURSE_CONNECTORS)
    )
    connector_ratio = connector_hits / len(sentences)

    return (consistency + connector_ratio) / 2

# Anahtar kelimelerin metin içindeki sıklığını hesaplar.
def _keyword_frequency(joined_text: str, keywords: Sequence[str]) -> dict[str, int]:
    lowered = joined_text.lower()
    return {keyword: lowered.count(keyword.lower()) for keyword in keywords}

# Metnin okunabilirlik skorunu hesaplar. Flesch-Kincaid formülüne dayalıdır.
def _readability_score(joined_text: str, nlp: spacy.language.Language) -> float:
    doc = nlp(joined_text)
    sentences = [sent for sent in doc.sents if sent.text.strip()]
    words = [tok.text for tok in doc if tok.is_alpha]
    if not sentences or not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)
    return 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59


@dataclass
class TextFeatures:
    sentiment_score: float
    open_ended_question_frequency: float
    empathy_marker_count: int
    language_complexity: float
    syntactic_coherence: float
    keyword_frequency: dict[str, int]
    readability_score: float

# Metin özelliklerini çıkarır ve TextFeatures veri sınıfı olarak döner.
def extract_text_features(
    turns: Sequence[Turn],
    speakers: Iterable[str] | None = None,
    keywords: Sequence[str] = DEFAULT_RESOLUTION_KEYWORDS,
) -> TextFeatures:
    subset = _filter_turns(turns, speakers)
    joined_text = " ".join(t.text for t in subset)
    nlp = _nlp()

    return TextFeatures(
        sentiment_score=_sentiment_score(subset),
        open_ended_question_frequency=_open_ended_question_frequency(subset),
        empathy_marker_count=_empathy_marker_count(joined_text),
        language_complexity=_language_complexity(joined_text, nlp),
        syntactic_coherence=_syntactic_coherence(joined_text, nlp),
        keyword_frequency=_keyword_frequency(joined_text, keywords),
        readability_score=_readability_score(joined_text, nlp),
    )


@dataclass

class VisualCueFeatures:
    eye_contact_duration: float
    head_tilt_degrees: float
    smiling_frequency: float
    body_posture_openness: float
    gesture_frequency: float

# mikro ifadeler face_mesh
def extract_visual_cue_features(frames=None) -> VisualCueFeatures:
    raise NotImplementedError(
        "Visual cue extraction requires video input - not yet available in this pipeline."
    )


@dataclass
# yüz ifadeleri özellikleri sonra yapılacak
class ExpressionFeatures:
    emotional_expression: dict[str, float]
    microexpression_frowning_frequency: float
    head_nod_frequency: float

# videodan almak için fonksiyon
def extract_expression_features(frames=None) -> ExpressionFeatures:
    raise NotImplementedError(
        "Facial expression extraction requires video input - not yet available in this pipeline."
    )

# ses tonu özellikleri sonra yapılacak
@dataclass
class VoiceToneFeatures:
    speech_rate_wpm: float
    pitch_variation: float
    pause_filler_frequency: float
    tone_consistency: float
    volume_variation: float


def extract_voice_tone_features(audio=None) -> VoiceToneFeatures:
    raise NotImplementedError(
        "Voice tone extraction requires audio input - not yet available in this pipeline."
    )


@dataclass
class SessionFeatures:
    text: TextFeatures
    visual: VisualCueFeatures | None = None
    expressions: ExpressionFeatures | None = None
    voice: VoiceToneFeatures | None = None


def extract_session_features(transcript_text: str) -> SessionFeatures:
    turns = parse_transcript(transcript_text)
    return SessionFeatures(text=extract_text_features(turns))
