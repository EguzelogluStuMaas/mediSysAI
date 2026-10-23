from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

if TYPE_CHECKING:
    import spacy

if __package__:
    from .sentiment import SentimentResult, analyze_sentiment
    from .sentence_splitter import Sentence
    from .transcript_parser import Turn, parse_transcript
else:
    from sentiment import SentimentResult, analyze_sentiment
    from sentence_splitter import Sentence
    from transcript_parser import Turn, parse_transcript



# Türkçe ifade listeleri


DEFAULT_RESOLUTION_KEYWORDS: tuple[str, ...] = (
    "anlaşma",
    "anlaştık",
    "uzlaşma",
    "çözüldü",
    "çözüm",
    "mutabakat",
)

OPEN_QUESTION_STARTERS: tuple[str, ...] = (
    "ne ",
    "nasıl",
    "neden",
    "hangi ",
    "sizce ",
    "anlatır mısınız",
    "tarif eder misiniz",
)

EMPATHY_PHRASES: tuple[str, ...] = (
    "sizi anlıyorum",
    "sizi duyuyorum",
    "söylediklerinizi anlıyorum",
    "nasıl hissediyorsunuz",
    "bu size nasıl hissettiriyor",
    "böyle hissetmeniz anlaşılır",
    "bunun sizin için zor olduğunu görüyorum",
    "anladığım kadarıyla",
    "böyle düşünmenizi anlayabiliyorum",
)

DISCOURSE_CONNECTORS: tuple[str, ...] = (
    "çünkü",
    "bu nedenle",
    "ancak",
    "böylece",
    "sonuç olarak",
    "bu da demek oluyor ki",
    "başka bir deyişle",
)

EVALUATIVE_PHRASES: tuple[str, ...] = (
    "haklı",
    "haksız",
    "haklısınız",
    "haksızsınız",
    "haklısın",
    "haksızsın",
    "suç sizde",
    "siz hatalısınız",
)

DIRECTIVE_PHRASES: tuple[str, ...] = (
    "kabul etmelisiniz",
    "kabul etmelisin",
    "kabul etmek zorundasınız",
    "imzalamalısınız",
    "başka seçeneğiniz yok",
)

PARTICIPATION_PHRASES: tuple[str, ...] = (
    "ne düşünüyorsunuz",
    "nasıl değerlendiriyorsunuz",
    "sizin görüşünüz",
    "sizi dinleyelim",
    "siz ne dersiniz",
    "anlatır mısınız",
)



# Ortak metin yardımcıları


def normalize_turkish(text: str) -> str:
    """Türkçe büyük/küçük harf ve boşluk normalizasyonu."""
    return " ".join(
        text.translate(str.maketrans("Iİ", "ıi"))
        .lower()
        .split()
    )


def contains_phrase(text: str, phrase: str) -> bool:
    """
    Kelime sınırlarını gözeterek ifade arar.

    Örneğin 'haklı' ifadesini 'haklılık' içinde eşleştirmez.
    Girdilerin önceden normalize edilmiş olması beklenir.
    """
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None



SUPPORT_CATEGORIES = (
    "empathy",
    "open_question",
    "participation_invitation",
)

REVIEW_CATEGORIES = (
    "evaluative_language",
    "directive_language",
)

COMMUNICATION_CATEGORIES = SUPPORT_CATEGORIES + REVIEW_CATEGORIES


@dataclass
class SentenceFeatures:
    sentence: Sentence
    matches: dict[str, tuple[str, ...]]
    resolution_keywords: tuple[str, ...]
    discourse_connectors: tuple[str, ...]
    sentiment: SentimentResult | None = None

    @property
    def indicators(self) -> dict[str, int]:
        """
        Her kategori bir cümlede en fazla 1 puan alır.

        Aynı cümlede birden fazla kategori bulunabilir.
        Aynı kategorideki tekrarlar puanı artırmaz.
        """
        return {
            category: int(bool(self.matches[category]))
            for category in COMMUNICATION_CATEGORIES
        }


def extract_sentence_features(
    sentences: Sequence[Sentence],
    *,
    speaker_ids: Iterable[str] | None = None,
    keywords: Sequence[str] = DEFAULT_RESOLUTION_KEYWORDS,
    sentiment_analyzer: Callable[[str], SentimentResult] | None = None,
) -> list[SentenceFeatures]:
    """
    Parser tarafından hazırlanmış cümleleri değerlendirir.

    Bu fonksiyon yeniden cümle bölmez veya turları birleştirmez.

    speaker_ids=None:
        Bütün konuşmacıları dahil eder.

    speaker_ids=[]:
        Hiçbir konuşmacıyı dahil etmez.

    sentiment_analyzer=None:
        Sentiment çalıştırılmaz; sonuç None kalır.

    Sentiment için sentence.text kullanılır. Stopword filtreli
    tokenlar sentiment girdisi olarak kullanılmaz.

    Bu kurallar alıntı, ironi ve olumsuzlama bağlamını çözmez.
    Çözüm kelimeleri anlaşma sağlandığının kanıtı değildir.
    """
    if isinstance(speaker_ids, str):
        raise ValueError(
            "speaker_ids must be a collection, not one string."
        )

    selected_ids = (
        None if speaker_ids is None else set(speaker_ids)
    )

    categories = {
        "empathy": EMPATHY_PHRASES,
        "participation_invitation": PARTICIPATION_PHRASES,
        "evaluative_language": EVALUATIVE_PHRASES,
        "directive_language": DIRECTIVE_PHRASES,
    }

    result: list[SentenceFeatures] = []

    for sentence in sentences:
        if (
            selected_ids is not None
            and sentence.speaker_id not in selected_ids
        ):
            continue

        if not sentence.text.strip():
            continue

        normalized = normalize_turkish(sentence.text)

        matches = {
            category: tuple(
                phrase
                for phrase in phrases
                if contains_phrase(normalized, phrase)
            )
            for category, phrases in categories.items()
        }

        question = normalized.strip(" \"'“”‘’«»")

        matches["open_question"] = tuple(
            starter.strip()
            for starter in OPEN_QUESTION_STARTERS
            if question.rstrip("!…").endswith("?")
            and re.match(
                r"^" + re.escape(starter.strip()) + r"(?!\w)",
                question,
            )
        )

        resolution_keywords = tuple(dict.fromkeys(
            keyword
            for keyword in keywords
            if contains_phrase(
                normalized,
                normalize_turkish(keyword),
            )
        ))

        discourse_connectors = tuple(
            phrase
            for phrase in DISCOURSE_CONNECTORS
            if contains_phrase(normalized, phrase)
        )

        sentiment = (
            sentiment_analyzer(sentence.text)
            if sentiment_analyzer is not None
            else None
        )

        result.append(
            SentenceFeatures(
                sentence=sentence,
                matches=matches,
                resolution_keywords=resolution_keywords,
                discourse_connectors=discourse_connectors,
                sentiment=sentiment,
            )
        )

    return result


# ---------------------------------------------------------
# Eski metin feature akışı
#
# Eski çağıranlarla uyumluluk için korunmuştur.
# Yeni cümle akışı aşağıdaki spaCy/hece/okunabilirlik
# hesaplarını kullanmaz.
# ---------------------------------------------------------

nlp_instance: spacy.language.Language | None = None


def nlp() -> spacy.language.Language:
    """Eski metin ölçümleri için spaCy hattını gerektiğinde yükler."""
    import spacy

    global nlp_instance

    if nlp_instance is None:
        pipeline = spacy.blank("en")
        pipeline.add_pipe("sentencizer")
        nlp_instance = pipeline

    return nlp_instance


def count_syllables(word: str) -> int:
    """
    Eski İngilizce hece tahmini.

    Türkçe için uyarlanmış değildir.
    """
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


def filter_turns(
    turns: Sequence[Turn],
    speakers: Iterable[str] | None,
) -> list[Turn]:
    """Eski API: konuşma turlarını görünen konuşmacı adına göre seçer."""
    if speakers is None:
        return list(turns)

    wanted = {speaker.lower() for speaker in speakers}

    return [
        turn
        for turn in turns
        if turn.speaker.lower() in wanted
    ]


def sentiment_score(turns: Sequence[Turn]) -> float:
    """Eski API: tur bazındaki compound skorlarının ortalaması."""
    if not turns:
        return 0.0

    scores = [
        analyze_sentiment(turn.text).compound
        for turn in turns
    ]

    return sum(scores) / len(scores)


def open_ended_question_frequency(
    turns: Sequence[Turn],
) -> float:
    """
    Eski API: açık uçlu soru olarak eşleşen turların oranı.

    Turun tamamının başlangıcına ve sonuna bakar.
    """
    if not turns:
        return 0.0

    hits = 0

    for turn in turns:
        stripped = turn.text.strip()

        if not stripped.endswith("?"):
            continue

        lowered = stripped.lower()

        if any(
            lowered.startswith(starter)
            for starter in OPEN_QUESTION_STARTERS
        ):
            hits += 1

    return hits / len(turns)


def empathy_marker_count(joined_text: str) -> int:
    """Eski API: empati ifadelerinin metindeki tekrar sayısı."""
    lowered = joined_text.lower()

    return sum(
        lowered.count(phrase)
        for phrase in EMPATHY_PHRASES
    )


def language_complexity(
    joined_text: str,
    nlp: spacy.language.Language,
) -> float:
    """Eski API: kelime başına tahmini hece sayısı."""
    doc = nlp(joined_text)
    words = [token.text for token in doc if token.is_alpha]

    if not words:
        return 0.0

    syllables = sum(count_syllables(word) for word in words)
    return syllables / len(words)


def syntactic_coherence(
    joined_text: str,
    nlp: spacy.language.Language,
) -> float:
    """
    Eski API: cümle uzunluğu tutarlılığı ve bağlaç oranı.

    Anlamsal tutarlılık ölçümü değildir.
    """
    doc = nlp(joined_text)
    sentences = [
        sentence
        for sentence in doc.sents
        if sentence.text.strip()
    ]

    if not sentences:
        return 0.0

    lengths = [
        len([token for token in sentence if token.is_alpha])
        for sentence in sentences
    ]

    avg_len = statistics.mean(lengths)

    if avg_len == 0:
        consistency = 0.0
    elif len(lengths) > 1:
        std_dev = statistics.stdev(lengths)
        consistency = 1 - min(std_dev / avg_len, 1.0)
    else:
        consistency = 1.0

    connector_hits = sum(
        1
        for sentence in sentences
        if any(
            connector in sentence.text.lower()
            for connector in DISCOURSE_CONNECTORS
        )
    )

    connector_ratio = connector_hits / len(sentences)
    return (consistency + connector_ratio) / 2


def keyword_frequency(
    joined_text: str,
    keywords: Sequence[str],
) -> dict[str, int]:
    """Eski API: anahtar kelimelerin metindeki tekrar sayıları."""
    lowered = joined_text.lower()

    return {
        keyword: lowered.count(keyword.lower())
        for keyword in keywords
    }


def readability_score(
    joined_text: str,
    nlp: spacy.language.Language,
) -> float:
    """
    Eski Flesch-Kincaid benzeri okunabilirlik hesabı.

    Türkçe için uyarlanmış değildir.
    """
    doc = nlp(joined_text)

    sentences = [
        sentence
        for sentence in doc.sents
        if sentence.text.strip()
    ]

    words = [
        token.text
        for token in doc
        if token.is_alpha
    ]

    if not sentences or not words:
        return 0.0

    syllables = sum(count_syllables(word) for word in words)

    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)

    return (
        0.39 * words_per_sentence
        + 11.8 * syllables_per_word
        - 15.59
    )


@dataclass
class TextFeatures:
    sentiment_score: float
    open_ended_question_frequency: float
    empathy_marker_count: int
    language_complexity: float
    syntactic_coherence: float
    keyword_frequency: dict[str, int]
    readability_score: float


def extract_text_features(
    turns: Sequence[Turn],
    speakers: Iterable[str] | None = None,
    keywords: Sequence[str] = DEFAULT_RESOLUTION_KEYWORDS,
) -> TextFeatures:
    """Eski tur bazlı metin feature API'si."""
    subset = filter_turns(turns, speakers)
    joined_text = " ".join(turn.text for turn in subset)
    nlp_pipeline = nlp()

    return TextFeatures(
        sentiment_score=sentiment_score(subset),
        open_ended_question_frequency=(
            open_ended_question_frequency(subset)
        ),
        empathy_marker_count=empathy_marker_count(joined_text),
        language_complexity=language_complexity(
            joined_text,
            nlp_pipeline,
        ),
        syntactic_coherence=syntactic_coherence(
            joined_text,
            nlp_pipeline,
        ),
        keyword_frequency=keyword_frequency(
            joined_text,
            keywords,
        ),
        readability_score=readability_score(
            joined_text,
            nlp_pipeline,
        ),
    )


# ---------------------------------------------------------
# Eski tur bazlı tarafsızlık bulguları
# ---------------------------------------------------------

@dataclass
class ImpartialityEvidence:
    turn_index: int
    speaker: str
    text: str
    category: str
    matched_phrase: str
    addressed_party: str | None = None


@dataclass
class ImpartialityFeatures:
    mediator_turn_count: int
    evaluative_language_turn_count: int
    directive_language_turn_count: int
    participation_invitation_turn_count: int
    invitations_by_party: dict[str, int]
    evidence: list[ImpartialityEvidence]


def extract_impartiality_features(
    turns: Sequence[Turn],
    speakers: Iterable[str],
) -> ImpartialityFeatures:
    """
    Seçilen arabulucunun dilinden incelemeye açık bulgular çıkarır.

    speakers açıkça verilmelidir.

    Diğer konuşmacılar taraf kabul edilir. Davet ancak tek bir
    açık hitap bulunduğunda bir tarafa bağlanır.

    Her tur kategori başına en fazla bir kez sayılır.

    Alıntı, ironi ve olumsuzlama çözülmez. Eşleşmeler doğrulanmış
    taraflılık veya baskı tespiti olarak yorumlanmamalıdır.
    """
    wanted = {
        normalize_turkish(speaker)
        for speaker in speakers
    }

    if not wanted or "" in wanted:
        raise ValueError(
            "En az bir arabulucu konuşmacısı seçilmelidir."
        )

    parties: dict[str, str] = {}
    selected: list[Turn] = []

    for turn in turns:
        speaker = normalize_turkish(turn.speaker)

        if speaker in wanted:
            if turn.text.strip():
                selected.append(turn)
        elif speaker:
            parties.setdefault(speaker, turn.speaker)

    invitations = {
        name: 0
        for name in parties.values()
    }

    evidence: list[ImpartialityEvidence] = []

    counts = {
        "evaluative_language": 0,
        "directive_language": 0,
        "participation_invitation": 0,
    }

    categories = {
        "evaluative_language": EVALUATIVE_PHRASES,
        "directive_language": DIRECTIVE_PHRASES,
        "participation_invitation": PARTICIPATION_PHRASES,
    }

    for turn in selected:
        normalized = normalize_turkish(turn.text)

        addressed = [
            name
            for key, name in parties.items()
            if re.search(
                r"(?:^|[.!?]\s+)"
                + re.escape(key)
                + r"\s*[,;:]",
                normalized,
            )
        ]

        for category, phrases in categories.items():
            matches = [
                phrase
                for phrase in phrases
                if contains_phrase(normalized, phrase)
            ]

            if not matches:
                continue

            counts[category] += 1
            party = None

            if (
                category == "participation_invitation"
                and len(addressed) == 1
            ):
                party = addressed[0]
                invitations[party] += 1

            for phrase in matches:
                evidence.append(
                    ImpartialityEvidence(
                        turn_index=turn.index,
                        speaker=turn.speaker,
                        text=turn.text,
                        category=category,
                        matched_phrase=phrase,
                        addressed_party=party,
                    )
                )

    return ImpartialityFeatures(
        mediator_turn_count=len(selected),
        evaluative_language_turn_count=(
            counts["evaluative_language"]
        ),
        directive_language_turn_count=(
            counts["directive_language"]
        ),
        participation_invitation_turn_count=(
            counts["participation_invitation"]
        ),
        invitations_by_party=invitations,
        evidence=evidence,
    )


# ---------------------------------------------------------
# Video ve ses feature iskeletleri
# ---------------------------------------------------------

@dataclass
class VisualCueFeatures:
    eye_contact_duration: float
    head_tilt_degrees: float
    smiling_frequency: float
    body_posture_openness: float
    gesture_frequency: float


def extract_visual_cue_features(frames=None) -> VisualCueFeatures:
    raise NotImplementedError(
        "Visual cue extraction requires video input "
        "- not yet available in this pipeline."
    )


@dataclass
class ExpressionFeatures:
    emotional_expression: dict[str, float]
    microexpression_frowning_frequency: float
    head_nod_frequency: float


def extract_expression_features(frames=None) -> ExpressionFeatures:
    raise NotImplementedError(
        "Facial expression extraction requires video input "
        "- not yet available in this pipeline."
    )


@dataclass
class VoiceToneFeatures:
    speech_rate_wpm: float
    pitch_variation: float
    pause_filler_frequency: float
    tone_consistency: float
    volume_variation: float


def extract_voice_tone_features(audio=None) -> VoiceToneFeatures:
    raise NotImplementedError(
        "Voice tone extraction requires audio input "
        "- not yet available in this pipeline."
    )


# ---------------------------------------------------------
# Eski oturum feature API'si
# ---------------------------------------------------------

@dataclass
class SessionFeatures:
    text: TextFeatures
    visual: VisualCueFeatures | None = None
    expressions: ExpressionFeatures | None = None
    voice: VoiceToneFeatures | None = None


def extract_session_features(
    transcript_text: str,
) -> SessionFeatures:
    turns = parse_transcript(transcript_text)

    return SessionFeatures(
        text=extract_text_features(turns)
    )