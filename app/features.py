import re
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

import spacy

from sentiment import analyze_sentiment
from transcript_parser import Turn, parse_transcript
# Türkçe anahtar kelime ve ifade listeleri.
# NLP, duygu analizi ve okunabilirlik hesapları henüz Türkçeye uyarlanmadı.
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

nlp_instance: spacy.language.Language | None = None

# Metni cümlelere ayırmak için temel bir spaCy NLP yapısı oluşturur.
def nlp() -> spacy.language.Language:
    global nlp_instance
    if nlp_instance is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp_instance = nlp
    return nlp_instance

# Metnin hece sayısını hesaplar.
def count_syllables(word: str) -> int:
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
def filter_turns(turns: Sequence[Turn], speakers: Iterable[str] | None) -> list[Turn]:
    if speakers is None:
        return list(turns)
    wanted = {s.lower() for s in speakers}
    return [t for t in turns if t.speaker.lower() in wanted]

# Konuşma turlarının duygu analizini hesaplar. A konuşmalarının skoru x B konusşmacısının konuşmalarının skoru y gibi. bu skorların ortalamasını verir.
def sentiment_score(turns: Sequence[Turn]) -> float:
    if not turns:
        return 0.0
    scores = [analyze_sentiment(t.text).compound for t in turns]
    return sum(scores) / len(scores)

# Açık uçlu soruların sıklığını hesaplar. Bu açkık uçlu sorular OPEN_QUESTION_STARTERS ile başlayan ve soru işareti ile biten cümlelerdir.
def open_ended_question_frequency(turns: Sequence[Turn]) -> float:
    if not turns:
        return 0.0
    hits = 0
    for turn in turns:
        stripped = turn.text.strip()
        if not stripped.endswith("?"):
            continue
        lowered = stripped.lower()
        if any(lowered.startswith(starter) for starter in OPEN_QUESTION_STARTERS):
            hits += 1
    return hits / len(turns)

# Empati ifadelerinin sayısını hesaplar.
def empathy_marker_count(joined_text: str) -> int:
    lowered = joined_text.lower()
    return sum(lowered.count(phrase) for phrase in EMPATHY_PHRASES)

# Dilin karmaşıklığını hesaplar. Ortalama hece sayısı üzerinden bir ölçüm sağlar. Uzun cümleler için daha yüksek bir karmaşıklık değeri döner bu da toplam skordan çıkartılır

def language_complexity(joined_text: str, nlp: spacy.language.Language) -> float:
    doc = nlp(joined_text)
    words = [tok.text for tok in doc if tok.is_alpha]
    if not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    return syllables / len(words)

# Akıcılık hesabı. Cümle uzunluklarındaki tutarlılık ve bağlaç kullanımına göre bir ölçüm sağlar.
# Akıcılık skoru 1 - (std_dev / avg_len). (std_dev / avg_len) 0 a yakınsa Cümle uzunlukları birbirine çok yakındır ve oldukça düzenlidir.
#(std_dev / avg_len) 1 e yakınlaştıkça cümle uzunlukları arasındaki tutarsızlık artar ve akıcılık düşer.
def syntactic_coherence(joined_text: str, nlp: spacy.language.Language) -> float:
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
        1 for sent in sentences if any(c in sent.text.lower() for c in DISCOURSE_CONNECTORS)
    )
    connector_ratio = connector_hits / len(sentences)

    return (consistency + connector_ratio) / 2

# Anahtar kelimelerin metin içindeki sıklığını hesaplar.
def keyword_frequency(joined_text: str, keywords: Sequence[str]) -> dict[str, int]:
    lowered = joined_text.lower()
    return {keyword: lowered.count(keyword.lower()) for keyword in keywords}

# Metnin okunabilirlik skorunu hesaplar. Flesch-Kincaid formülüne dayalıdır.
def readability_score(joined_text: str, nlp: spacy.language.Language) -> float:
    doc = nlp(joined_text)
    sentences = [sent for sent in doc.sents if sent.text.strip()]
    words = [tok.text for tok in doc if tok.is_alpha]
    if not sentences or not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
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
    subset = filter_turns(turns, speakers)
    joined_text = " ".join(t.text for t in subset)
    nlp_pipeline = nlp()

    return TextFeatures(
        sentiment_score=sentiment_score(subset),
        open_ended_question_frequency=open_ended_question_frequency(subset),
        empathy_marker_count=empathy_marker_count(joined_text),
        language_complexity=language_complexity(joined_text, nlp_pipeline),
        syntactic_coherence=syntactic_coherence(joined_text, nlp_pipeline),
        keyword_frequency=keyword_frequency(joined_text, keywords),
        readability_score=readability_score(joined_text, nlp_pipeline),
    )


# 3 tip ifade daha :
# Evaluative(değerlendirme)
# Directive(yönlendirici) arabulucubu tip ifadeleri kullanarak tarafı bir yöne doğru itiyor
# Participation Invitation(Katılım Daveti)
EVALUATIVE_PHRASES = (
    "haklı", "haksız", "haklısınız", "haksızsınız", "haklısın", "haksızsın",
    "suç sizde", "siz hatalısınız",
)
DIRECTIVE_PHRASES = (
    "kabul etmelisiniz", "kabul etmelisin", "kabul etmek zorundasınız",
    "imzalamalısınız", "başka seçeneğiniz yok",
)
PARTICIPATION_PHRASES = (
    "ne düşünüyorsunuz", "nasıl değerlendiriyorsunuz", "sizin görüşünüz",
    "sizi dinleyelim", "siz ne dersiniz", "anlatır mısınız",
)
#eşit süre, söz hakkı, arabulucunu rolüne dikkat edildi mi, eşit ilgi endeksi, iletişimde mesafeyi korudu mu( duygusal mesafe)
#görsel duruş


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


def normalize_turkish(text: str) -> str:
    return " ".join(text.translate(str.maketrans("Iİ", "ıi")).lower().split())


def contains_phrase(text: str, phrase: str) -> bool:
    # Örneğin "haklı" ifadesini "haklılık" içinde eşleştirme.
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def extract_impartiality_features(
    turns: Sequence[Turn], speakers: Iterable[str]
) -> ImpartialityFeatures:
    """Seçilen arabulucunun dilinden incelemeye açık bulgular çıkarır.

    speakers açıkça verilmelidir; tarafların ifadeleri arabulucuya yazılmaz.
    Diğer konuşmacılar taraf kabul edilir. Söz verme çağrısı ancak cümle başında
    "Taraf A, ..." gibi tek bir açık hitap varsa bir tarafa bağlanır; aksi hâlde
    addressed_party=None kalır. Her tur, kategori ve taraf için en fazla bir
    kez sayılır. Çağrı sayıları konuşma süresi veya eşit muamele ölçümü değildir.

    Alıntı, ironi ve olumsuzlama anlamı çözülmez. Eşleşmeler taraflılık veya baskı
    tespiti olarak değil, tam cümlesiyle uzman incelemesine sunulmalıdır.
    """
    wanted = {normalize_turkish(speaker) for speaker in speakers}
    if not wanted or "" in wanted:
        raise ValueError("En az bir arabulucu konuşmacısı seçilmelidir.")

    parties: dict[str, str] = {}
    selected: list[Turn] = []
    for turn in turns:
        speaker = normalize_turkish(turn.speaker)
        if speaker in wanted:
            if turn.text.strip():
                selected.append(turn)
        elif speaker:
            parties.setdefault(speaker, turn.speaker)

    invitations = {name: 0 for name in parties.values()}
    evidence: list[ImpartialityEvidence] = []
    counts = {"evaluative_language": 0, "directive_language": 0,
              "participation_invitation": 0}
    categories = {
        "evaluative_language": EVALUATIVE_PHRASES,
        "directive_language": DIRECTIVE_PHRASES,
        "participation_invitation": PARTICIPATION_PHRASES,
    }
    for turn in selected:
        normalized = normalize_turkish(turn.text)
        addressed = [
            name for key, name in parties.items()
            if re.search(r"(?:^|[.!?]\s+)" + re.escape(key) + r"\s*[,;:]",
                         normalized)
        ]
        for category, phrases in categories.items():
            matches = [phrase for phrase in phrases
                       if contains_phrase(normalized, phrase)]
            if not matches:
                continue
            counts[category] += 1
            party = None
            if category == "participation_invitation" and len(addressed) == 1:
                party = addressed[0]
                invitations[party] += 1
            for phrase in matches:
                evidence.append(ImpartialityEvidence(
                    turn_index=turn.index,
                    speaker=turn.speaker,
                    text=turn.text,
                    category=category,
                    matched_phrase=phrase,
                    addressed_party=party,
                ))

    return ImpartialityFeatures(
        mediator_turn_count=len(selected),
        evaluative_language_turn_count=counts["evaluative_language"],
        directive_language_turn_count=counts["directive_language"],
        participation_invitation_turn_count=counts["participation_invitation"],
        invitations_by_party=invitations,
        evidence=evidence,
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
