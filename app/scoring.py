from dataclasses import dataclass
from typing import Iterable, Sequence

from features import (
    empathy_marker_count,
    filter_turns,
    open_ended_question_frequency,
    extract_impartiality_features
)
from transcript_parser import Turn


# Her ölçüt %10: altı ölçütün toplam ağırlığı %60, en yüksek toplam puan 60.
DEFAULT_WEIGHTS: dict[str, float] = {
    "questioning": 0.10,
    "active_listening": 0.10,
    "empathy": 0.10,
    "impartiality": 0.10,
    "reframing": 0.10,
    "process_management": 0.10,
}


def _scorable_turns(
    turns: Sequence[Turn], speakers: Iterable[str] | None
) -> list[Turn]:
    return [turn for turn in filter_turns(turns, speakers) if turn.text.strip()]


def questioning_score(
    turns: Sequence[Turn], speakers: Iterable[str] | None = None
) -> float | None:
    """Açık uçlu soru olarak eşleşen konuşma turlarının yüzdesi (0–100).

    Soru kalitesini ölçmez; features.py içindeki başlangıç ve soru işareti
    kontrolünü kullanır. Konuşma turunun ortasındaki soruları kaçırabilir.
    """
    selected = _scorable_turns(turns, speakers)
    if not selected:
        return None
    return 100.0 * open_ended_question_frequency(selected)


def active_listening_score(
    turns: Sequence[Turn], speakers: Iterable[str] | None = None
) -> float | None:
    """Aktif dinleme ve özetleme: henüz hesaplanamıyor.

    Tarafın ifadesi ile arabulucunun özeti arasında anlam karşılaştırması gerekir.
    Empati ifadesi sayısı veya cümle tutarlılığı bu ölçütün yerine geçmez.
    """
    return None


def empathy_score(
    turns: Sequence[Turn], speakers: Iterable[str] | None = None
) -> float | None:
    """En az bir empati ifadesi içeren konuşma turlarının yüzdesi (0–100).

    Aynı turdaki tekrarlar puanı artırmaz. Bu yalnızca ifade kullanımını ölçer;
    ifadenin bağlama uygunluğunu veya duygunun doğru anlaşıldığını ölçmez.
    """
    selected = _scorable_turns(turns, speakers)
    if not selected:
        return None
    matching_turns = sum(empathy_marker_count(turn.text) > 0 for turn in selected)
    return 100.0 * matching_turns / len(selected)

# Tarafsızlık skoru: arabulucunun tarafsız yaklaşımını ölçer. PARTICIPATION_INVITATION, EMPATHY ve open ended questionların sıklıkları bulunur. EVALUATIVE VE DIRECTIVE sıklıkları bulunur
def impartiality_score(
    turns: Sequence[Turn], speakers: Iterable[str] | None = None
 ) -> float | None:
    selected = _scorable_turns(turns, speakers)
    if not selected:
        return None

    features = extract_impartiality_features(
        selected, speakers={turn.speaker for turn in selected}
    )
    turn_count = len(selected)
    participation_frequency = features.participation_invitation_turn_count / turn_count
    empathy_frequency = sum(
        empathy_marker_count(turn.text) > 0 for turn in selected
    ) / turn_count
    question_frequency =    open_ended_question_frequency(selected)
    evaluative_frequency = features.evaluative_language_turn_count / turn_count
    directive_frequency = features.directive_language_turn_count / turn_count

    if not any((participation_frequency, empathy_frequency, question_frequency,
                evaluative_frequency, directive_frequency)):
        return None

    positive_frequency = (
        participation_frequency + empathy_frequency + question_frequency
    ) / 3.0
    negative_frequency = (evaluative_frequency + directive_frequency) / 2.0
    return max(0.0, min(100.0, 50.0 + 50.0 * (positive_frequency - negative_frequency)))

def reframing_score(
    turns: Sequence[Turn], speakers: Iterable[str] | None = None
) -> float | None:
    """Yapıcı yeniden ifade etme: henüz hesaplanamıyor.

    Tarafın suçlayıcı ifadesi ile arabulucunun bunu ihtiyaç veya çözülebilir
    sorun olarak yeniden ifadesi karşılaştırılmalı. Mevcut özelliklerde yok.
    """
    return None


def process_management_score(
    turns: Sequence[Turn], speakers: Iterable[str] | None = None
) -> float | None:
    """Süreç ve seçenek geliştirme: henüz hesaplanamıyor.

    Konuları netleştirme, seçenekleri araştırma ve sonraki adımları belirleme
    davranışları gerekir. Çözüm anahtar kelimelerinin sayısı tek başına yetmez.
    """
    return None


@dataclass
class ScoringResult:
    scores: dict[str, float | None]
    weights: dict[str, float]
    total_score: float | None

    @property
    def available_weight(self) -> float:
        """Hesaplanabilen ölçütlerin toplam ağırlığı (örneğin 0.20)."""
        return sum(self.weights[name] for name, score in self.scores.items()
                   if score is not None)


def scoring(
    turns: Sequence[Turn], speakers: Iterable[str] | None = None
) -> ScoringResult:
    """Ölçüt puanlarını hesaplar ve tamamı mevcutsa ağırlıklı toplar.

    speakers=None bütün konuşmacıları kapsar; arabulucu değerlendirmesinde
    speakers=["Arabulucu"] verilmelidir. Eksik puan sıfır sayılmaz ve mevcut
    ölçütlerin ağırlıkları otomatik olarak artırılmaz.
    """
    # Bir kez tüketilebilen iterable'ları her ölçütte yeniden kullanabilmek için.
    selected_speakers = tuple(speakers) if speakers is not None else None
    scores = {
        "questioning": questioning_score(turns, selected_speakers),
        "active_listening": active_listening_score(turns, selected_speakers),
        "empathy": empathy_score(turns, selected_speakers),
        "impartiality": impartiality_score(turns, selected_speakers),
        "reframing": reframing_score(turns, selected_speakers),
        "process_management": process_management_score(turns, selected_speakers),
    }
    weights = DEFAULT_WEIGHTS.copy()
    total_score = None
    if all(score is not None for score in scores.values()):
        total_score = sum(
            score * weights[name]
            for name, score in scores.items()
            if score is not None
        )
    return ScoringResult(scores=scores, weights=weights, total_score=total_score)

