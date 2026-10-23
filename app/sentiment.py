from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer: SentimentIntensityAnalyzer | None = None


@dataclass
class SentimentResult:
    compound: float
    positive: float
    neutral: float
    negative: float 


def get_analyzer() -> SentimentIntensityAnalyzer:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    global analyzer
    if analyzer is None:
        try:
            analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon")
            analyzer = SentimentIntensityAnalyzer()
    return analyzer


def analyze_sentiment(text: str) -> SentimentResult:
    scores = get_analyzer().polarity_scores(text)
    return SentimentResult(
        compound=scores["compound"],
        positive=scores["pos"],
        neutral=scores["neu"],
        negative=scores["neg"],
    )
