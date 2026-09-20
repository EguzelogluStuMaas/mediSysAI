from dataclasses import dataclass

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_analyzer: SentimentIntensityAnalyzer | None = None


@dataclass
class SentimentResult:
    compound: float
    positive: float
    neutral: float
    negative: float


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        try:
            _analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon")
            _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def analyze_sentiment(text: str) -> SentimentResult:
    scores = _get_analyzer().polarity_scores(text)
    return SentimentResult(
        compound=scores["compound"],
        positive=scores["pos"],
        neutral=scores["neu"],
        negative=scores["neg"],
    )
