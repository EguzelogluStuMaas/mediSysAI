"""Turkish text preparation with separate model and optional feature inputs.

No stemming, automatic spelling correction, or model tokenizer replacement.
The model text retains case, punctuation, negation and discourse markers.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


# A deliberately small configurable list, not a universal Turkish vocabulary.
DEFAULT_STOPWORDS = frozenset({"bir", "bu", "şu", "ve", "ile", "için", "da", "de", "ki"})
PROTECTED_WORDS = frozenset({
    "değil", "yok", "hayır", "asla", "hiç", "hiçbir", "evet",
    "ama", "ancak", "fakat", "çünkü", "mı", "mi", "mu", "mü",
    "ne", "nasıl", "neden", "hangi", "kim", "ben", "sen", "siz", "biz",
    "o", "onlar", "sizi", "bizi", "bana", "size",
})
TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def turkish_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(str.maketrans("Iİ", "ıi")).lower()


def normalize_text(text: str) -> str:
    """Normalize Unicode and spacing while preserving language signals."""
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class PreprocessedText:
    original_text: str
    model_text: str
    tokens: tuple[str, ...]
    filtered_tokens: tuple[str, ...]
    stopwords_removed: bool

    @property
    def feature_text(self) -> str:
        """Optional bag-of-words input; do not substitute for model_text."""
        return " ".join(self.filtered_tokens)


def preprocess_text(
    text: str, *, remove_stopwords: bool = False,
    stopwords: Iterable[str] | None = None,
) -> PreprocessedText:
    model_text = normalize_text(text)
    tokens = tuple(TOKEN_PATTERN.findall(turkish_lower(model_text)))
    if isinstance(stopwords, str):
        raise ValueError("stopwords must be a collection of words, not one string.")
    if remove_stopwords:
        supplied = DEFAULT_STOPWORDS if stopwords is None else stopwords
        excluded = {turkish_lower(word) for word in supplied} - PROTECTED_WORDS
        filtered = tuple(token for token in tokens if token not in excluded)
    else:
        filtered = tokens
    return PreprocessedText(text, model_text, tokens, filtered, remove_stopwords)
