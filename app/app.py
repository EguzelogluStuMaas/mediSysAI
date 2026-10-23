from pathlib import Path
from hashlib import sha256
from uuid import uuid4

import streamlit as st

from transcript_parser import parse_transcript_document
from features import extract_sentence_features
from scoring import score_sentence_features
from sentiment import analyze_sentiment
from document_reader import DocumentReadError, SUPPORTED_EXTENSIONS, read_document


CATEGORY_LABELS = {
    "empathy": "Empati ifadesi",
    "open_question": "Açık uçlu soru",
    "participation_invitation": "Katılım daveti",
    "evaluative_language": "Değerlendirici ifade",
    "directive_language": "Yönlendirici ifade",
}


st.title("Arabuluculuk Cümle Analizi")

uploaded_file = st.file_uploader(
    "Transkript yükle",
    type=list(SUPPORTED_EXTENSIONS),
    help=(
        "TXT: Konuşmacı: metin başlıkları. "
        "JSON: speakers ve turns listeleri. DOCX doğrudan okunur; "
        "DOC için Windows ve masaüstü Microsoft Word gerekir. En fazla 20 MB."
    ),
)



# Transkripti oku


if uploaded_file is not None:
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
else:
    sample_path = (
        Path(__file__).resolve().parent / "sample_transcript.txt"
    )
    filename = sample_path.name
    try:
        file_bytes = sample_path.read_bytes()
    except OSError:
        st.info("Örnek dosya bulunamadı. Analiz etmek istediğiniz dosyayı yükleyin.")
        st.stop()

    st.caption(
        "Örnek transkript gösteriliyor. "
        "Kendi dosyanızı yükleyebilirsiniz."
    )

# Keep extraction within this user's session. Widget changes must not reopen
# Word, and uploading a different document must reset the editable preview.
input_key = (filename, sha256(file_bytes).hexdigest())
cached_input = st.session_state.get("document_source")
if cached_input is None or cached_input["key"] != input_key:
    try:
        with st.spinner("Dosya okunuyor…"):
            loaded = read_document(filename, file_bytes)
    except DocumentReadError as error:
        st.error(str(error))
        st.stop()
    cached_input = {
        "key": input_key, "input": loaded, "session_id": uuid4().hex,
    }
    st.session_state["document_source"] = cached_input
    st.session_state["word_preview_text"] = loaded.text
    st.session_state["word_analysis_text"] = None

loaded = cached_input["input"]
raw_text = loaded.text
source_format = loaded.parser_format

if loaded.source_format in {"doc", "docx"}:
    st.subheader("Word metnini kontrol et")
    for warning in loaded.warnings:
        st.warning(warning)
    st.caption(
        "Konuşmaları 'Arabulucu: ...', 'Taraf A: ...' biçiminde düzenleyin. "
        "Konuşmacısı belli olmayan ifadeleri tahminle etiketlemeyin. "
        "Analize almak istemediğiniz başlık ve senaryo açıklamalarını silin."
    )
    with st.form("word_preview"):
        edited_text = st.text_area(
            "Analize gönderilecek metin", key="word_preview_text", height=320,
        )
        submitted = st.form_submit_button("Bu metni analiz et")
    if submitted:
        st.session_state["word_analysis_text"] = edited_text
    if st.session_state["word_analysis_text"] is None:
        st.info("Metni kontrol edip 'Bu metni analiz et' düğmesine basın.")
        st.stop()
    raw_text = st.session_state["word_analysis_text"]

if not raw_text.strip():
    st.info("Analiz edilecek metin boş.")
    st.stop()

try:
    transcript = parse_transcript_document(
        raw_text,
        format=source_format,
        session_id=(cached_input["session_id"] if source_format == "txt" else None),
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

for issue in transcript.issues:
    st.warning(issue.message)

if not transcript.turns:
    st.info("Analiz edilecek konuşma bulunamadı.")
    st.stop()


# ---------------------------------------------------------
# Analiz seçenekleri
# ---------------------------------------------------------

all_speakers = list(transcript.speakers)

default_speakers = [
    speaker_id
    for speaker_id, speaker in transcript.speakers.items()
    if speaker.role == "mediator"
]

selected_ids = st.multiselect(
    "Analiz edilecek konuşmacılar",
    options=all_speakers,
    default=default_speakers,
    format_func=lambda speaker_id: (
        f"{transcript.speakers[speaker_id].display_name} "
        f"({speaker_id})"
    ),
)

include_unknown = st.checkbox(
    "Konuşmacısı bilinmeyen cümleleri dahil et",
    value=False,
)

use_sentiment = st.checkbox(
    "Deneysel VADER sentiment analizini çalıştır",
    value=False,
)

st.caption(
    "Türkçe ifade kuralları kullanılır. Alıntı, ironi ve "
    "olumsuzlama bağlamı henüz çözülmez; eşleşmeleri cümleleriyle "
    "birlikte inceleyin. VADER bu projede Türkçe için doğrulanmadı; "
    "etkinleştirilirse ilk kullanımda sözlük indirebilir."
)


# ---------------------------------------------------------
# Parser çıktısından cümleleri seç
# ---------------------------------------------------------

sentences = [
    sentence
    for sentence in transcript.sentences()
    if (
        sentence.speaker_id in selected_ids
        or (
            include_unknown
            and sentence.speaker_id is None
        )
    )
]

if not sentences:
    st.info(
        "Analiz için en az bir konuşmacı veya "
        "bilinmeyen cümleler seçeneğini seçin."
    )
    st.stop()


# ---------------------------------------------------------
# Cümle bazlı özellik çıkarımı ve skorlama
# ---------------------------------------------------------

try:
    features = extract_sentence_features(
        sentences,
        sentiment_analyzer=(
            analyze_sentiment if use_sentiment else None
        ),
    )
except (ImportError, LookupError, OSError) as error:
    st.warning(
        "Sentiment çalıştırılamadı; ifade analizi devam ediyor. "
        f"{error}"
    )
    features = extract_sentence_features(sentences)

result = score_sentence_features(features)
summary = result.overall


# ---------------------------------------------------------
# Genel toplamlar
# ---------------------------------------------------------

left, middle, right = st.columns(3)

left.metric(
    "Analiz edilen cümle",
    summary.sentence_count,
)

middle.metric(
    "Destekleyici ifade puanı",
    summary.support_points,
)

right.metric(
    "İncelenecek ifade puanı",
    summary.review_points,
)

st.caption(
    "Her kategori bir cümlede en fazla 1 puan alır. "
    "Destekleyici toplam: empati + açık uçlu soru + katılım daveti. "
    "İncelenecek toplam: değerlendirici + yönlendirici ifade. "
    "Bunlar tarafsızlık veya başarı notu değildir; "
    "toplamlar transkript uzunluğuyla artabilir."
)


# ---------------------------------------------------------
# Kategori bazında ayrı sonuçlar
# ---------------------------------------------------------

st.subheader("Kategori sonuçları")

st.dataframe(
    [
        {
            "Kategori": label,
            "Eşleşen cümle": summary.category_totals[key],
            "Cümlelerin yüzdesi": (
                summary.category_percentages[key]
            ),
        }
        for key, label in CATEGORY_LABELS.items()
    ],
    hide_index=True,
)


# ---------------------------------------------------------
# Konuşmacı bazında ayrı sonuçlar
# ---------------------------------------------------------

st.subheader("Konuşmacı bazında sonuçlar")

st.dataframe(
    [
        {
            "Konuşmacı": speaker.speaker,
            "Konuşmacı ID": speaker.speaker_id,
            "Cümle": speaker.summary.sentence_count,
            "Destekleyici puan": speaker.summary.support_points,
            "İncelenecek puan": speaker.summary.review_points,
            **{
                label: speaker.summary.category_totals[key]
                for key, label in CATEGORY_LABELS.items()
            },
            "Sentiment ortalaması": (
                speaker.summary.sentiment_mean.compound
                if speaker.summary.sentiment_mean is not None
                else None
            ),
        }
        for speaker in result.by_speaker
    ],
    hide_index=True,
)


# ---------------------------------------------------------
# Sentiment: iletişim puanlarından ayrı
# ---------------------------------------------------------

if summary.sentiment_mean is not None:
    st.subheader(
        "Sentiment — deneysel, ifade puanlarından ayrı"
    )

    st.write(
        {
            "Analiz edilen cümle": summary.sentiment_count,
            "Compound ortalaması": (
                summary.sentiment_mean.compound
            ),
            "Compound toplamı": summary.sentiment_compound_sum,
            "Olumlu ortalaması": summary.sentiment_mean.positive,
            "Nötr ortalaması": summary.sentiment_mean.neutral,
            "Olumsuz ortalaması": summary.sentiment_mean.negative,
        }
    )


# ---------------------------------------------------------
# Her cümlenin sonucu ve eşleşen ifadeler
# ---------------------------------------------------------

st.subheader("Cümleler ve eşleşme kanıtları")

st.dataframe(
    [
        {
            "Sıra": row.features.sentence.index + 1,
            "Cümle ID": row.features.sentence.id,
            "Tur ID": row.features.sentence.turn_id,
            "Konuşmacı": row.features.sentence.speaker,
            "Metin": row.features.sentence.original_text,
            **{
                label: row.scores[key]
                for key, label in CATEGORY_LABELS.items()
            },
            "Destekleyici puan": row.support_points,
            "İncelenecek puan": row.review_points,
            "Eşleşmeler": "; ".join(
                f"{CATEGORY_LABELS[key]}: {', '.join(matches)}"
                for key, matches in row.features.matches.items()
                if matches
            ),
            "Sentiment": (
                row.features.sentiment.compound
                if row.features.sentiment is not None
                else None
            ),
        }
        for row in result.sentences
    ],
    hide_index=True,
)


# ---------------------------------------------------------
# Bilgi amaçlı çözüm kelimeleri
# ---------------------------------------------------------

with st.expander("Çözüm ifadeleri — bilgi amaçlı"):
    st.write(summary.resolution_keyword_sentence_counts)

    st.caption(
        "Her kelimenin kaç cümlede geçtiğini gösterir; "
        "anlaşma sağlandığını göstermez."
    )
