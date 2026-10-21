from pathlib import Path
import streamlit as st
from transcript_parser import parse_transcript
from features import extract_text_features
from scoring import impartiality_score

st.title("Mediator Score Dashboard")
uploaded_file = st.file_uploader(
    "Upload a transcript",
    type=["txt"],
    help=(
        "Use a UTF-8 text file with Speaker: text headers; speech may span multiple lines. "
        "Label the mediator Arabulucu or Mediator for impartiality scoring."
    ),
)

if uploaded_file is not None:
    try:
        raw_text = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("Could not read this file. Save it as UTF-8 text and upload it again.")
        st.stop()
    st.subheader("Uploaded Transcript Analysis")
    st.caption(f"File: {uploaded_file.name}")
else:
    sample_transcript_path = Path(__file__).resolve().parent / "sample_transcript.txt"
    raw_text = sample_transcript_path.read_text(encoding="utf-8")
    st.subheader("Sample Transcript Analysis")
    st.caption("Upload a .txt file to analyze your own transcript.")

if not raw_text.strip():
    st.error("The transcript is empty. Upload a file containing Speaker: text lines.")
    st.stop()

# Parse transcript into turns
try:
    turns = parse_transcript(raw_text)
except ValueError as error:
    st.error(str(error))
    st.stop()
if not turns:
    st.error("No conversation turns found. Use one Speaker: text turn per line.")
    st.stop()

# Extract text features
text_features = extract_text_features(turns)
impartiality_score_value = impartiality_score(
    turns, speakers=["Arabulucu", "Mediator"]
)

# Display sentiment score
st.metric(
    label="Overall Sentiment Score",
    value=f"{text_features.sentiment_score:.2f}",
    delta="Positive" if text_features.sentiment_score > 0 else "Negative"
)
st.metric(
    label="Mediator Impartiality Score",
    value=(
        f"{impartiality_score_value:.2f} / 100"
        if impartiality_score_value is not None else "N/A"
    ),
    help=(
        "Uses only Arabulucu / Mediator turns. Participation invitations, empathy "
        "and open-ended questions increase the score; evaluative and directive "
        "language decrease it. Higher scores indicate more impartiality signals."
    ),
)
if impartiality_score_value is None:
    st.caption("Not enough evidence: no mediator turns or matching language signals.")
else:
    st.caption("Rule-based language indicator using Turkish phrases; not a validated measure of impartiality.")


# Optional: Display additional metrics
st.subheader("Key Metrics")
st.write(f"- **Open-ended questions**: {text_features.open_ended_question_frequency:.2f}")
st.write(f"- **Empathy markers**: {text_features.empathy_marker_count}")
st.write(f"- **Readability score**: {text_features.readability_score:.2f}")
