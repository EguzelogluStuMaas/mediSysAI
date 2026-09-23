import os
import sys
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text
from transcript_parser import parse_transcript
from features import extract_text_features, TextFeatures
from sentiment import analyze_sentiment

# Load sample transcript
sample_transcript_path = Path(__file__).resolve().parent / "sample_transcript.txt"
with open(sample_transcript_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Parse transcript into turns
turns = parse_transcript(raw_text)

# Extract text features
text_features = extract_text_features(turns)

# Display results in Streamlit
st.title("Mediator Score Dashboard")
st.write("### Sample Transcript Analysis")

# Display sentiment score
st.metric(
    label="Overall Sentiment Score",
    value=f"{text_features.sentiment_score:.2f}",
    delta="Positive" if text_features.sentiment_score > 0 else "Negative"
)

# Optional: Display additional metrics
st.subheader("Key Metrics")
st.write(f"- **Open-ended questions**: {text_features.open_ended_question_frequency:.2f}")
st.write(f"- **Empathy markers**: {text_features.empathy_marker_count}")
st.write(f"- **Readability score**: {text_features.readability_score:.2f}")
