import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


from scoring import compute_scores
from sentiment import score_turns
from transcript_parser import parse_transcript

SAMPLE_PATH = Path(__file__).parent / "sample_transcript.txt"

st.set_page_config(page_title="Mediation Transcript Grader", layout="centered")
st.title("Mediation Transcript Grader")
st.caption(
    "Demo: grades a mediator's performance from a plain-text transcript using "
    "VADER sentiment analysis across 10 equally weighted features."
)

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload a .txt transcript", type=["txt"])
    if uploaded is not None:
        st.session_state.raw_text = uploaded.read().decode("utf-8")
with col2:
    st.write("")
    st.write("")
    if st.button("Load sample transcript"):
        st.session_state.raw_text = SAMPLE_PATH.read_text(encoding="utf-8")

raw_text = st.text_area(
    "Or paste transcript text (one turn per line: 'Speaker: text')",
    value=st.session_state.raw_text,
    height=250,
    key="raw_text",
)


def render_feature_chart(feature_scores: dict[str, float]) -> go.Figure:
    items = sorted(feature_scores.items(), key=lambda kv: kv[1])
    labels = [name for name, _ in items]
    values = [score for _, score in items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#2a78d6",
            text=[f"{v:.0f}" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Feature scores (0-100)",
        xaxis=dict(range=[0, 105], gridcolor="#e1e0d9", zeroline=False),
        yaxis=dict(gridcolor="#e1e0d9"),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        showlegend=False,
        margin=dict(l=10, r=40, t=40, b=10),
        bargap=0.35,
    )
    return fig


if st.button("Grade transcript", type="primary"):
    if not raw_text.strip():
        st.warning("Paste a transcript or upload a file first.")
    else:
        try:
            turns = parse_transcript(raw_text)
            scored = score_turns(turns)
            result = compute_scores(scored)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.metric("Overall score", f"{result.final_score:.1f} / 100")

            st.plotly_chart(render_feature_chart(result.feature_scores), use_container_width=True)

            table = pd.DataFrame(
                {
                    "Feature": list(result.feature_scores.keys()),
                    "Score": list(result.feature_scores.values()),
                    "Weight": [result.weights[name] for name in result.feature_scores],
                }
            )
            table["Contribution"] = table["Score"] * table["Weight"]
            st.dataframe(
                table.set_index("Feature").style.format(
                    {"Score": "{:.1f}", "Weight": "{:.0%}", "Contribution": "{:.1f}"}
                ),
                use_container_width=True,
            )

            with st.expander("Parsed turns & sentiment"):
                turns_df = pd.DataFrame(
                    {
                        "Speaker": [t.speaker for t in scored],
                        "Role": [t.role for t in scored],
                        "Text": [t.text for t in scored],
                        "Compound": [t.compound for t in scored],
                    }
                )
                st.dataframe(turns_df, use_container_width=True)
