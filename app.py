"""
app.py — Streamlit UI for the Hasamex AI Engineer case study.

Run with:  streamlit run app.py
(see README.md for full setup instructions)
"""

import os
import glob

import streamlit as st
from dotenv import load_dotenv

from transcript_parser import load_transcripts, all_segments
from interview_guide import load_questions
from retrieval import TfidfIndex
from llm_engine import (
    answer_question_for_expert,
    answer_all_questions_for_expert,
    cross_transcript_themes,
    ask_freeform_question,
    Citation,
)

load_dotenv()

st.set_page_config(page_title="Expert Call Analyzer — Hasamex Case Study", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@st.cache_resource
def load_everything():
    transcript_paths = sorted(glob.glob(os.path.join(DATA_DIR, "Transcript_*.txt")))
    transcripts = load_transcripts(transcript_paths)
    segments = all_segments(transcripts)
    questions = load_questions(os.path.join(DATA_DIR, "Interview_Guide.txt"))
    index = TfidfIndex(segments)
    return transcripts, segments, questions, index


transcripts, segments, questions, index = load_everything()
experts = [t.expert for t in transcripts]


def render_citation(c: Citation):
    badge = "✅ verified" if c.verified else "⚠️ could not verify against source"
    st.markdown(
        f"> \"{c.quote}\"  \n"
        f"**{c.expert}** · `{c.source_file}` · timestamp `{c.timestamp}` · {badge}"
    )


st.title("🩺 Expert Call Analyzer")
st.caption(
    "European Robotic Surgery Market — analyzes 3 expert-call transcripts against "
    "the interview guide, extracts cited quotes, and surfaces cross-expert themes "
    "and disagreements. Every claim below is grounded in a retrieved transcript "
    "segment and quote-checked before display."
)

if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "No GEMINI_API_KEY found. Add your Gemini API key to `.env` and restart the app."
    )
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Per-Expert Answers", "🔍 Common Themes & Disagreements", "💬 Ask a Question", "📄 Raw Transcripts"]
)

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.subheader("Answer the interview guide, per expert")

    selected_expert = st.selectbox("Choose an expert", experts)

    if st.button("Generate answers for this expert", type="primary"):

        expert_segments = index.all_for_expert(selected_expert)

        with st.spinner("Answering all 6 questions from transcript..."):
            try:
                results = answer_all_questions_for_expert(
                    questions,
                    expert_segments
                )

                for q, result in zip(questions, results):

                    with st.expander(q, expanded=True):

                        if result.not_mentioned:
                            st.info("Not addressed in this transcript.")
                        else:
                            st.write(result.answer_text)

                        for c in result.citations:
                            render_citation(c)

            except Exception as e:

                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.warning(
                        "Gemini free-tier quota is currently exhausted. "
                        "The optimized version now uses only one API call "
                        "for all 6 questions."
                    )

                elif "503" in str(e) or "UNAVAILABLE" in str(e):
                    st.warning(
                        "Gemini is temporarily busy. "
                        "Please try again later."
                    )

                else:
                    st.error(f"Could not generate answers: {e}")
# ---------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("Common themes and disagreements across all 3 experts")
    if st.button("Run cross-transcript analysis", type="primary"):
        by_expert = {t.expert: t.segments for t in transcripts}
        with st.spinner("Comparing experts..."):
            result = cross_transcript_themes(by_expert)

        st.markdown("### 🟢 Common themes")
        for theme in result.get("themes", []):
            st.markdown(f"**{theme['summary']}**")
            for c in theme.get("citations", []):
                badge = "✅" if c.get("verified") else "⚠️"
                st.markdown(f"> \"{c['quote']}\" — **{c['expert']}**, `{c['timestamp']}` {badge}")
            st.divider()

        st.markdown("### 🔴 Disagreements")
        for dis in result.get("disagreements", []):
            st.markdown(f"**{dis['summary']}**")
            for c in dis.get("citations", []):
                badge = "✅" if c.get("verified") else "⚠️"
                st.markdown(f"> \"{c['quote']}\" — **{c['expert']}**, `{c['timestamp']}` {badge}")
            st.divider()

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Ask a question across all transcripts")
    user_q = st.text_input("Your question", placeholder="e.g. Do all experts agree ROI is the top barrier?")
    if st.button("Ask") and user_q.strip():
        with st.spinner("Searching transcripts and answering..."):
            relevant = index.search(user_q, top_k=8)
            result = ask_freeform_question(user_q, relevant)
        if result.not_mentioned:
            st.info("None of the transcripts address this.")
        else:
            st.write(result.answer_text)
        for c in result.citations:
            render_citation(c)

# ---------------------------------------------------------------- Tab 4
with tab4:
    st.subheader("Source transcripts (for reference / manual spot-checking)")
    for t in transcripts:
        with st.expander(f"{t.expert} — {t.role}, {t.market} ({t.source_file})"):
            for seg in t.segments:
                st.markdown(f"**[{seg.timestamp}]** Q: {seg.question}")
                st.markdown(f"A: {seg.answer}")
                st.markdown("---")
