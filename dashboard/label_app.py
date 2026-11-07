"""Blind labeling tool: you never see the judge's scores while labeling.

make label    (then pick labeler A or B in the sidebar)
Labeler B only sees answers that A already labeled, up to the overlap limit.
"""
import os
from pathlib import Path as _P

os.chdir(_P(__file__).resolve().parents[1])  # paths below are relative to the repo root
from pathlib import Path

import pandas as pd
import streamlit as st

from indic_judge import CRITERIA
from indic_judge.label import queue, save_label

RUBRIC = Path("prompts/rubric.md").read_text("utf-8")
FILES = {"A": "labels/human_a.jsonl", "B": "labels/human_b.jsonl"}

st.set_page_config(page_title="Label answers", layout="wide")
runs = sorted([p for p in Path("runs").glob("*") if (p / "answers.csv").exists()],
              key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    st.info("Run `make eval` (or `make full`) first so there are answers to label.")
    st.stop()
run = st.sidebar.selectbox("Answers from run", runs, format_func=lambda p: p.name)
who = st.sidebar.radio("Labeler", ["A", "B"])
overlap = st.sidebar.number_input("B overlap size", 10, 200, 50) if who == "B" else None

answers = pd.read_csv(run / "answers.csv", dtype={"item_id": str})
items = pd.read_json(run / "items.jsonl", lines=True, dtype={"item_id": str}).set_index("item_id")
q = queue(answers, FILES[who], restrict_to=FILES["A"] if who == "B" else None, limit=overlap)
st.sidebar.write(f"Remaining in queue: **{len(q)}**")
with st.sidebar.expander("Rubric"):
    st.markdown(RUBRIC)
if q.empty:
    st.success("Queue empty. Run `python -m indic_judge.cli analyze runs/<run_id>`.")
    st.stop()

a = q.iloc[0]
it = items.loc[a.item_id]
st.caption(f"{a.item_id} · lang={it.lang}")
st.markdown(f"#### {it.question}")
c1, c2 = st.columns(2)
c1.markdown(f"**Reference answer**\n\n{it.reference}")
c2.markdown(f"**Candidate answer**\n\n{a.answer}")
with st.expander("Passage", expanded=False):
    st.write(it.context)

with st.form(f"f-{a.item_id}-{a.generator}", clear_on_submit=True):
    cols = st.columns(4)
    scores = {c: cols[i].radio(c, [1, 2, 3, 4], index=None, horizontal=True)
              for i, c in enumerate(CRITERIA)}
    note = st.text_input("Note (optional)")
    if st.form_submit_button("Save and next"):
        if any(v is None for v in scores.values()):
            st.error("Score all four criteria.")
        else:
            save_label(FILES[who], who, a.item_id, a.generator, scores, note)
            st.rerun()
