"""Agreement dashboard. Shows only what a real run wrote to runs/<run_id>/."""
import os
from pathlib import Path as _P

os.chdir(_P(__file__).resolve().parents[1])  # paths below are relative to the repo root
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RUNS = Path("runs")
st.set_page_config(page_title="Indic LLM-Judge Agreement", layout="wide")
st.title("Can an LLM judge grade Hindi and Kannada answers like humans do?")

runs = sorted([p for p in RUNS.glob("*") if (p / "summary.json").exists()],
              key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    st.info("No completed runs yet. Run `make eval`, then label with `make label`.")
    st.stop()
run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
s = json.loads((run / "summary.json").read_text("utf-8"))
agree_p = run / "agreement.csv"
agree = pd.read_csv(agree_p) if agree_p.stat().st_size > 1 else pd.DataFrame()

c = st.columns(4)
c[0].metric("Items", s["n_items"])
c[1].metric("Answers judged", s["n_answers"])
c[2].metric("Human A labels", s["n_human_a"])
c[3].metric("Human B labels", s["n_human_b"])

if agree.empty:
    st.warning("No human labels overlap this run yet, so there are no agreement numbers.")
else:
    fmt = lambda v, lo, hi: f"{v:.2f} [{lo:.2f}, {hi:.2f}]"
    view = agree.assign(
        kappa_ci=[fmt(*r) for r in agree[["kappa", "kappa_lo", "kappa_hi"]].values],
        qwk_ci=[fmt(*r) for r in agree[["qwk", "qwk_lo", "qwk_hi"]].values],
        alpha_ci=[fmt(*r) for r in agree[["alpha_ord", "alpha_lo", "alpha_hi"]].values],
    )[["pair", "lang", "criterion", "n", "exact_agree", "kappa_ci", "qwk_ci", "alpha_ci"]]
    st.subheader("Agreement (95% bootstrap CI)")
    pair = st.selectbox("Pair", sorted(view.pair.unique()))
    v = view[view.pair == pair]
    st.markdown("**Overall and per language**")
    st.dataframe(v[v.criterion == "all"], hide_index=True, use_container_width=True)
    st.markdown("**Per language × criterion (quadratic-weighted kappa)**")
    piv = agree[(agree.pair == pair) & (agree.criterion != "all")].pivot(
        index="criterion", columns="lang", values="qwk")
    st.dataframe(piv.round(2),
                 use_container_width=True)
    with st.expander("All rows"):
        st.dataframe(v, hide_index=True, use_container_width=True)

    conf = s.get("confusion", {}).get(pair, {})
    if conf:
        st.subheader("Confusion matrix")
        key = st.selectbox("Language | criterion", sorted(conf))
        a, b = pair.split("~")
        m = pd.DataFrame(conf[key], index=[f"{a}={i}" for i in range(1, 5)],
                         columns=[f"{b}={i}" for i in range(1, 5)])
        st.dataframe(m, use_container_width=False)

st.subheader("Bias checks")
st.json(s["bias"])
st.subheader("Judge cost and latency")
st.json(s["cost_latency"])

st.subheader("10 worst judge–human disagreements")
w_p = run / "worst_disagreements.csv"
w = pd.read_csv(w_p) if w_p.exists() and w_p.stat().st_size > 1 else pd.DataFrame()
if w.empty:
    st.caption("Needs human labels.")
for r in w.itertuples():
    with st.expander(f"{r.item_id} · {r.generator} · {r.lang} · total |diff| = {r.abs_diff}"):
        st.markdown(f"**Question:** {r.question}\n\n**Reference:** {r.reference}\n\n"
                    f"**Answer:** {r.answer}\n\n**Judge rationale:** {r.rationale}")
        st.table(pd.DataFrame({
            "judge": [getattr(r, f"{c}_judge") for c in
                      ("correctness", "faithfulness", "fluency", "completeness")],
            "human": [getattr(r, f"{c}_human") for c in
                      ("correctness", "faithfulness", "fluency", "completeness")]},
            index=["correctness", "faithfulness", "fluency", "completeness"]))
