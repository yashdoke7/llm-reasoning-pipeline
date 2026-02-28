"""
dashboard/app.py
Streamlit demo application with two modes:
  1. LIVE EVAL: Enter a reasoning problem, pick a model, watch step-by-step evaluation
  2. RESULTS:   Browse saved evaluation results and charts from W&B

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import streamlit as st
import yaml

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Reasoning Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load config ───────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open(_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)

cfg = load_config()

# ── Init components (cached) ──────────────────────────────────
@st.cache_resource
def get_components():
    from models.groq_client import get_client
    from eval.task_decomposer import TaskDecomposer
    from eval.step_evaluator import StepEvaluator
    from eval.hallucination_scorer import HallucinationScorer
    from eval.drift_detector import DriftDetector
    from mitigation.retriever import WikipediaRetriever
    from mitigation.regrounder import Regrounder, InterventionMode

    if not os.environ.get("GROQ_API_KEY"):
        return None, None, None, None, None, None, None

    client = get_client(cfg)
    decomposer = TaskDecomposer()
    step_evaluator = StepEvaluator(client=client, evaluator_model=cfg["models"]["evaluator_model"])
    hall_scorer = HallucinationScorer(client=client, model=cfg["models"]["evaluator_model"])
    drift_detector = DriftDetector(client=client, model=cfg["models"]["evaluator_model"])
    retriever = WikipediaRetriever()
    regrounder = Regrounder(client=client, retriever=retriever)
    return client, decomposer, step_evaluator, hall_scorer, drift_detector, retriever, regrounder


# ── Styling ───────────────────────────────────────────────────
VERDICT_COLORS = {"VALID": "#2ECC71", "INVALID": "#E74C3C", "UNCERTAIN": "#F39C12"}
VERDICT_ICONS = {"VALID": "✅", "INVALID": "❌", "UNCERTAIN": "⚠️"}


def render_step_card(step_eval: dict) -> None:
    verdict = step_eval.get("verdict", "UNCERTAIN")
    color = VERDICT_COLORS.get(verdict, "#AAAAAA")
    icon = VERDICT_ICONS.get(verdict, "❓")
    conf = step_eval.get("confidence", 0.0)
    error_type = step_eval.get("error_type") or ""

    with st.container():
        col1, col2 = st.columns([0.07, 0.93])
        with col1:
            st.markdown(f"<div style='font-size:24px;text-align:center'>{icon}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 12px; margin-bottom: 8px;'>"
                f"<b>Step {step_eval['step_index']}</b> — "
                f"<span style='color:{color};font-weight:bold'>{verdict}</span> "
                f"(conf: {conf:.2f})"
                f"{'  |  <i>' + error_type + '</i>' if error_type else ''}"
                f"<br><small style='color:#888'>{step_eval.get('explanation','')}</small>"
                f"<br><span style='color:#333'>{step_eval.get('step_text','')[:300]}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🧠 LLM Reasoning Evaluator")
st.sidebar.markdown("Step-level evaluation of LLM reasoning chains")
st.sidebar.divider()

mode = st.sidebar.radio("Mode", ["Live Evaluation", "Browse Results", "About"])

model_options = {m["display_name"]: m["id"] for m in cfg["models"]["eval_models"]}
selected_display = st.sidebar.selectbox("Model", list(model_options.keys()))
selected_model_id = model_options[selected_display]

category_options = cfg["eval"]["categories"]
selected_category = st.sidebar.selectbox("Task Category", category_options)

# ── Main content ──────────────────────────────────────────────

if mode == "Live Evaluation":
    st.title("Live Step-Level Reasoning Evaluation")
    st.markdown(
        "Enter a reasoning problem below. The system will generate a chain-of-thought "
        "reasoning trace and evaluate each step for logical validity, hallucination, "
        "and consistency."
    )

    # Check API key
    if not os.environ.get("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY not set. Run:\n```\nexport GROQ_API_KEY=your_key\n"
            "streamlit run dashboard/app.py\n```"
        )
        st.stop()

    client, decomposer, step_evaluator, hall_scorer, drift_detector, retriever, regrounder = get_components()

    # Example problems
    EXAMPLES = {
        "Math (Arithmetic)": "A store sells apples for $0.75 each and oranges for $1.20 each. Sarah buys 4 apples and 3 oranges. She pays with a $10 bill. How much change does she receive?",
        "Math (Multi-step)": "A train travels at 80 km/h for 2 hours, then increases speed to 120 km/h for 1.5 hours. A second train starts at the same location 1 hour later and travels at 100 km/h in the same direction. How far has the second train traveled when the first train is exactly 50 km ahead of it?",
        "Causal Reasoning": "If gravity on Earth were half of its current strength, how would the height of the atmosphere change, and what would be the effect on weather patterns?",
        "Factual": "According to the context: The speed of light is 299,792 km/s. Light takes 8.3 minutes to travel from the Sun to Earth. The distance from Earth to the Moon is 384,400 km. How long does it take light to travel from the Sun to the Moon, assuming they are on the same side of Earth?",
    }
    example_choice = st.selectbox("Load an example problem", ["(custom)"] + list(EXAMPLES.keys()))
    default_problem = EXAMPLES.get(example_choice, "")

    problem_input = st.text_area(
        "Problem",
        value=default_problem,
        height=120,
        placeholder="Enter a reasoning problem here...",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    run_mitigation = col2.checkbox("Run RAG re-grounding on failures", value=True)
    run_btn = col1.button("▶ Evaluate", type="primary", use_container_width=True)

    if run_btn and problem_input.strip():
        from eval.task_decomposer import TaskDecomposer
        from mitigation.regrounder import InterventionMode

        with st.spinner(f"Generating reasoning trace with {selected_display}..."):
            # Build category-specific kwargs (avoid passing None to formatters)
            task_fields = {}
            if selected_category == "multistep_arithmetic":
                task_fields = {"question": problem_input.strip()}
            elif selected_category == "factual_consistency":
                task_fields = {"context": problem_input.strip(), "questions": [problem_input.strip()]}
            elif selected_category == "causal_counterfactual":
                task_fields = {"premise": problem_input.strip(), "question": "What would follow from this premise?"}
            elif selected_category == "tool_use_planning":
                task_fields = {"goal": problem_input.strip(), "available_tools": [], "constraints": []}

            decomposed = decomposer.build(
                task_id="demo_001",
                category=selected_category,
                **task_fields,
            )
            try:
                resp = client.complete(
                    model=selected_model_id,
                    user_prompt=decomposed.prompt,
                    system_prompt=decomposer.get_system_prompt(selected_category),
                    temperature=0.1,
                    max_tokens=1024,
                )
                decomposed = decomposer.parse_response(decomposed, resp.content)
            except Exception as e:
                st.error(f"Generation error: {e}")
                st.stop()

        st.success(f"Generated {len(decomposed.steps)} reasoning steps")

        # Show raw response in expander
        with st.expander("Raw LLM Response"):
            st.text(decomposed.raw_response)

        # Evaluate steps
        with st.spinner("Evaluating each step..."):
            task_eval = step_evaluator.evaluate(decomposed, model_name=selected_display)
            hall_scores = hall_scorer.score_steps("demo", decomposed.steps, decomposed.prompt[:400])
            drift_result = drift_detector.detect("demo", decomposed.prompt, decomposed.steps, selected_category)

        # ── Results display ───────────────────────────────────
        st.divider()
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Steps", task_eval.num_steps)
        col_b.metric("Invalid Steps", task_eval.num_invalid, delta=None)
        col_c.metric("Error Propagated", "Yes" if task_eval.error_propagated else "No")
        col_d.metric("Drift Detected", "Yes" if drift_result.has_drift else "No")

        st.markdown("### Step-by-Step Evaluation")
        eval_dicts = step_evaluator.to_dict(task_eval)
        for se_dict in eval_dicts["step_evaluations"]:
            render_step_card(se_dict)

        # Final answer
        st.markdown("### Final Answer")
        fa_color = "#2ECC71" if task_eval.final_answer_correct else "#E74C3C" if task_eval.final_answer_correct is False else "#888"
        st.markdown(
            f"<div style='border: 2px solid {fa_color}; border-radius: 8px; padding: 12px;'>"
            f"<b>{decomposed.final_answer}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Hallucination summary
        hall_summary = hall_scorer.aggregate(hall_scores)
        if hall_summary["flagged_steps"] > 0:
            st.warning(
                f"⚠️ {hall_summary['flagged_steps']} step(s) flagged for potential hallucination. "
                f"Hallucination rate: {hall_summary['hallucination_rate']:.1%}"
            )

        # Drift
        if drift_result.has_drift:
            st.warning(f"⚠️ Reasoning drift detected: {drift_result.drift_explanation}")

        # Mitigation
        if run_mitigation and (task_eval.num_invalid > 0 or hall_summary["flagged_steps"] > 0):
            st.divider()
            st.markdown("### RAG Re-Grounding")
            suspicious = [c for s in hall_scores for c in s.suspicious_claims]
            with st.spinner("Retrieving context and re-grounding..."):
                rg = regrounder.reground(
                    task=decomposed,
                    model=selected_model_id,
                    suspicious_claims=suspicious,
                    mode=InterventionMode.FULL,
                )
            if rg.success:
                col1, col2 = st.columns(2)
                col1.markdown("**Original Answer**")
                col1.info(rg.original_final_answer)
                col2.markdown("**Re-grounded Answer**")
                col2.success(rg.regrounded_answer)
                if rg.retrieved_docs:
                    with st.expander(f"Retrieved {len(rg.retrieved_docs)} Wikipedia sources"):
                        for doc in rg.retrieved_docs:
                            st.markdown(f"**{doc.title}**")
                            st.text(doc.content[:500])


elif mode == "Browse Results":
    st.title("Evaluation Results Browser")
    results_path = os.path.join(str(_ROOT), cfg["paths"]["outputs"], "eval_results.json")

    if not os.path.exists(results_path):
        st.info("No evaluation results found. Run `python experiments/run_baseline_eval.py` first.")
    else:
        with open(results_path) as f:
            results = json.load(f)

        st.json(results, expanded=False)

        # Model summary table
        if "model_summary" in results:
            st.subheader("Model Comparison")
            import pandas as pd
            rows = []
            for m in results["model_summary"]:
                rows.append({
                    "Model": m["model"],
                    "Step Failure Rate": f"{m['overall_step_failure_rate']:.3f}",
                    "Final Accuracy": f"{m['overall_final_accuracy']:.3f}",
                    "Error Propagation": f"{m['overall_error_propagation_rate']:.3f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Step failure breakdown
        if "step_failure_breakdown" in results:
            st.subheader("Step Failure Distribution")
            breakdown = results["step_failure_breakdown"]
            try:
                import pandas as pd
                import altair as alt
                df = pd.DataFrame([
                    {"Step Index": int(k), "Failure Rate": v["failure_rate"]}
                    for k, v in breakdown.items()
                ])
                chart = alt.Chart(df).mark_bar(color="#E74C3C").encode(
                    x=alt.X("Step Index:O", title="Step Index"),
                    y=alt.Y("Failure Rate:Q", title="Failure Rate", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["Step Index", "Failure Rate"],
                ).properties(width=700, height=300)
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.json(breakdown)

        # Failure report
        if "failure_report" in results:
            st.subheader("Fine-Tuning Target")
            fr = results["failure_report"]
            st.success(f"Recommended fine-tuning target: **{fr.get('finetune_target_category')}**")
            if fr.get("worst_performing"):
                import pandas as pd
                st.dataframe(pd.DataFrame(fr["worst_performing"]), use_container_width=True)


else:  # About
    st.title("About This Project")
    st.markdown("""
## LLM Reasoning Evaluation + LoRA Fine-Tuning Pipeline

**What this does:**
- Evaluates LLM reasoning at the **step level**, not just final answer
- Identifies **where** and **why** reasoning fails across 4 task categories
- Uses failure analysis to guide **targeted LoRA fine-tuning**
- Benchmarks fine-tuned models across **quantization levels** (Q4, Q8)
- Provides **RAG-based mitigation** for hallucination-prone steps

**Task categories:**
- Multi-step arithmetic (GSM8K)
- Factual consistency
- Tool-use planning
- Causal/counterfactual reasoning

**Models evaluated:** Llama-3.1-8B, Llama-3.3-70B, Llama-4-Scout (via Groq API)

**Tech stack:** Groq, LangChain, PEFT, bitsandbytes, llama.cpp, W&B, Streamlit
    """)
