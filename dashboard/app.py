"""
dashboard/app.py
Streamlit demo application with three modes:
  1. LIVE EVAL: Enter a reasoning problem, pick a model, watch step-by-step evaluation
  2. RESULTS:   Browse + compare saved evaluation results with charts
  3. ABOUT:     Project description

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
import glob as globlib
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
    with open(_ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()

# ── Init components (cached) ──────────────────────────────────
@st.cache_resource
def get_components():
    from models import get_client
    from eval.task_decomposer import TaskDecomposer
    from eval.step_evaluator import StepEvaluator
    from eval.hallucination_scorer import HallucinationScorer
    from eval.drift_detector import DriftDetector
    from mitigation.retriever import WikipediaRetriever
    from mitigation.regrounder import Regrounder, InterventionMode

    provider = cfg.get("provider", "groq")
    try:
        client = get_client(cfg, provider=provider)
    except Exception:
        return None, None, None, None, None, None, None

    decomposer = TaskDecomposer()
    step_evaluator = StepEvaluator(client=client, evaluator_model=cfg["models"]["evaluator_model"])
    hall_scorer = HallucinationScorer(client=client, model=cfg["models"]["evaluator_model"])
    drift_detector = DriftDetector(client=client, model=cfg["models"]["evaluator_model"])
    retriever = WikipediaRetriever()
    regrounder = Regrounder(client=client, retriever=retriever)
    return client, decomposer, step_evaluator, hall_scorer, drift_detector, retriever, regrounder


# ── Helpers ───────────────────────────────────────────────────

def load_all_eval_results() -> list[dict]:
    """Load all eval_results_*.json files from outputs/."""
    outputs_dir = str(_ROOT / cfg["paths"]["outputs"])
    files = sorted(globlib.glob(os.path.join(outputs_dir, "eval_results_*.json")))
    results = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            data["_filename"] = os.path.basename(f)
            per_metrics = data.get("per_run_metrics", [])
            model_summary = data.get("model_summary", [])
            data["_is_valid_run"] = bool(per_metrics) and bool(model_summary)
            data["_num_metrics_rows"] = len(per_metrics)
            results.append(data)
        except Exception:
            pass
    return results


def extract_model_name(filename: str) -> str:
    """Extract model name from filename like eval_results_qwen2.5-3b_mit_123.json"""
    name = filename.replace("eval_results_", "").split("_mit_")[0].split("_nomir_")[0]
    return name


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

mode = st.sidebar.radio("Mode", ["Results Comparison", "Live Evaluation", "About"])

# ── Results Comparison (main page) ────────────────────────────

if mode == "Results Comparison":
    st.title("📊 Evaluation Results Comparison")

    all_results = load_all_eval_results()

    if not all_results:
        st.info("No evaluation results found. Run the evaluation first:\n\n"
                "```bash\npython experiments/run_baseline_eval.py --provider ollama "
                "--models qwen2.5:3b --judge-provider groq --samples 20 --no-wandb\n```")
        st.stop()

    include_invalid = st.checkbox("Include invalid/empty runs", value=False)

    valid_runs = [r for r in all_results if r.get("_is_valid_run")]
    invalid_runs = [r for r in all_results if not r.get("_is_valid_run")]

    if invalid_runs:
        st.info(
            f"Detected {len(invalid_runs)} invalid/empty run files. "
            "These are usually aborted runs with 0 completed tasks."
        )

    runs_for_ui = all_results if include_invalid else valid_runs

    # Let user pick which runs to compare
    run_options = {}
    for r in runs_for_ui:
        fname = r["_filename"]
        model = extract_model_name(fname)
        tag = "[INVALID] " if not r.get("_is_valid_run") else ""
        label = f"{model} ({fname})"
        if tag:
            label = f"{tag}{label}"
        run_options[label] = r

    if not run_options:
        st.warning("No valid runs available with current filter settings.")
        st.stop()

    selected_runs = st.multiselect(
        "Select runs to compare",
        list(run_options.keys()),
        default=list(run_options.keys())[-2:] if len(run_options) >= 2 else list(run_options.keys()),
    )

    if not selected_runs:
        st.warning("Select at least one run to view results.")
        st.stop()

    st.subheader("Run Health")
    try:
        import pandas as pd
        import altair as alt

        health_rows = []
        for r in all_results:
            health_rows.append({
                "Run": r["_filename"],
                "Status": "valid" if r.get("_is_valid_run") else "invalid",
                "Metric Rows": r.get("_num_metrics_rows", 0),
            })
        health_df = pd.DataFrame(health_rows)
        chart = alt.Chart(health_df).mark_bar().encode(
            x=alt.X("Run:N", sort=None),
            y=alt.Y("Metric Rows:Q"),
            color=alt.Color("Status:N", scale=alt.Scale(domain=["valid", "invalid"], range=["#2ECC71", "#E74C3C"])),
            tooltip=["Run", "Status", "Metric Rows"],
        ).properties(height=220)
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass

    # ── Overall comparison table ──────────────────────────────
    st.subheader("Overall Model Comparison")

    import pandas as pd

    rows = []
    for label in selected_runs:
        r = run_options[label]
        for m in r.get("model_summary", []):
            rows.append({
                "Model": m["model"],
                "Tasks": m["total_tasks"],
                "Accuracy": m["overall_final_accuracy"],
                "Step Failure": m["overall_step_failure_rate"],
                "Error Propagation": m["overall_error_propagation_rate"],
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({
                "Accuracy": "{:.1%}",
                "Step Failure": "{:.1%}",
                "Error Propagation": "{:.1%}",
            }),
            use_container_width=True,
        )

        # Accuracy vs step-failure scatter makes regressions obvious.
        try:
            import altair as alt
            scatter = alt.Chart(df).mark_circle(size=150).encode(
                x=alt.X("Step Failure:Q", title="Step Failure Rate"),
                y=alt.Y("Accuracy:Q", title="Final Accuracy", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N"),
                tooltip=["Model", "Tasks", "Accuracy", "Step Failure", "Error Propagation"],
            ).properties(height=320)
            st.markdown("**Accuracy vs Step Failure**")
            st.altair_chart(scatter, use_container_width=True)
        except Exception:
            pass

    # ── Per-category comparison ───────────────────────────────
    st.subheader("Per-Category Breakdown")

    cat_rows = []
    for label in selected_runs:
        r = run_options[label]
        for run_metrics in r.get("per_run_metrics", []):
            cat_rows.append({
                "Model": run_metrics["model"],
                "Category": run_metrics["category"],
                "Tasks": run_metrics["total_tasks"],
                "Accuracy": run_metrics["final_accuracy"],
                "Step Failure": run_metrics["step_failure_rate"],
                "Invalid Steps": run_metrics["invalid_steps"],
                "Total Steps": run_metrics["total_steps"],
                "Hallucination Rate": run_metrics.get("hallucination_rate", 0),
            })

    if cat_rows:
        cat_df = pd.DataFrame(cat_rows)
        st.dataframe(
            cat_df.style.format({
                "Accuracy": "{:.1%}",
                "Step Failure": "{:.1%}",
                "Hallucination Rate": "{:.1%}",
            }),
            use_container_width=True,
        )

        # ── Charts ────────────────────────────────────────────
        st.subheader("Visual Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Accuracy by Category**")
            try:
                import altair as alt
                chart_acc = alt.Chart(cat_df).mark_bar().encode(
                    x=alt.X("Category:N", title="Category"),
                    y=alt.Y("Accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Model:N"),
                    xOffset="Model:N",
                    tooltip=["Model", "Category", "Accuracy"],
                ).properties(height=350)
                st.altair_chart(chart_acc, use_container_width=True)
            except ImportError:
                st.bar_chart(cat_df.pivot(index="Category", columns="Model", values="Accuracy"))

        with col2:
            st.markdown("**Step Failure Rate by Category**")
            try:
                import altair as alt
                chart_sf = alt.Chart(cat_df).mark_bar().encode(
                    x=alt.X("Category:N", title="Category"),
                    y=alt.Y("Step Failure:Q", title="Step Failure Rate", scale=alt.Scale(domain=[0, 0.3])),
                    color=alt.Color("Model:N"),
                    xOffset="Model:N",
                    tooltip=["Model", "Category", "Step Failure"],
                ).properties(height=350)
                st.altair_chart(chart_sf, use_container_width=True)
            except ImportError:
                st.bar_chart(cat_df.pivot(index="Category", columns="Model", values="Step Failure"))

    # ── Step failure distribution ─────────────────────────────
    st.subheader("Step Failure Distribution (by step index)")
    for label in selected_runs:
        r = run_options[label]
        breakdown = r.get("step_failure_breakdown", {})
        if breakdown:
            model_name = r.get("model_summary", [{}])[0].get("model", label)
            st.markdown(f"**{model_name}**")
            try:
                import altair as alt
                bf_df = pd.DataFrame([
                    {"Step Index": int(k), "Failure Rate": v["failure_rate"]}
                    for k, v in breakdown.items()
                ])
                chart = alt.Chart(bf_df).mark_bar(color="#E74C3C").encode(
                    x=alt.X("Step Index:O", title="Step Index"),
                    y=alt.Y("Failure Rate:Q", title="Failure Rate", scale=alt.Scale(domain=[0, 0.5])),
                    tooltip=["Step Index", "Failure Rate"],
                ).properties(height=250)
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.json(breakdown)

    # ── Fine-tuning recommendation ────────────────────────────
    st.subheader("Fine-Tuning Recommendation")
    for label in selected_runs:
        r = run_options[label]
        fr = r.get("failure_report", {})
        if fr.get("finetune_target_category"):
            model_name = fr.get("finetune_target_model", "")
            st.success(f"**{model_name}** → Target: **{fr['finetune_target_category']}**")

    # ── Raw JSON explorer ─────────────────────────────────────
    with st.expander("📄 Raw JSON Explorer"):
        selected_raw = st.selectbox("Select run", selected_runs)
        st.json(run_options[selected_raw], expanded=False)


elif mode == "Live Evaluation":
    st.title("Live Step-Level Reasoning Evaluation")
    st.markdown(
        "Enter a reasoning problem below. The system will generate a chain-of-thought "
        "reasoning trace and evaluate each step for logical validity, hallucination, "
        "and consistency."
    )

    model_options = {m["display_name"]: m["id"] for m in cfg["models"]["eval_models"]}
    selected_display = st.sidebar.selectbox("Model", list(model_options.keys()))
    selected_model_id = model_options[selected_display]
    category_options = cfg["eval"]["categories"]
    selected_category = st.sidebar.selectbox("Task Category", category_options)

    client, decomposer, step_evaluator, hall_scorer, drift_detector, retriever, regrounder = get_components()

    if client is None:
        st.error("No LLM provider available. Set GROQ_API_KEY in .env or ensure Ollama is running.")
        st.stop()

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

        with st.expander("Raw LLM Response"):
            st.text(decomposed.raw_response)

        with st.spinner("Evaluating each step..."):
            task_eval = step_evaluator.evaluate(decomposed, model_name=selected_display)
            hall_scores = hall_scorer.score_steps("demo", decomposed.steps, decomposed.prompt[:400])
            drift_result = drift_detector.detect("demo", decomposed.prompt, decomposed.steps, selected_category)

        st.divider()
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Steps", task_eval.num_steps)
        col_b.metric("Invalid Steps", task_eval.num_invalid)
        col_c.metric("Error Propagated", "Yes" if task_eval.error_propagated else "No")
        col_d.metric("Drift Detected", "Yes" if drift_result.has_drift else "No")

        st.markdown("### Step-by-Step Evaluation")
        eval_dicts = step_evaluator.to_dict(task_eval)
        for se_dict in eval_dicts["step_evaluations"]:
            render_step_card(se_dict)

        st.markdown("### Final Answer")
        fa_color = "#2ECC71" if task_eval.final_answer_correct else "#E74C3C" if task_eval.final_answer_correct is False else "#888"
        st.markdown(
            f"<div style='border: 2px solid {fa_color}; border-radius: 8px; padding: 12px;'>"
            f"<b>{decomposed.final_answer}</b></div>",
            unsafe_allow_html=True,
        )

        hall_summary = hall_scorer.aggregate(hall_scores)
        if hall_summary["flagged_steps"] > 0:
            st.warning(f"⚠️ {hall_summary['flagged_steps']} step(s) flagged for potential hallucination. "
                       f"Rate: {hall_summary['hallucination_rate']:.1%}")

        if drift_result.has_drift:
            st.warning(f"⚠️ Reasoning drift detected: {drift_result.drift_explanation}")

        if run_mitigation and (task_eval.num_invalid > 0 or hall_summary["flagged_steps"] > 0):
            st.divider()
            st.markdown("### RAG Re-Grounding")
            suspicious = [c for s in hall_scores for c in s.suspicious_claims]
            with st.spinner("Retrieving context and re-grounding..."):
                rg = regrounder.reground(
                    task=decomposed, model=selected_model_id,
                    suspicious_claims=suspicious, mode=InterventionMode.FULL,
                )
            if rg.success:
                col1, col2 = st.columns(2)
                col1.markdown("**Original Answer**")
                col1.info(rg.original_final_answer)
                col2.markdown("**Re-grounded Answer**")
                col2.success(rg.regrounded_answer)


else:  # About
    st.title("About This Project")
    st.markdown("""
## LLM Reasoning Evaluation & Fine-Tuning Pipeline

**What this does:**
- Evaluates LLM reasoning at the **step level**, not just final answer
- Uses an independent **70B judge** (Llama 3.3 via Groq) to grade each step
- Identifies **where exactly** reasoning breaks across 4 task categories
- Uses failure analysis to guide **targeted QLoRA fine-tuning**
- Benchmarks fine-tuned vs base models with proper comparison

**Task categories:**
- **Multistep Arithmetic** — GSM8K benchmark (1319 test problems)
- **Factual Consistency** — 60 generated diverse tasks
- **Tool-Use Planning** — 31 generated multi-tool tasks
- **Causal Counterfactual** — 60 generated reasoning tasks

**Key results:**
- Step failure rate: **11.7% → 1.1%** (91% reduction after fine-tuning)
- Tool-use: 0% step failures, 100% accuracy
- Fine-tuned 3B beats baseline 8B on reasoning quality

**Tech stack:** Ollama · Groq · QLoRA · PEFT · TRL · llama.cpp · HuggingFace · Streamlit
    """)

