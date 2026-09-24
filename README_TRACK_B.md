# Week 17 MLOps - Track B: Agentic AI MLOps Pipeline

This repository branch (`track-b`) applies production MLOps disciplines to the **Week 16 AI Assistant & Agentic Verification Loop**. Because agentic LLM systems lack traditional weights or epochs, tracking and evaluation are applied directly to **prompts, execution trajectories, structured traces, and regression test suites** using `uv`, `MLflow`, and `Evidently AI`.

---

## 1. Environment & Reproducibility (uv)

### Problems Solved by `uv`
- **Complex Agent Tooling Matrix**: The agent stack combines FastAPI, Streamlit, ChromaDB, OpenAI client, Pydantic v2, and Evidently LLM evaluators. Traditional `pip` dependency resolution frequently results in version conflicts across pydantic core and numpy.
- **Hermetic Lockfile Pinning**: `uv.lock` guarantees exact, reproducible installation of 170+ dependencies across Linux, macOS, and container environments.
- **One-Command Setup**:
  ```bash
  uv sync
  ```
  This command creates an isolated `.venv`, installs all locked wheels in under 1 second, and ensures consistent execution.

---

## 2. Experiment Tracking & Prompt Iteration Strategy (MLflow)

### Iteration Rationale & Failure Diagnoses
We explicitly versioned three system prompt configurations to demonstrate hypothesis-driven prompt engineering guided by trace analysis rather than speculative tweaks.

1. **`prompt_v1` (Naive Baseline)**:
   - *Prompt*: `"You are a basic AI assistant. Answer user queries. Use tools if available. Stop after the first answer."`
   - *Trace Diagnosis*: On comparison queries (Paris vs. Kathmandu), the agent stopped searching after querying Paris. On simulated upstream tool failure (503), it terminated with an unhandled raw error dictionary.
   - *Failure Artifact*: `traces/trace_prompt_v1_compare_current_weather_in_par.json` (early loop termination, missing Kathmandu).

2. **`prompt_v2` (Decomposed Tool-Aware)**:
   - *Prompt*: `"You are a tool-aware AI assistant. When a query involves multiple entities (such as comparing two cities), systematically query tools for EACH entity before answering. Summarize findings accurately."`
   - *Trace Diagnosis*: Correctly decomposed Paris and Kathmandu into sequential tool calls. However, on simulated 503 outage, it retried repeatedly without a structured fallback disclaimer, exhausting loop iterations without verification.
   - *Failure Artifact*: `traces/trace_prompt_v2_check_weather_in_tokyo_during.json` (max iterations hit on tool failure).

3. **`prompt_v3` (Robust Self-Verifying)**:
   - *Prompt*: `"You are a robust agentic AI assistant. Step 1: Decompose multi-step queries and invoke tools with verified arguments. Step 2: Fact-check answers against tool outputs and ChromaDB knowledge contexts. Step 3: If any tool encounters an error, state the outage clearly without fabricating unverified data. Step 4: Verify constraints before finalizing."`
   - *Trace Diagnosis*: Resolved both failures. Decomposes multi-entity comparisons and, upon detecting a 503 outage, invokes guardrails to output a verified graceful degradation notice with zero hallucination.

### Side-by-Side Prompt Run Comparison Table

| Metric | `prompt_v1` (Naive) | `prompt_v2` (Tool-Aware) | `prompt_v3` (Self-Verifying) | Winning Version |
| :--- | :--- | :--- | :--- | :--- |
| **Completion Rate** | 60.0% | 80.0% | **100.0%** | **`prompt_v3`** |
| **Evidently Pass Rate (`pct_tests_passed`)** | 60.0% | 80.0% | **100.0%** | **`prompt_v3`** |
| **Mean Iterations per Query** | **1.00** | 1.60 | 1.20 | `prompt_v3` (efficient multi-turn) |
| **Mean Token Consumption** | **93.6** | 159.6 | 123.6 | `prompt_v3` (optimal balance) |
| **Multi-Entity Comparison Handling** | Failed (Paris only) | Passed (Paris + KTM) | **Passed (Paris + KTM)** | `prompt_v3` |
| **Service Outage Fallback Handling** | Failed (raw error) | Failed (loop exhaust) | **Passed (verified notice)** | `prompt_v3` |

### Structured Trace Artifacts
Every test query execution writes a structured step-by-step JSON record logged directly to MLflow:
- `traces/trace_<prompt_ver>_<query_slug>.json`
- Each trace includes: `query`, `prompt_version`, `total_iterations`, `termination_reason`, `execution_time_seconds`, and a list of steps with `{step, tool, args, result, reasoning, intermediate_decision, draft}`.

---

## 3. Monitoring & Regression Testing (Evidently AI)

### Golden Reference Dataset & Evaluation Suite
`eval_evidently.py` evaluates agent answers against a curated golden reference dataset (5 benchmark test queries covering single tool calls, multi-entity comparisons, ChromaDB RAG retrieval, upstream 503 outages, and out-of-domain abstention).

### Evaluator Checks:
1. **Reference-Based Correctness**: Verifies that key factual entities and statements in the reference answer are present without contradictions.
2. **Goal Completion**: Verifies that the agent finished its self-verification loop without early exit or unhandled exceptions.

### Evidently Regression Results & Trend

| Prompt Version | Evidently Test Pass Rate | Failed Test Cases & Diagnosis |
| :--- | :--- | :--- |
| `prompt_v1` | **60.0%** | TC-1 (Paris only; missing Kathmandu) & TC-4 (raw 503 error) |
| `prompt_v2` | **80.0%** | TC-4 (exhausted max iterations on tool outage) |
| `prompt_v3` | **100.0%** | None. All 5 regression tests passed. |

### Judge Sanity Check
Manual review of agent outputs confirms the automated judge verdicts:
- In TC-1 with `prompt_v1`, the output was `"In Paris, the weather is 22C and Cloudy."` The judge flagged missing `'kathmandu'`, which is factually accurate.
- In TC-4 with `prompt_v2`, the agent output did not include the required verified outage disclaimer.
- In TC-4 with `prompt_v3`, the agent cleanly stated `"Notice: The weather service is currently unreachable (503 Service Unavailable)... no fabricated values are provided"`, which correctly triggered a PASS.

All Evidently test suites are exported to HTML and tracked in MLflow:
- `reports/evidently_agent_eval.html`
- `reports/evidently_agent_eval_prompt_v1.html`
- `reports/evidently_agent_eval_prompt_v2.html`
- `reports/evidently_agent_eval_prompt_v3.html`

---

## 4. How to Run (Step-by-Step)

### 1. Environment Sync
```bash
uv sync
```

### 2. Execute Prompt Iteration Experiments & MLflow Trace Logging
```bash
uv run python main.py --run-eval
```
Runs 5 benchmark queries across `prompt_v1`, `prompt_v2`, and `prompt_v3`, records structured JSON traces in `traces/`, and logs parameters, metrics, and trace artifacts to MLflow.

### 3. Run Evidently AI LLM Regression Test Suite
```bash
uv run python eval_evidently.py
```
Evaluates all prompt versions against the golden reference set, exports HTML evaluation reports to `reports/`, and logs `pct_tests_passed` to the corresponding MLflow runs.

### 4. Start Backend API & Streamlit UI
```bash
# Start FastAPI backend
uv run python main.py --port 8000 &

# Start Streamlit UI
uv run streamlit run ui.py
```
In the Streamlit interface, navigate to the **Agentic Loop** tab to select between `prompt_v1`, `prompt_v2`, and `prompt_v3` and inspect the live verification steps and tool summaries.

### 5. Launch MLflow Dashboard
```bash
uv run mlflow ui --port 5000
```
Open `http://localhost:5000` to inspect experiment `week17-agent-eval`, compare the 3 prompt runs side-by-side, inspect `pct_tests_passed` metrics, and view uploaded trace JSON and Evidently HTML artifacts.
