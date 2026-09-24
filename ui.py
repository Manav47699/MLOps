import os
import streamlit as st
import requests
import json

# backend url (http://backend:8000 inside docker)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Week 16 AI Assistant", layout="wide")

st.title("Week 16 AI Assistant (Agentic Edition)")
st.caption("Frontend interface for FastAPI backend with RAG, Tool Calling, and Self-Checking Agentic Loop.")

# sidebar settings
st.sidebar.header("Configuration")

system_prompts = {
    "General Assistant": "You are a helpful and polite AI assistant.",
    "Code Reviewer": "You are a concise code evaluator. Analyze code directly.",
    "Fellowship Mentor": "You are an experienced mentor guiding AI Fellowship students.",
    "Brief & Direct": "Answer concisely in two sentences or less."
}
selected_prompt_name = st.sidebar.selectbox(
    "System Prompt",
    list(system_prompts.keys())
)
chosen_system_prompt = system_prompts[selected_prompt_name]

models = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "meta-llama/Meta-Llama-3-8B-Instruct (vLLM)",
    "mistralai/Mistral-7B-Instruct-v0.2 (vLLM)"
]
selected_model_option = st.sidebar.selectbox("Model", models)
model_name = selected_model_option.split(" ")[0]

# generation parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
top_p = st.sidebar.slider("Top-P", min_value=0.1, max_value=1.0, value=1.0, step=0.05)

use_tool = st.sidebar.checkbox("Enable Weather Tool (Function Calling)", value=False)
use_structured = st.sidebar.checkbox("Use Structured JSON Output (Pydantic)", value=False)

# check if backend is reachable
st.sidebar.markdown("---")
try:
    health_res = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("Backend: Online (Port 8000)")
    else:
        st.sidebar.warning(f"Backend status: {health_res.status_code}")
except Exception:
    st.sidebar.error("Backend: Offline (Start main.py)")


# tabs for agentic loop, chat, rag, and ingestion
tab_agent, tab1, tab2, tab3 = st.tabs([
    "Agentic Loop (Self-Checking Agent)",
    "Chat & Tools",
    "RAG Search",
    "Ingest Documents"
])

with tab_agent:
    st.subheader("Self-Checking Agentic Loop (`/agent-chat`)")
    st.markdown(
        """
        This agent demonstrates a **Self-Check Verification Loop** with **Context Engineering (Tool Result Clearing)**.
        - **Step 1:** Formulate action / tool invocation and draft preliminary response.
        - **Step 2:** Self-verify factual accuracy against tool evidence or retrieved RAG context.
        - **Step 3:** Terminate upon successful verification or iteratively refine (max 3 iterations).
        """
    )

    preset_choice = st.selectbox(
        "Quick Example Queries",
        [
            "Custom Query",
            "What is the weather in Tokyo?",
            "Compare the weather in Paris and Kathmandu and recommend where to walk.",
            "What topics are covered in Week 15 of the AI Fellowship?",
            "Simulated Failure Test (Weather in Tokyo)",
            "What is the secret recipe for quantum coffee? (Out of Domain)"
        ]
    )

    default_query = "Compare the weather in Paris and Kathmandu and recommend where to walk."
    default_failure = False
    if preset_choice == "What is the weather in Tokyo?":
        default_query = "What is the weather in Tokyo?"
        default_failure = False
    elif preset_choice == "Compare the weather in Paris and Kathmandu and recommend where to walk.":
        default_query = "Compare the weather in Paris and Kathmandu and recommend where to walk."
        default_failure = False
    elif preset_choice == "What topics are covered in Week 15 of the AI Fellowship?":
        default_query = "What topics are covered in Week 15 of the AI Fellowship?"
        default_failure = False
    elif preset_choice == "Simulated Failure Test (Weather in Tokyo)":
        default_query = "What is the weather in Tokyo?"
        default_failure = True
    elif preset_choice == "What is the secret recipe for quantum coffee? (Out of Domain)":
        default_query = "What is the secret recipe for quantum coffee?"
        default_failure = False

    agent_query = st.text_area("User Goal / Query", value=default_query, height=70)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        simulate_failure_toggle = st.checkbox(
            "Simulate Tool Failure (Failure Injection Test)",
            value=default_failure,
            help="Simulates a 503 Service Unavailable error to verify that the agent degrades gracefully without hallucinating fake data."
        )
    with col_b:
        max_iters_input = st.slider("Max Loop Iterations Guard", min_value=1, max_value=3, value=3)

    if st.button("Run Agentic Loop", type="primary", key="btn_agent_run"):
        if not agent_query.strip():
            st.warning("Please enter a query.")
        else:
            payload = {
                "query": agent_query.strip(),
                "simulate_failure": simulate_failure_toggle,
                "temperature": temperature,
                "top_p": top_p,
                "max_iterations": max_iters_input
            }
            with st.spinner("Agent running iterative self-check loop..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/agent-chat", json=payload)
                    if res.status_code == 200:
                        data = res.json()

                        # summary cards
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Trajectory Length", f"{data.get('iterations', 1)} iter(s)")
                        m2.metric("Loop Completed", "Yes" if data.get("completed") else "Guarded Max")
                        m3.metric("Tokens Consumed", f"{data.get('total_tokens', 0)} tokens")
                        m4.metric("Context Technique", "Tool Result Clearing")

                        st.markdown("---")
                        st.subheader("Intermediate Thought Iterations")

                        trajectory = data.get("trajectory", [])
                        for step in trajectory:
                            passed = step.get("verification_passed", False)
                            iter_num = step.get("iteration", 1)
                            icon = "✅" if passed else "🔄"
                            badge = "PASSED SELF-CHECK" if passed else "REFINEMENT REQUIRED"

                            with st.expander(f"{icon} Iteration {iter_num}: {step.get('action_taken')} [{badge}]", expanded=True):
                                # step 1: draft
                                st.markdown("##### Step 1: Tool Action & Candidate Draft")
                                st.markdown(f"**Action Executed:** `{step.get('action_taken')}`")
                                st.info(f"**Draft Response:**\n\n{step.get('draft_response')}")

                                # tool result clearing summary
                                st.markdown("##### Context Engineering: Tool Result Clearing")
                                st.markdown(
                                    "*Raw tool outputs are cleared to a single-line summary after each step to prevent token bloat.*"
                                )
                                st.code(f"{step.get('cleared_tool_summary')}", language="markdown")
                                with st.popover("Inspect Raw Verbose Payload (Cleared from memory)"):
                                    st.code(step.get("raw_tool_output", ""), language="json")

                                # step 2: self-check
                                st.markdown("##### Step 2: Self-Verification (Self-Check)")
                                if passed:
                                    st.success(f"**Verdict: PASSED**\n\n**Critique:** {step.get('verification_critique')}")
                                else:
                                    st.warning(f"**Verdict: FAILED (Refinement Needed)**\n\n**Critique:** {step.get('verification_critique')}")

                        # step 3: final output
                        st.markdown("---")
                        st.subheader("Step 3: Final Output")
                        if simulate_failure_toggle:
                            st.warning(data.get("final_answer"))
                        else:
                            st.success(data.get("final_answer"))

                    elif res.status_code == 429:
                        st.error("Rate limit reached. Please wait a moment.")
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

with tab1:
    st.subheader("Direct Chat")
    user_prompt = st.text_area("User Message", placeholder="Ask something, e.g. 'What is the weather in Kathmandu?'")

    if st.button("Send", type="primary"):
        if not user_prompt.strip():
            st.warning("Please enter a message.")
        else:
            payload = {
                "prompt": user_prompt,
                "system_prompt": chosen_system_prompt,
                "model": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "use_tool": use_tool,
                "use_structured_output": use_structured
            }

            try:
                res = requests.post(f"{BACKEND_URL}/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.info(f"Model used: {data.get('model')}")

                    if "tool_called" in data:
                        st.markdown(f"**Tool Called:** `{data['tool_called']}`")
                        st.write("Tool Arguments:", data.get("tool_args"))
                        st.write("Tool Result:", data.get("tool_result"))

                    if data.get("structured_data"):
                        st.markdown("**Structured Output (Pydantic):**")
                        st.json(data["structured_data"])

                    st.markdown("**Response:**")
                    st.write(data.get("content"))
                elif res.status_code == 429:
                    st.error("Rate limit reached. Please wait a moment before sending another request.")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab2:
    st.subheader("RAG Document Q&A")
    rag_query = st.text_input("Question", placeholder="e.g. What does Week 15 focus on?")

    if st.button("Search & Answer"):
        if not rag_query.strip():
            st.warning("Please enter a question.")
        else:
            payload = {
                "question": rag_query,
                "temperature": temperature,
                "top_p": top_p
            }
            try:
                res = requests.post(f"{BACKEND_URL}/rag-chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.info(f"Model used: {data.get('model')}")
                    st.markdown("**Answer:**")
                    st.write(data.get("answer"))

                    st.markdown("**Retrieved Context from ChromaDB:**")
                    retrieved = data.get("retrieved_context", [])
                    if retrieved:
                        for i, doc in enumerate(retrieved):
                            src = doc.get("metadata", {}).get("source", "Unknown")
                            with st.expander(f"Chunk {i+1} - Source: {src}"):
                                st.write(doc.get("text"))
                    else:
                        st.write("No matching documents found.")
                elif res.status_code == 429:
                    st.error("Rate limit reached. Please wait a moment.")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab3:
    st.subheader("Document Ingestion")
    st.write("Load text documents from a directory and index them in ChromaDB.")

    docs_dir = st.text_input("Folder path", value="./docs")

    if st.button("Run Ingestion"):
        try:
            res = requests.post(f"{BACKEND_URL}/ingest", params={"folder_path": docs_dir})
            if res.status_code == 200:
                count = res.json().get("chunks_ingested", 0)
                st.success(f"Ingestion complete: {count} chunks added to ChromaDB.")
            else:
                st.error(f"Ingestion failed: {res.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
