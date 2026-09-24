# Week 16 Evaluation Harness Report: Agentic Verification Loop

- **Task Completion Rate:** 100.0% (5/5)
- **Tool-Call Correctness:** 100.0% (5/5)
- **Average Trajectory Length:** 1.40 iterations/query
- **Average Tokens Consumed:** 182.8 tokens/query (Total: 914)

### Detailed Evaluation Results Table

| Test ID | Query Type / Scenario | Completed | Tool Correct | Trajectory Length | Tokens | Failure Classification | Final Output Summary |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **TC-1** | Single Tool Weather Lookup | Yes | Yes | 1 | 80 | None (Passed) | The current weather in Tokyo is 18C with Sunny skies.... |
| **TC-2** | RAG Knowledge Retrieval | Yes | Yes | 1 | 310 | None (Passed) | Week 15 of the AI Fellowship focuses on Applied AI and Engineering AI Systems. Key topics ... |
| **TC-3** | Multi-Step Comparison | Yes | Yes | 2 | 199 | None (Passed) | In Paris, the weather is 22C and Cloudy. In Kathmandu, the weather is 24C and Pleasant. Fo... |
| **TC-4** | Failure Injection (Fault Handling) | Yes | Yes | 2 | 232 | None (Passed) | Notice: The weather service is currently unreachable (503 Service Unavailable). As a resul... |
| **TC-5** | Boundary / Out-of-Domain | Yes | Yes | 1 | 93 | None (Passed) | I searched the knowledge base and available tools, but no verifiable information was found... |

### Failure Log & Taxonomy Analysis
- **Hard Failures (0 recorded):** Zero unhandled crashes or unhandled exceptions occurred. The strict `max_iterations = 3` guard prevented infinite execution.
- **Soft Failures (0 recorded):** No hallucinated answers detected. In out-of-domain queries, the agent verified lack of evidence and appropriately abstained.
- **Cascading Soft Failures (0 recorded):** Under simulated tool failure (TC-4), the self-check verifier identified the 503 error immediately, preventing the agent from making speculative claims or compounding errors into subsequent steps.