# ML Co-pilot 🧠🤖  
*Turning one ML engineer into a whole 24/7 ML team using Google ADK*

---

## 1. Problem & Goals

Modern ML work is rarely “just run this model.” A typical workflow includes:

- Clarifying the task and scoping experiments  
- Searching papers / repos / Kaggle baselines  
- Implementing and debugging code  
- Iterating on failures and reporting results  
- Keeping the human in the loop for important decisions  

This project is an **agentic ML co-pilot** built with **Google ADK** that behaves like a small ML team:

- A **Project Planner** that behaves like a senior ML team lead  
- A **Researcher cluster** (web + Kaggle + “brain” aggregator)  
- An **ML Engineer** that writes and executes Python code with an **LLM-as-a-judge** loop  
- **Observability** via ADK plugins + AgentOps

The capstone demo shows them working together end-to-end on tasks like:

> “Find a 100% accuracy solution for the Iris dataset on the internet (Google / Kaggle), implement that single approach, train, print metrics, and save the model locally.”

---

## 2. Setup & Installation

### 2.1 Prerequisites

- **Python**: 3.10–3.12 (developed on 3.12)
- **Node.js**: **20+** (required by the Kaggle MCP server via `mcp-remote`)
- A **Google API key** for Gemini models
- (Optional but recommended) **AgentOps API key** for telemetry

### 2.2 Clone & Install

```bash
git clone https://github.com/Sashmark97/ML-Co-pilot.git
cd ML-Co-pilot

python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2.3 Environment Variables

Each agent directory (`ml_engineer`, `ml_researcher`, `project_planner`) can have its own `.env`.
Typical contents:

```bash
# Required for all agents
GOOGLE_API_KEY=your_google_api_key

# AgentOps (optional but recommended)
AGENTOPS_API_KEY=your_agentops_api_key

# Kaggle MCP (needed for ml_researcher)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

ADK automatically loads `.env` for the given app directory.

### 2.4 Running with ADK Web UI

From the repo root:

```bash
# Run all apps in this directory
adk web .
```

Then in the ADK UI you can select and run each app separately:

* `project_planner`
* `ml_researcher`
* `ml_engineer`

Or run a specific app directly, e.g.:

```bash
adk web ml_researcher
```

Each agent can be used:

* **Independently** (e.g., only research, or only code execution), or
* As part of a **multi-agent workflow** coordinated externally (by the user) or internally (via shared state and conventions).

---

## 3. High-Level Architecture

### 3.1 Agent Roles and Data Flow

```text
User
  ↓
Project Planner (project_planner/)
  ↓ (plan & HITL-style approval — currently stubbed to auto-approve)
Research Orchestrator (ml_researcher/)
  ├─ WebResearchAgent      – uses Google Search
  ├─ KaggleResearchAgent   – uses Kaggle MCP server
  └─ ResearchBrain         – merges web + Kaggle into a concrete plan
  ↓
ML_Engineer Loop (ml_engineer/)
  ├─ ML_Engineer           – writes & executes Python via run_python tool
  └─ EngineerJudge         – LLM-as-a-judge, exits loop when task is satisfied
  ↓
Project Planner (optionally summarizes)
  ↓
User (final summary + pointers to artifacts)
```

### 3.2 Core Design Ideas

* **LLM-as-team lead**: Project Planner decides *what* to do and structures experiment plans with a HITL-style approval prompt.
* **LLM-as-researcher**: Web & Kaggle agents fetch **papers, repos, benchmarks, and Kaggle resources**, then the Brain agent condenses them into something directly actionable.
* **LLM-as-engineer**:

  * Generates a **single executable Python script** per attempt.
  * Uses a custom **`run_python` tool** to execute code in the local environment.
  * A strict **EngineerJudge** checks if the task is *formally* satisfied (right dataset, no crash, required outputs printed), not whether the model is “good enough”.
* **LoopAgent** for ML engineer:

  * Up to N attempts (e.g. 3).
  * Judge can call `exit_loop` early when success criteria are met.
* **Observability**:

  * ADK’s `LoggingPlugin` for structured logs.
  * Custom `InvocationMetricsPlugin` counting agent/tool invocations.
  * **AgentOps** integration for cost & trace inspection.

---

## 4. Repository Structure

```text
ML-Co-pilot/
├─ ml_engineer/
│  ├─ agent.py          # ML_Engineer + EngineerJudge + LoopAgent root_agent
│  ├─ .env              # GOOGLE_API_KEY, AGENTOPS_API_KEY, etc. for this app
│  └─ (optional) debug_runner.py
│
├─ ml_common/
│  ├─ observability.py  # AgentOps observability tools
│  └─ plugins.py   # plugins for debugging locally
│
├─ ml_researcher/
│  ├─ agent.py          # WebResearchAgent, KaggleResearchAgent, ResearchBrain,
│  │                   # and a ResearchOrchestrator LoopAgent as root_agent
│  ├─ .env              # GOOGLE_API_KEY, KAGGLE_USERNAME, KAGGLE_KEY, etc.
│  └─ debug_runner.py   # Local asyncio test runner with observability plugins
│
├─ project_planner/
│  ├─ agent.py          # Project Planner root_agent (senior ML TL + HITL stub)
│  ├─ .env              # GOOGLE_API_KEY, AGENTOPS_API_KEY
│
├─ ml_team/
│  ├─ agent.py          # Root agent for team work between all present agents
│  ├─ .env              # GOOGLE_API_KEY, AGENTOPS_API_KEY
│
├─ requirements.txt     # Python dependencies
└─ README.md            # (this file)
```

Each `<app>/agent.py` exposes a `root_agent` that ADK can load via `adk web`.

---

## 5. Agent Details

### 5.1 Project Planner (`project_planner/agent.py`)

**Role:** Senior ML/DL team lead and entry point for complex tasks.

Behavior:

* Internally classifies each user message into:

  * **Planning Request** – design experiment / project / pipeline.
  * **Explanation Request** – primarily conceptual/theoretical.
* For **Explanation**:

  * Outputs: `Summary:`, `Details:`, optional `Practical tips:`.
* For **Planning**:

  * Outputs:

    1. **High-level idea**
    2. **Experiment plan** (numbered steps)
    3. **Models & approaches to try** (baselines + stronger models)
    4. **Implementation notes for the team**:

       * Explicit hand-offs, e.g.:

         * “Hand off to Research Agent: find recent SOTA OCR models and benchmarks.”
         * “Hand off to ML_Engineer: implement training loop with X, log metrics Y.”
    5. **Questions for the user (if any)**
    6. **HITL prompt**:

       ```text
       Do you approve this plan? Please reply with:
         - APPROVE
         - REVISE: <your corrections>
         - REJECT
       HITL_STATUS: PLAN_APPROVED   # currently stubbed to auto-approve
       ```

> For the capstone and robustness, the planner currently stubs the HITL to `PLAN_APPROVED` in the output so that downstream agents can run without explicit human input during evaluation. The HITL structure is present and can be re-activated later by conditioning on actual user replies.

---

### 5.2 Researcher Cluster (`ml_researcher/agent.py`)

#### 5.2.1 WebResearchAgent

* **Model**: `gemini-2.0-flash`
* **Tools**: `google_search` (ADK’s built-in Google Search tool)
* Responsibilities:

  * Find **recent papers**, **blog posts**, **documentation**, and **GitHub repos / HF models**.
  * Typical queries: `"trocr arxiv"`, `"donut document understanding transformer"`, `"state of the art OCR 2024 site:arxiv.org"`, etc.
* Output structure:

```text
WEB_NOTES:
MODELS:
- Name (Year) – short description; paper title and venue if available.
DATASETS:
- Name – type of data – typical use – synthetic/real-world – benchmark status.
IMPLEMENTATIONS:
- Library / framework – repo or HF id – maturity (official / community / example).
METRICS:
- Metric – what it measures – typical ranges or SOTA values.

WEB_SUMMARY:
- 3–8 sentences tuned for an ML engineer about what matters most.
```

This agent can be used entirely **standalone**, e.g.:

> “Give me a summary of SOTA OCR models and their main datasets.”

---

#### 5.2.2 KaggleResearchAgent

* **Model**: `gemini-2.5-flash-lite` (or similar)
* **Tools**: `kaggle_mcp` via `MCPToolset` and `npx mcp-remote https://www.kaggle.com/mcp`
* Responsibilities:

  * Find relevant Kaggle **datasets**, **competitions**, and **notebooks**.
* Output structure:

```text
KAGGLE_NOTES:
DATASETS:
- Dataset name (slug if available) – description – target – size/timespan – why useful.
COMPETITIONS:
- Competition name (slug) – task/target – main metric – why relevant.
NOTEBOOKS:
- Notebook title / id – what it demonstrates – why useful as starting point.

KAGGLE_SUMMARY:
- 2–6 sentences summarizing most practically useful Kaggle resources.
```

If Kaggle tools are limited / return nothing, it still returns **search handles** (e.g., keywords to try on Kaggle manually).

This agent can also be used **alone**, e.g.:

> “Find Kaggle datasets and competitions relevant to invoice OCR / document understanding.”

---

#### 5.2.3 ResearchBrain

* **Model**: `gemini-2.0-flash`
* **Tools**: none (pure reasoning)
* Inputs:

  * `STATE_RESEARCH_TASK`
  * `WEB_NOTES`, `WEB_SUMMARY`
  * `KAGGLE_NOTES`, `KAGGLE_SUMMARY`
* Output structure:

```text
FINAL_SUMMARY:
- 1–3 paragraphs.
- High-level, opinionated explanation targeted at an ML engineer.
- Names important models / papers / datasets / competitions and their role.

IMPLEMENTATION_SUGGESTIONS:
- 5–15 bullets of concrete next steps for code implementation.
- Can reference repos (org/repo), HF IDs, Kaggle slugs, metrics, and target ranges.
```

You can invoke **ResearchBrain** as a separate app to get a merged view when state keys are pre-populated (via ADK Web state editor), or more commonly as part of the `ml_researcher` root `LoopAgent` that orchestrates Web + Kaggle + Brain automatically.

---

### 5.3 ML Engineer Loop (`ml_engineer/agent.py`)

#### 5.3.1 `run_python` Tool

* A **plain Python callable** wired as an ADK tool.
* Executes arbitrary Python code:

  * `compile(..., "exec")` + `exec(...)` in an isolated namespace.
  * Captures `stdout` / `stderr` via `contextlib.redirect_stdout/redirect_stderr`.
  * Returns:

    ```text
    STATUS: OK or ERROR: <ExceptionType>: <message>

    STDOUT:
    ...

    STDERR:
    ...
    ```

> ⚠️ This is intentionally **unsafe** and is for local experimentation / evaluation only. For real deployments, you’d replace this with a sandboxed executor.

#### 5.3.2 ML_Engineer (LlmAgent)

* **Model**: `gemini-2.5-flash` (full flash; `flash-lite` had AFC incompatibility quirks)
* **Tools**: `[run_python]`
* Behavior per attempt:

1. Restates understanding & plan in 1–2 sentences.
2. Writes **one complete Python script** in a single code block:

   * Uses small / toy datasets (e.g. `sklearn.datasets.load_iris`, `torchvision.datasets.MNIST`).
   * Short training runs (few epochs, small subsets).
   * Must **print key results**: metrics, paths of saved models, etc.
3. Calls `run_python(code=...)` **exactly once**.
4. Reads the tool result (STATUS/STDOUT/STDERR) and summarizes what happened.
5. Writes a short JSON-like feedback / summary into `STATE_FEEDBACK`.

This agent can be invoked **independently** for code tasks, e.g.:

> “Train a PyTorch MLP on MNIST for 1 epoch, print accuracy, and save the model to `mnist_mlp.pth`.”

With the judge + loop, it will attempt to fix errors in subsequent iterations.

#### 5.3.3 EngineerJudge (LlmAgent)

* **Tools**: `exit_loop` via `FunctionTool`
* Sees:

  * The original task
  * The engineer’s latest message
  * The `run_python` result
  * Previous feedback (`STATE_FEEDBACK`)
* Logic:

  * **Task is satisfied** if:

    * Correct dataset / library family used.
    * Script executed without unhandled exceptions (`STATUS: OK`).
    * Key requested outputs visible in `STDOUT` (e.g. “Accuracy: 0.99”, “Model saved to ...”).
  * If satisfied:

    * Calls `exit_loop` and **outputs nothing else**.
  * If not satisfied:

    * Returns a JSON-like feedback object in `STATE_FEEDBACK` with:

      * `"status"`: `"RETRY"` or `"WAITING"`
      * `"reason"`: short cause
      * `"hints"`: bullet hints for next attempt

#### 5.3.4 EngineerLoop (LoopAgent)

* Wraps `[ML_Engineer, EngineerJudge]` with `max_iterations` (e.g., 3).
* Performs iterative execution and bug-fixing until:

  * Judge calls `exit_loop`, or
  * `max_iterations` is reached.

---
### 5.4 `ml_team`: End-to-End Multi-Agent Orchestration

**Role:**
`ml_team` is the “all-hands” demo agent that wires the previously defined specialists into a single workflow. It shows how a **planner**, **researchers**, and an **ML engineer + judge** can cooperate on a user task from natural-language request to trained model and saved artifact.

Concretely, `ml_team` is the agent you use when you want to see the whole system work together on a *single ML job* (e.g. *“get 100% on Iris and save the model”*), rather than testing individual agents in isolation.

---

#### 5.4.1 Responsibilities & Flow

At a high level, `ml_team` orchestrates:

1. **Project planning (HITL-styled):**

   * User sends a natural language task (e.g. *“Find how to get 100% accuracy on Iris from the internet, implement that approach, train and save the model.”*).
   * `project_planner` interprets the request as a **PLANNING_REQUEST**, drafts:

     * High-level idea,
     * Step-by-step experiment plan,
     * Models/approaches to try,
     * Implementation notes for the team.
   * For the capstone demo, HITL is **stubbed**: the planner ends with `HITL_STATUS: PLAN_APPROVED` so downstream agents can proceed without waiting for an actual human response.

2. **Research fan-out:**

   * Once the plan is “approved”, `ml_team` triggers the **research side**:

     * `ml_researcher` / `WebResearchAgent` uses Google Search to find:

       * Recent papers,
       * Blog posts / docs,
       * GitHub repos, example code, metrics and baselines.
     * `KaggleResearchAgent` (via MCP Kaggle server) searches for:

       * Relevant Kaggle datasets,
       * Competitions,
       * Notebooks that demonstrate similar tasks.
     * `ResearchBrain` fuses their outputs into:

       * `FINAL_SUMMARY`: a concise narrative for implementation,
       * `IMPLEMENTATION_SUGGESTIONS`: concrete bullets (which repo/model/dataset/metric to use).

3. **Implementation & execution (ML Engineer + Judge loop):**

   * `ml_team` hands the *implementation suggestions* to the **ML Engineer loop**:

     * `ML_Engineer` writes a **single executable Python script** based on the research summary (e.g. Iris SVM / Decision Tree / Naive Bayes, etc.).
     * Code is executed via the unsafe `run_python` tool (local environment).
     * `EngineerJudge` inspects:

       * STATUS / stdout / stderr,
       * Whether the task was formally satisfied (correct dataset, no exceptions, metrics printed, model saved).
     * If the judge is satisfied, it calls `exit_loop` and the engineer stops iterating.
     * If not, the judge returns structured feedback (`{"status": "RETRY", "reason": ..., "hints": [...]}`), and the engineer retries with a corrected script.

4. **Final user-facing report:**

   * After a successful run, `ml_team` can:

     * Surface the **final metrics** (e.g. 100% Iris accuracy, F1 = 1.0),
     * Confirm that the model has been saved (e.g. `iris_model.joblib`),
     * Echo a short explanation of *what approach was chosen and why* (e.g. “SVM with linear kernel chosen based on web + Kaggle research”).

This mirrors a miniature ML team: planner → research → engineering → evaluation → report.

---
#### 5.4.2 Example End-to-End Scenario

A typical `ml_team` demo scenario used in the project:

> “I need you to find on the internet (Google search or Kaggle) how to score 100% on the IRIS dataset, implement that single approach, train the model, print metrics, and save the trained model locally.”

The orchestrated behavior is:

1. **Planner** designs the Iris experiment (research, pick one strong approach, implement, train, evaluate, save).
2. **Researchers** discover that simple classifiers (Decision Trees, SVMs, Naive Bayes, etc.) routinely reach 100% accuracy on Iris, and propose a concrete recipe using standard `scikit-learn` APIs.
3. **ResearchBrain** condenses this into a small set of **implementation bullets**:

   * Load Iris from `sklearn.datasets`,
   * Instantiate e.g. `DecisionTreeClassifier` or `SVC(kernel="linear")`,
   * Train, print accuracy/precision/recall/F1,
   * Save to `iris_model.joblib` with `joblib.dump`.
4. **ML Engineer + Judge loop**:

   * Engineer writes and executes a script that:

     * Loads Iris,
     * Trains the chosen model,
     * Prints metrics,
     * Saves the model file.
   * Judge verifies:

     * Dataset is Iris,
     * Script runs without exceptions,
     * Metric(s) and “model saved” message are printed.
   * On success, the loop exits.
5. **User** can see:

   * The *research reasoning* behind the chosen approach,
   * The *actual metrics* and the fact that a real artifact was written to disk.
   * The *saved model* locally on their machine

---

#### 5.4.3 How to Run `ml_team`

You can run the orchestrated demo via:

* **Web UI (recommended for inspection):**

  ```bash
  adk web ml_team
  ```

  Then in the browser, send a message like:

  ```text
  I need you to find on the internet (Google search or Kaggle) how to score 100% on the IRIS dataset, implement that single approach, train the model, print metrics, and save the trained model locally.
  ```

  You’ll see:

  * Planner’s plan and auto-approval,
  * Research agents doing web/Kaggle lookups,
  * ResearchBrain’s integrated summary,
  * ML Engineer + Judge loop executing code and iterating if needed,
  * Final metrics and confirmation that the model has been saved.

* **Individual agents still available:**

  Even with `ml_team` in place, you can still invoke each agent separately for more fine-grained experiments:

  * Use only **research**:

    * Ask the research app (e.g. `ml_researcher`) directly:

      > “Summarize recent SOTA OCR models and typical datasets/metrics.”
  * Use only **engineering**:

    * Ask the engineer app (e.g. `ml_engineer`) to:

      > “Write and run code that trains a LogisticRegression classifier on Iris, prints test accuracy and F1, and saves the model to `iris_logreg.joblib`.”
  * Use only **planning**:

    * Ask `ml_project_planner`:

      > “Design an experiment plan to benchmark classic ML vs CNN vs Vision Transformer on MNIST.”

`ml_team` is therefore both a **full-system demo** and an example of how to glue specialized agents together into a single, user-facing experience.

---

## 6. Observability & Telemetry

### 6.1 ADK LoggingPlugin

* Uses `google.adk.plugins.logging_plugin.LoggingPlugin` to log:

  * User messages
  * Agent starts / finishes
  * Tool calls
  * LLM requests

* When using a debug runner like `ml_researcher/debug_runner.py`, logs are printed with helpful prefixes, e.g.:

  ```text
  [logging_plugin] 🚀 USER MESSAGE RECEIVED
  [logging_plugin] 🤖 AGENT STARTING
  ```

### 6.2 InvocationMetricsPlugin (Custom)

* Implements a custom plugin (subclass of `BasePlugin`) that:

  * Counts:

    * Agent invocations
    * Tool invocations
    * LLM requests
  * Logs aggregate metrics such as:

    ```text
    [Metrics] Agent 'WebResearchAgent' invoked. Total agent calls: 2
    [Metrics] Tool call count: 5
    ```

Used for quick sanity checks on how “busy” the system is.

### 6.3 AgentOps Integration

At the top of each app’s entry script you’ll see something like:

```python
import agentops
import os
from dotenv import load_dotenv

load_dotenv()

agentops.init(
    api_key=os.getenv("AGENTOPS_API_KEY"),
    trace_name="ml-co-pilot",
)
```

This automatically:

* Tracks **latency**, **spend**, and **tool usage**
* Provides a web dashboard to inspect traces and resource usage for your ADK agents.

---

## 7. Example Usage Scenarios

### 7.1 Full-Team Example: Iris 100% Accuracy (End-to-End)

1. Run `adk web .` and select the **`project_planner`** app.

2. Ask:

   > “I need you to find on the internet (Google search or Kaggle) how to score 100% on the IRIS dataset, implement that single approach, train the model, print metrics, and save the trained model locally.”

3. Planner:

   * Produces a full experiment plan.
   * Outputs `HITL_STATUS: PLAN_APPROVED` (stubbed for demo).

4. Researcher app (`ml_researcher`) is used to:

   * Let WebResearchAgent find SOTA / common high-accuracy approaches.
   * Let KaggleResearchAgent look for Kaggle baselines or notebooks.
   * Let ResearchBrain merge into a final summary plus implementation suggestions (e.g., tuned SVM or Decision Tree).

5. Engineer app (`ml_engineer`) is then used to:

   * Implement the chosen approach,
   * Train it,
   * Print metrics (ideally 100% accuracy),
   * Save a model artifact like `iris_model.joblib`.

6. The user (or the planner) can then summarize results and direct further experiments.

> In a more integrated setup, this orchestration could be automated via a higher-level controller agent or an external orchestrator calling each app in sequence with shared state.

---

### 7.2 Standalone Research Brain: OCR SOTA Summary

You do **not** need the full team for research tasks.

1. Run `adk web ml_researcher`.

2. Use the Web UI to send a research task (e.g., via `STATE_RESEARCH_TASK` or plain message):

   > “Summarize recent SOTA OCR models and relevant datasets, with a focus on document OCR.”

3. The orchestrated root agent will:

   * Use WebResearchAgent to search for TrOCR, Donut, LayoutLMv3, etc.
   * Use KaggleResearchAgent to find any relevant OCR / document datasets or competitions.
   * Use ResearchBrain to produce a final answer with:

     * `FINAL_SUMMARY` – models, trends, and key papers.
     * `IMPLEMENTATION_SUGGESTIONS` – which repos to clone, which metrics, etc.

You can also develop a pattern where you **manually** set `WEB_NOTES` / `KAGGLE_NOTES` and invoke only **ResearchBrain** for aggregation.

---

### 7.3 Standalone ML Engineer + Judge: Code Execution Tasks

For pure coding / experimentation:

1. Run `adk web ml_engineer`.

2. Ask:

   > “Write and execute PyTorch code that trains a small MLP on MNIST for 1 epoch, prints test accuracy, and saves the model to `mnist_mlp.pth`.”

3. The ML_Engineer agent will:

   * Draft a full Python script.
   * Call `run_python` with that script.

4. EngineerJudge will:

   * Inspect the STATUS/STDOUT/STDERR.
   * If something fails (e.g., import error, missing dataset), it will provide hints in `STATE_FEEDBACK`.
   * The LoopAgent will allow another attempt with fixes.

This is useful even without planner or researcher: you essentially get a **“code skeleton + executor + strict QA”** loop for ML scripts.

---

### 7.4 Planner Only: Designing Experiment Roadmaps

You can run **only** the `project_planner` app if you just want experiment designs and human-in-the-loop-ish plans.

Examples:

* “Design an experiment plan to compare TrOCR vs Donut vs LayoutLMv3 for invoice OCR.”
* “Propose ablations and metrics to evaluate different metric-learning losses for a face recognition model.”

The planner will:

* Generate high-level ideas,
* Concrete experiment lists,
* Explicit “hand-offs” for hypothetical Research/Engineer agents, and
* End with a HITL prompt.

Even with HITL stubbed to `PLAN_APPROVED` in the final line (for the demo), the structure and content are still valid for manual review.

---

## 8. Mapping to Capstone Requirements

The project demonstrates:

1. **Tools**

   * Built-in: `google_search`
   * Custom: `run_python` (unsafe local code executor)
   * MCP: Kaggle MCP server via `MCPToolset` / `mcp-remote`
   * Loop control: `exit_loop` tool used by the judge

2. **Long-running Operations**

   * `LoopAgent` for ML_Engineer + EngineerJudge:

     * State across iterations (`STATE_FEEDBACK`)
     * Early termination via `exit_loop`

3. **Sessions & Memory**

   * ADK’s session & state management:

     * `STATE_RESEARCH_TASK`, `STATE_WEB_NOTES`, `STATE_KAGGLE_NOTES`, `STATE_FINAL_SUMMARY`, `STATE_FEEDBACK`, etc.
   * Allows independent and chained usage of agents with shared contextual state.

4. **Observability**

   * ADK `LoggingPlugin` for structured event logs
   * Custom `InvocationMetricsPlugin`
   * **AgentOps** for end-to-end traces, latency and cost tracking

---

## 9. Limitations & Future Work

* **HITL**:

  * The scaffolding is implemented (planner formats, HITL statuses).
  * For robustness during evaluation, HITL is **stubbed to auto-approve** so downstream agents always see `PLAN_APPROVED`.
  * A future iteration should:

    * Store HITL state explicitly in session state,
    * Gate other agents strictly on *actual* user responses.

* **Safety / Sandbox**:

  * `run_python` executes unrestricted Python in the current environment.
  * For real-world use, replace with:

    * Docker or K8s-based sandbox,
    * Resource limits,
    * Possibly a job queue / worker model.

* **Long-Term Memory**:

  * Currently relies on basic ADK session state.
  * Integrating `MemoryBank` or external stores (BigQuery, Firestore, etc.) would enable persistent cross-session knowledge.  


* **Towards a Full ML Department Co-pilot**

  This repo implements the **core spine** (project planner, research cluster, ML engineer + judge). The longer-term vision is to grow it into a **virtual ML department** that a single expert can drive:

  * **Project Manager Agent** – front-door interface for non-expert users in natural language, clarifying business goals, constraints, and success metrics, then translating them into technical briefs.
  * **ML Team Lead Agent** – refines technical specs, designs project roadmaps, decides which downstream agents/tools are needed (research-focused vs production-focused projects), and keeps metrics and “definition of done” aligned with the user.
  * **ML Researcher Agent** – a more powerful version of the current researcher: systematically mines Arxiv, GitHub, Kaggle, etc., but writes only *task-relevant slices* (e.g., just the new loss function from a paper, or just the model definition block from a repo) into a shared memory bank.
  * **Data Engineer Agent** – turns discovered datasets (zips, APIs, Kaggle datasets) into standardized, toggle-able dataset loaders and pipelines; manages splits (train/val/test) and schema normalization across experiments.
  * **Data Analyst Agent** – analyzes datasets and experiment outputs, surfaces data issues (imbalance, leakage, noise), suggests cleaning or merging, and writes human-readable dataset reports to the memory bank and back to the user.
  * **ML Engineer Agent (Repo-level)** – evolves from “single script engineer” into a repository-level maintainer: adds models, losses, configs, and training utilities requested by Researcher / Data Engineer, and waits for HITL sign-off before major refactors.
  * **ML Researcher Agent** – takes Recon + data summaries, proposes hypotheses and experiment matrices, self-critiques them, and negotiates experiment priorities with the Team Lead and user (HITL).
  * **Experiment Runner / Orchestrator Agent** – checks experiment configs, estimates runtime / cost (GPU hours, cloud spend), deduplicates similar runs, and asks the user to approve expensive jobs before executing and monitoring them (watching logs, early-stopping obviously broken runs).
  * **Innovation Agent** – periodically scans memory, failures, bottlenecks, and “TODOs” across agents to propose system-level changes: new tools, new data abstractions, better experiment templates, or self-healing for infinite loops and dead ends.

  Future iterations can gradually implement these roles on top of the current core, starting with:

  * Moving from **single-script execution** to a persistent **ML repo** that agents collaboratively extend.
  * Wiring a shared **MemoryBank** so Data Engineer, Researcher, and Innovation can build on each other’s work across tasks and time.
  * Targeting harder end-to-end tasks, like:

    * “Build a repo with comprehensive benchmarks on IAM Handwritten Forms Dataset using classic ML, CNNs, and vision transformers.”
    * “Tackle this Kaggle competition URL end-to-end: data ingestion → baselines → SOTA re-implementation → result packaging.”

---

## 10. Conclusion

**ML Co-pilot** shows how to build a small, but genuinely useful, **agentic ML workflow** with Google ADK:

* A Project Planner acting like a senior ML TL,
* A multi-source Researcher cluster (Web + Kaggle + Brain),
* An ML Engineer that generates and executes code with an LLM-as-a-judge safety net,
* And observability across all of it via Logging, custom metrics, and AgentOps.

You can use the agents:

* Individually (for research, planning, or code execution), or
* Together as a coordinated pipeline solving non-trivial ML tasks end-to-end.

This repository is intended both as a **capstone submission** and as a **template** for future, more complex agentic ML systems.

```
::contentReference[oaicite:0]{index=0}
```
