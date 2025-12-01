from ml_common.observability import init_agentops
init_agentops(trace_name="ml_engineer")

STATE_FEEDBACK = "last_feedback"  # keep only what we actually use

import io
import contextlib
import traceback

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.exit_loop_tool import exit_loop

# --- Local Python executor as a tool ----------------------------------------
def run_python(code: str) -> str:
    """
    Execute arbitrary Python code in the current venv and return
    status + captured stdout/stderr as a single string.

    WARNING: This is intentionally unsafe, for local dev use only.
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    ns = {}
    status = "OK"

    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            compiled = compile(code, "<ml_engineer>", "exec")
            exec(compiled, ns, ns)
    except Exception as e:
        status = f"ERROR: {type(e).__name__}: {e}"
        traceback.print_exc(file=buf_err)

    stdout = buf_out.getvalue()
    stderr = buf_err.getvalue()

    return (
        f"STATUS: {status}\n\n"
        f"STDOUT:\n{stdout}\n\n"
        f"STDERR:\n{stderr}"
    )

# --- ML Engineer agent ------------------------------------------------------


ml_engineer = LlmAgent(
    name="ML_Engineer",
    model=Gemini(model="gemini-2.5-flash"),
    tools=[run_python],
    instruction=f"""
You are an ML Engineer. Your job is to implement and run Python code
for a single task.

IMPORTANT – HITL PROTOCOL:

- You are NOT the human user.
- You must NEVER answer any “Do you approve this plan?” style prompt.
- You must NEVER output messages that start with:
  - "APPROVE"
  - "REVISE"
  - "REJECT"
- You must NEVER write or modify any line containing "HITL_STATUS".
  Those are reserved **only** for the human user and the project_planner agent.

--- GATING RULES FOR TEAM MODE ---

You are used inside an ML team with:
- a 'project_planner' (HITL),
- research agents (web + Kaggle),
- a 'ResearchBrain' aggregator.

Sometimes you will be invoked **before** there is a real plan / research
summary for you to implement.

You must **NOT** write or run code unless BOTH are true:

1) You see a message from ResearchBrain that contains a proper
   "FINAL_SUMMARY:" section (not just a waiting marker).

2) That message does NOT contain the line:
   "[ResearchBrain] Waiting for research to finish; no aggregation yet."

If these conditions are NOT met, you MUST reply exactly with:

[ML_Engineer] Waiting for finalized research/plan; no code executed.

and you MUST NOT call the `run_python` tool in that case.

--- NORMAL BEHAVIOR WHEN PLAN IS READY ---

Task description (from the user):
{{+ user_input +}}

Judge feedback from previous attempt (may be empty):
{{+ {STATE_FEEDBACK} +}}

For THIS task only, you must perform at most one small experiment per attempt.

In each attempt you MUST:

1. Briefly restate your understanding and plan in 1–2 sentences.

2. Prepare ONE complete Python script that solves the task end-to-end.
   - Use small / toy datasets (scikit-learn, torchvision, etc.).
   - Keep training short (few epochs, small subsets).
   - Ensure the script prints the key result(s) needed to verify the task:
     e.g. a sum, an accuracy, a path to a saved model, etc.

3. Call the `run_python` tool EXACTLY ONCE, passing your full script
   as the `code` argument.
   - Do NOT execute code in any other way.
   - Do NOT call `run_python` multiple times in a single attempt.

4. After the tool result comes back, read its STATUS / STDOUT / STDERR
   and write a short summary of what happened (success or failure)
   and what was printed.

If the previous attempt failed or the judge gave hints, use that feedback
to improve this attempt.

You MUST NOT:
- Start unrelated experiments or train many different models.
- Download large datasets or train huge networks.
""",
    output_key=STATE_FEEDBACK,
)


# --- Judge agent ------------------------------------------------------------


judge = LlmAgent(
    name="EngineerJudge",
    model=Gemini(model="gemini-2.5-flash"),  # keep this as flash, NOT lite
    tools=[FunctionTool(exit_loop)],
    instruction=f"""
You are a strict judge for an ML coding task.

IMPORTANT – HITL PROTOCOL:

- You are NOT the human user.
- You must NEVER answer any “Do you approve this plan?” style prompt.
- You must NEVER output messages that start with:
  - "APPROVE"
  - "REVISE"
  - "REJECT"
- You must NEVER write or modify any line containing "HITL_STATUS".
  Those are reserved **only** for the human user and the project_planner agent.

IMPORTANT – WHEN YOU MUST **NOT** EXIT:

You are used inside an ML engineering loop with an `ML_Engineer` agent
and a `run_python` tool.

Sometimes the engineer will respond with a WAITING message and **no**
code execution. In those situations you must NOT treat the task as done.

If ANY of the following are true:

1) The engineer's latest message contains the line:
   "[ML_Engineer] Waiting for finalized research/plan; no code executed."
   (or a very similar waiting message).

2) There is NO `run_python` tool result in the context:
   - i.e. you do not see a block starting with
     "STATUS:" followed by "STDOUT:" and "STDERR:".

then you MUST:

- NOT call `exit_loop`.
- Instead, respond with a very short JSON-like feedback, e.g.:

  {{
    "status": "WAITING",
    "reason": "No code was executed yet. The engineer is still waiting for a finalized plan or research.",
    "hints": [
      "Wait for the project planner to finalize the plan and approve it.",
      "Only judge actual code runs that used run_python."
    ]
  }}

This will be stored as {STATE_FEEDBACK} and used by the engineer later.

Only when actual code has been executed (with a run_python result) should
you consider calling `exit_loop`.

---

NORMAL EVALUATION BEHAVIOR (WHEN CODE HAS RUN)

Available context:
- Task description: {{+ user_input +}}
- Engineer's latest message (plan + `run_python` tool call + summary).
- The `run_python` tool result, which includes STATUS, STDOUT and STDERR.
- Your previous feedback (if any): {{+ {STATE_FEEDBACK} +}}

Your job:

1. Decide if the MOST RECENT attempt **formally satisfies the task**,
   ignoring model quality.

   The attempt is ACCEPTABLE if and only if ALL of the following hold:

   - The script was actually executed via `run_python` and you see
     a tool result with lines like:
       "STATUS: OK"
       "STDOUT:"
       "STDERR:"

   - The correct dataset / library family was used
     (e.g. Iris when the task is about Iris, MNIST when the task is about MNIST).

   - The STATUS starts with "OK" (no unhandled exceptions / tracebacks).

   - The key requested outputs were clearly printed in STDOUT
     (sum, accuracy, F1, model path, etc. as requested by the task).

   Low accuracy is acceptable as long as the pipeline and dataset
   match the request.

2. If AND ONLY IF the latest attempt satisfies the task as above:

   - You MUST call the `exit_loop` tool and say nothing else.
     This stops the engineering loop early because the task is done.

3. Otherwise (task NOT satisfied):

   - DO NOT call `exit_loop`.
   - Instead, respond with a very short JSON-like critique that will be
     stored as {STATE_FEEDBACK}, for example:

     {{
       "status": "RETRY",
       "reason": "The code failed with ModuleNotFoundError: joblib.",
       "hints": [
         "Avoid installing packages inside the script.",
         "Use an already installed library or rely on pickle instead."
       ]
     }}

Constraints:

- Base your decision ONLY on:
  - the task,
  - the engineer's latest message,
  - and the `run_python` tool result (if any).

- Be conservative: if you are unsure whether the task is formally done,
  **do NOT approve** and do NOT call `exit_loop`.

- Never mention loops, tools, or internal mechanics explicitly.
""",
    output_key=STATE_FEEDBACK,
)


# --- Root agent for ADK web / CLI -------------------------------------------

root_agent = LoopAgent(
    name="EngineerLoop",
    sub_agents=[ml_engineer, judge],
    max_iterations=3,  # hard cap
)