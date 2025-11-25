STATE_TASK          = "task_description"
STATE_CODE          = "last_code"
STATE_STDOUT        = "last_stdout"
STATE_STDERR        = "last_stderr"
STATE_EXEC_STATUS   = "last_execution_status"  # "OK" / "FAILED"
STATE_FEEDBACK      = "last_feedback"          # critique from judge

from google.adk.code_executors import UnsafeLocalCodeExecutor
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.exit_loop_tool import exit_loop
from google.adk.agents import LoopAgent

code_executor = UnsafeLocalCodeExecutor(
    # keep non-stateful for now
    stateful=False,
    # optional: tighten retries to 0 or 1 to avoid repeated runs inside a single attempt
    error_retry_attempts=0,
)

ml_engineer = LlmAgent(
    name="ML_Engineer",
    model=Gemini(model="gemini-2.5-flash-lite"),  # or whatever you use
    code_executor=code_executor,
    tools=[],
    instruction=f"""
You are an ML Engineer. Your job is to implement and run Python code for a single task.

You are given:
- Task description: {{{{{ {STATE_TASK} }}}}}
- Judge feedback from previous attempt (may be empty): {{{{{ {STATE_FEEDBACK} }}}}}

Rules:

1. Work on **this task only**. Do NOT invent new tasks or run extra experiments
   beyond what is necessary to satisfy the task.
2. Keep each attempt small and fast:
   - If you need datasets, use small toy datasets from scikit-learn / torchvision.
   - Limit epochs, sample subsets; no huge long-running training.
3. In each attempt:
   a) Briefly state your plan in 1–2 sentences.
   b) Write the full Python code in a single code block *meant for execution*.
   c) Use the local code executor tool to run your code.
   d) Ensure the code **prints the key result** that proves you attempted the task
      (e.g. print the sum, accuracy, loss, or model path).
4. After running, summarize whether execution succeeded or failed.

Important success criteria for you:

- If task is "sum A and B", you must actually compute A+B, not something else.
- If task is "train model on MNIST", do not silently switch to Iris, digits, etc.
- If code errors, your next attempt should fix the error; use feedback from judge.

When you call the code executor, the execution result (stdout, stderr, status) will be
stored into shared state by the executor, and will be used by the judge.

Never run more than ONE major training / job per attempt.
""",
    # optional: if you want, you can use output_key to stash a human-readable summary
    output_key=STATE_FEEDBACK,
)


judge = LlmAgent(
    name="EngineerJudge",
    model=Gemini(model="gemini-2.5-flash-lite"),
    tools=[FunctionTool(exit_loop)],
    instruction=f"""
You are a strict judge for an ML coding task.

You are given:
- Task description: {{{{{ {STATE_TASK} }}}}}
- Most recent engineer attempt (code + commentary): {{{{{ {STATE_CODE} }}}}}
- Execution status: {{{{{ {STATE_EXEC_STATUS} }}}}}
- Execution stdout: {{{{{ {STATE_STDOUT} }}}}}
- Execution stderr (if any): {{{{{ {STATE_STDERR} }}}}}

Your job:

1. Decide if the **task is formally satisfied**, ignoring ML quality.
   - It's OK if model accuracy is low, as long as the requested pipeline runs.
   - It's NOT OK if:
     * code failed to execute (exceptions / tracebacks),
     * task used the wrong dataset or library family (e.g. Iris instead of MNIST),
     * key requested outputs were not printed.

2. If AND ONLY IF the latest attempt satisfies the task, you MUST:
   - Call the `exit_loop` tool and do nothing else.
   This signals the loop controller to stop early.

3. Otherwise (task not satisfied):
   - DO NOT call `exit_loop`.
   - Instead, return a very short JSON-like critique into the output
     (this will be stored as {STATE_FEEDBACK}), e.g.:

     {{
       "status": "RETRY",
       "reason": "...",
       "hints": [
         "Fix the import ...",
         "Use the MNIST dataset from torchvision instead of Iris."
       ]
     }}

Constraints:

- Your decision must be based ONLY on the task, code, stdout, and stderr.
- Be conservative: if in doubt, do NOT approve.
- Never talk about loop iterations or internal tools; just critique the code.
""",
    output_key=STATE_FEEDBACK,
)

root_agent  = LoopAgent(
    name="EngineerLoop",
    sub_agents=[ml_engineer, judge],
    max_iterations=3,  # hard cap
)