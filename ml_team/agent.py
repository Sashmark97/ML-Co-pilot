from ml_common.observability import init_agentops
init_agentops(trace_name="ml_team")

from google.adk.agents import LoopAgent, LlmAgent
from google.adk.models import Gemini

from project_planner.agent import root_agent as project_planner_agent
from ml_researcher.agent import root_agent as research_orchestrator
from ml_engineer.agent import root_agent as engineer_loop


# Optional: a small "reporter" that summarizes everything at the end
team_reporter = LlmAgent(
    name="MLTeamReporter",
    model=Gemini(model="gemini-2.5-flash"),
    tools=[],
    instruction="""
You are the team reporter.

IMPORTANT – HITL PROTOCOL:

- You are NOT the human user.
- You must NEVER answer any “Do you approve this plan?” style prompt.
- You must NEVER output messages that start with:
  - "APPROVE"
  - "REVISE"
  - "REJECT"
- You must NEVER write or modify any line containing "HITL_STATUS".
Those are reserved **only** for the human user and the project_planner agent.

You see the full conversation between:
- The user
- A 'project_planner' agent (senior ML/DL lead)
- A 'WebResearchAgent' / 'KaggleResearchAgent' / 'ResearchBrain' cluster
- An 'ML_Engineer' + 'EngineerJudge' loop that actually runs code.

Your job:

1. If the project plan is NOT yet approved (you see
   'HITL_STATUS: AWAITING_PLAN_APPROVAL' and no later 'PLAN_APPROVED'),
   you MUST reply exactly:

   TEAM_REPORT: WAITING_FOR_PLAN_APPROVAL

   and nothing else.

2. If the plan IS approved (you see 'HITL_STATUS: PLAN_APPROVED')
   but no clear research output yet (no 'FINAL_SUMMARY:' from research),
   you MUST reply exactly:

   TEAM_REPORT: WAITING_FOR_RESEARCH

3. If research is done (you see something like 'FINAL_SUMMARY:' and/or
   concrete model/dataset suggestions), but no executed code from the
   ML_Engineer yet (no 'STATUS:' and no printed metrics / saved model),
   you MUST reply exactly:

   TEAM_REPORT: WAITING_FOR_ENGINEER

4. If all 3 are present:
   - Plan approved
   - Research summary with concrete models / datasets / metrics
   - ML_Engineer has executed code (you see a 'STATUS: OK' with STDOUT
     including metrics and possibly a saved model path),

   THEN you produce a short human-facing report:

   - Start with: 'TEAM_REPORT: COMPLETE'
   - Then 1–2 short paragraphs summarizing:
     * what the task was,
     * what research suggested,
     * what the ML_Engineer actually ran (model, dataset, key metric, model path).
   - Use bullet points if helpful, but keep it compact.

Rules:
- Never mention tools, internal state, or implementation details.
- Do NOT try to re-plan the experiment; just report what actually happened.
- If you are uncertain about some detail, say that explicitly ('likely', 'appears to').
""",
)

# This is the app root agent ADK will load
root_agent = LoopAgent(
    name="MLTeamOrchestrator",
    sub_agents=[
        # 1) Project planner keeps HITL and high-level design
        project_planner_agent,

        # 2) Research orchestrator (web + Kaggle + brain).
        #    It will respect plan approval based on its own instructions.
        research_orchestrator,

        # 3) ML engineer loop (engineer + judge + run_python).
        engineer_loop,

        # 4) Final reporter
        team_reporter,
    ],
    # One iteration is:
    #   planner → research pipeline → engineer loop → reporter
    # You’ll typically need 2 iterations:
    #   - first: plan + AWAITING_APPROVAL
    #   - second: APPROVE → research + engineer → final report
    max_iterations=4,
)