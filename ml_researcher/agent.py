from ml_common.observability import init_agentops

# initialize AgentOps for this process / agent tree
init_agentops(trace_name="ml_researcher")

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.models import Gemini
from google.adk.tools import google_search
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)

# ====== Shared state keys ======
STATE_RESEARCH_TASK   = "research_task"
STATE_WEB_NOTES       = "web_notes"
STATE_KAGGLE_NOTES    = "kaggle_notes"
STATE_FINAL_SUMMARY   = "final_summary"

# ====== Kaggle MCP toolset (MCP-only tools) ======
kaggle_mcp = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "mcp-remote",
                "https://www.kaggle.com/mcp",
            ],
        ),
        timeout=60,
    )
)

# ====== 1) WebResearchAgent: google_search ONLY ======
web_researcher = LlmAgent(
    name="WebResearchAgent",
    model=Gemini(model="gemini-2.0-flash"),
    tools=[google_search],
    instruction=f"""
You are a web research agent focused on ML research.

Task:
{{{{ {STATE_RESEARCH_TASK} }}}}

You have access to the `google_search` tool.
Use it to:
- Find recent papers (including Arxiv via queries like: "site:arxiv.org <topic>").
- Find blog posts, docs, GitHub repos, benchmarks, implementation details.

Do **not** mention tools or `google_search` explicitly in your final answer.

Your output MUST follow this exact structure and headings:

WEB_NOTES:
MODELS:
- Name (Year) – very short description; paper title and venue if you can find it. Note clearly if you are uncertain.
DATASETS:
- Name – type of data – typical use; note if it is synthetic or real-world; mention any common benchmark status.
IMPLEMENTATIONS:
- Library / framework – link or repo slug – 1–2 words about maturity (official / community / example).
METRICS:
- Metric name – what it measures – typical ranges or SOTA values if you can find them.

WEB_SUMMARY:
- 3–8 sentences summarizing the most important items **for an ML engineer implementing or extending systems**.
- Focus on: which models matter, which datasets are common, which repos are likely to be cloned first.

Additional rules:

1. Prefer concrete references:
   - Give paper titles + Arxiv IDs when possible.
   - Give GitHub org/repo names (e.g., "microsoft/trocr") and/or Hugging Face model IDs.
2. If you are not sure about something (e.g., a model name like "OCR-GPT"), **explicitly mark it as uncertain** or general trend instead of presenting it as a specific paper.
3. Do NOT wrap your response in JSON or any other structure.
4. Do NOT mention Kaggle here.
""",
    output_key=STATE_WEB_NOTES,
)

# ====== 2) KaggleResearchAgent: Kaggle MCP ONLY ======
kaggle_researcher = LlmAgent(
    name="KaggleResearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    tools=[kaggle_mcp],
    instruction=f"""
You are a Kaggle-focused research agent.

Task:
{{{{ {STATE_RESEARCH_TASK} }}}}

You have access to the Kaggle MCP toolset.
Use it to:
- Search for relevant Kaggle datasets.
- Search for relevant Kaggle competitions.
- Optionally, identify useful Kaggle notebooks if tools allow.

Do **not** mention tools or MCP explicitly in your final answer.

Your output MUST follow this exact structure and headings:

KAGGLE_NOTES:
DATASETS:
- Dataset name (slug if available) – short description – target column – approximate size or timespan – why it might be useful.
COMPETITIONS:
- Competition name (slug if available) – target / task – main metric – why it's relevant.
NOTEBOOKS:
- Notebook title or short ID – what it demonstrates – why it's useful as a starting point.

KAGGLE_SUMMARY:
- 2–6 sentences summarizing the **most practically useful Kaggle resources** or,
  if none are found, how a user could still leverage Kaggle (search terms, generic OCR or document datasets, etc.).

If tools return nothing, fail, or you cannot identify specific resources:
- Still fill the sections with at least one bullet each, explaining that:
  - No clearly-targeted resources were found, OR
  - Access was limited,
  - And provide suggested search queries like "ocr dataset", "document understanding", "handwritten text recognition" that a human could try.

Constraints:
- Do NOT call or refer to `google_search`.
- Do NOT just say "nothing relevant"; always provide at least some search handles and guidance.
- Do NOT wrap your response in JSON or any other structure.
""",
    output_key=STATE_KAGGLE_NOTES,
)

# ====== 3) ResearchBrain: merges web + Kaggle into final answer ======
brain_agent = LlmAgent(
    name="ResearchBrain",
    model=Gemini(model="gemini-2.0-flash"),
    tools=[],
    instruction=f"""
You are the coordinator that merges all research into a concise, actionable answer.

Inputs:
- Task: {{{{{ {STATE_RESEARCH_TASK} }}}}}
- Web research (notes + summary): {{{{{ {STATE_WEB_NOTES} }}}}}
- Kaggle research (notes + summary): {{{{{ {STATE_KAGGLE_NOTES} }}}}}

Your response is consumed by an **ML_Engineer agent** that will implement models and experiments.

Your output MUST have exactly these two top-level sections and headings:

FINAL_SUMMARY:
- 1–3 paragraphs.
- High-level explanation targeted at an ML engineer who will implement or experiment.
- Integrate BOTH web and Kaggle findings where useful.
- Name the most important models/papers/datasets/competitions and their *role* (baseline vs SOTA vs niche).

IMPLEMENTATION_SUGGESTIONS:
- A bullet list (5–15 bullets) of concrete next steps, tuned for someone who will write code.
- Examples of good bullets:
  - "Clone repo X and run the default training script on dataset Y."
  - "Fine-tune pre-trained model Z from Hugging Face on dataset Y with metric W."
  - "Use Kaggle dataset D as a proxy if no domain-specific data is available."
  - "Compare baseline model A with SOTA model B using metrics M, N."
- When possible, reference:
  - specific repos (org/repo),
  - HF model IDs,
  - Kaggle dataset or competition slugs,
  - concrete metrics and target ranges.

Rules:
- Do NOT mention internal tools, agents, or state keys.
- Do NOT wrap your response in JSON or any other machine format.
- Be opinionated and practical: assume the reader has PyTorch / sklearn / HF / basic Kaggle skills.
""",
    output_key=STATE_FINAL_SUMMARY,
)

# ====== Root agent: orchestrates Web -> Kaggle -> Brain ======
root_agent = LoopAgent(
    name="ResearchOrchestrator",
    sub_agents=[
        web_researcher,
        kaggle_researcher,
        brain_agent,
    ],
    max_iterations=1,
)
