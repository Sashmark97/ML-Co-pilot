import asyncio
import logging
import os

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

from ml_researcher.agent import root_agent


# ---- Basic env loading (mimic ADK CLI behavior enough for local debug) ----
def load_env():
    """
    Minimal .env loader for local debug.
    ADK CLI does this for you when you use `adk web`, but here we roll our own.
    """
    from dotenv import load_dotenv

    # Try project root .env as a fallback
    proj_root = os.path.dirname(os.path.dirname(__file__))
    env_path_local = os.path.join(proj_root, ".env")
    env_path_agent = os.path.join(os.path.dirname(__file__), ".env")

    # Agent-local .env wins if present
    if os.path.exists(env_path_agent):
        load_dotenv(env_path_agent, override=True)
    elif os.path.exists(env_path_local):
        load_dotenv(env_path_local, override=True)


# ---- Custom Invocation Metrics Plugin ----
class InvocationMetricsPlugin(BasePlugin):
    """Simple plugin to count agent + tool invocations."""

    def __init__(self) -> None:
        super().__init__(name="invocation_metrics")
        self.agent_count = 0
        self.tool_count = 0
        self.llm_request_count = 0

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        self.agent_count += 1
        logging.info(
            "[Metrics] Agent '%s' invoked. Total agent calls: %d",
            agent.name,
            self.agent_count,
        )

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        self.llm_request_count += 1
        logging.info(
            "[Metrics] LLM request #%d for model: %s",
            self.llm_request_count,
            llm_request.model,
        )

    async def before_tool_callback(
        self, *, tool_name: str, callback_context: CallbackContext
    ) -> None:
        self.tool_count += 1
        logging.info(
            "[Metrics] Tool '%s' invoked. Total tool calls: %d",
            tool_name,
            self.tool_count,
        )


async def main():
    load_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("🚀 Running ml_researcher with observability plugins...\n")

    session_service = InMemorySessionService()

    runner = Runner(
        agent=root_agent,
        app_name="ml_researcher",
        session_service=session_service,
        plugins=[
            LoggingPlugin(),
            InvocationMetricsPlugin(),
        ],
    )

    query = "Find recent SOTA OCR models and relevant Kaggle datasets for invoices"

    events = await runner.run_debug(
        query,
        user_id="debug_user_id",
        session_id="debug_session_id",
        verbose=True,
    )

    print("\n=== FINAL RESPONSE(S) ===")
    for e in events:
        if e.is_final_response():
            print(e.content)


if __name__ == "__main__":
    asyncio.run(main())