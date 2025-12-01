import logging

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.plugins.logging_plugin import LoggingPlugin


class InvocationMetricsPlugin(BasePlugin):
    """
    Simple observability plugin:
    - counts agent calls
    - counts tool calls
    - counts LLM requests

    All metrics are emitted via the standard logging system.
    """

    def __init__(self) -> None:
        super().__init__(name="invocation_metrics")
        self.agent_count: int = 0
        self.tool_count: int = 0
        self.llm_request_count: int = 0

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> None:
        self.agent_count += 1
        logging.info(
            "[Metrics] Agent '%s' invoked. Total agent calls: %d",
            agent.name,
            self.agent_count,
        )

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        callback_context: CallbackContext,
        tool_input,
    ) -> None:
        self.tool_count += 1
        logging.info(
            "[Metrics] Tool '%s' called. Total tool calls: %d",
            tool.name,
            self.tool_count,
        )

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> None:
        self.llm_request_count += 1
        logging.info(
            "[Metrics] LLM request #%d for model '%s'",
            self.llm_request_count,
            llm_request.model,
        )


def get_common_plugins():
    """
    Return the standard plugin stack you can attach to any runner.
    - LoggingPlugin: structured traces/logs for all agents & tools
    - InvocationMetricsPlugin: simple counters on top
    """
    return [
        LoggingPlugin(),
        InvocationMetricsPlugin(),
    ]