import os
import logging

from dotenv import load_dotenv
import agentops

# Load .env from the *caller’s* directory when imported
load_dotenv(override=False)

_initialized = False

def init_agentops(trace_name: str | None = None):
    """
    Idempotent AgentOps initialization.

    Call this at import time from each agent module.
    Multiple calls are safe.
    """
    global _initialized
    if _initialized:
        return

    api_key = os.getenv("AGENTOPS_API_KEY")

    if not api_key:
        logging.warning(
            "[AgentOps] AGENTOPS_API_KEY not set. "
            "AgentOps telemetry will be disabled for this process."
        )
        return

    agentops.init(
        api_key=api_key,
        trace_name=trace_name or "ml-co-pilot",  # shows up in AgentOps UI
        # auto_start_session=True by default; we keep it that way
    )

    logging.info(f"[AgentOps] Initialized with trace_name={trace_name or 'ml-co-pilot'}")
    _initialized = True