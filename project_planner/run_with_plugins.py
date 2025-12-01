import logging
from google.adk.runners import InMemoryRunner
from google.genai import types

from project_planner.agent import root_agent as planner_root_agent
from ml_common.plugins import get_common_plugins


def build_runner():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return InMemoryRunner(
        agent=planner_root_agent,
        plugins=get_common_plugins(),
    )