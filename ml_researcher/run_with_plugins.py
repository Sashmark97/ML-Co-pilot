import logging
from google.adk.runners import InMemoryRunner
from google.genai import types

from ml_researcher.agent import root_agent as research_root_agent
from ml_common.plugins import get_common_plugins


def build_runner():
    # Configure logging as you like (for production you’d often use INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    runner = InMemoryRunner(
        agent=research_root_agent,
        plugins=get_common_plugins(),
    )
    return runner


async def run_example():
    """
    Minimal example call – you don’t *have* to run this,
    but it shows how everything is wired.
    """
    runner = build_runner()
    content = types.Content(role="user", parts=[types.Part(text="""
Research current SOTA methods for OCR on document images.
Return:
- 2–3 key papers with arxiv IDs
- typical datasets & metrics
- mention any strong open-source repos we can start from
""")])

    events = runner.run_async(
        user_id="demo-user",
        session_id="research-demo-session",
        new_message=content,
    )

    async for event in events:
        if event.is_final_response():
            print("\n=== FINAL RESPONSE ===")
            for p in event.content.parts:
                if p.text:
                    print(p.text)