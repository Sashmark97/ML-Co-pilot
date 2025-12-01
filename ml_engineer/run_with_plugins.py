import logging
from google.adk.runners import InMemoryRunner
from google.genai import types

from ml_engineer.agent import root_agent as engineer_root_agent
from ml_common.plugins import get_common_plugins


def build_runner():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    runner = InMemoryRunner(
        agent=engineer_root_agent,
        plugins=get_common_plugins(),
    )
    return runner


async def run_example():
    runner = build_runner()
    content = types.Content(
        role="user",
        parts=[types.Part(text="""
For MNIST classification, train Logistic Regression (sklearn),
SVM (sklearn), and an MLP (PyTorch). Compare F1 on test set and
save the best model to best_MNIST_model.pth.
""")],
    )

    events = runner.run_async(
        user_id="demo-user",
        session_id="engineer-demo-session",
        new_message=content,
    )

    async for event in events:
        if event.is_final_response():
            print("\n=== FINAL RESPONSE ===")
            for p in event.content.parts:
                if p.text:
                    print(p.text)