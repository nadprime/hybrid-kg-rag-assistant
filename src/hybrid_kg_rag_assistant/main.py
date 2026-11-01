#!/usr/bin/env python
import sys
import warnings
import os
import contextlib
from pathlib import Path

# Set environment variable to suppress all warnings at system level
os.environ["PYTHONWARNINGS"] = "ignore"

# Store CrewAI data in project directory
project_root = Path(__file__).parent
storage_dir = project_root / "crewai_storage"

os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)

# Context manager to suppress stderr warnings
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


# Suppress all warnings before any imports
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Import with stderr suppression for the persistent Pydantic warning
with suppress_stderr():
    from hybrid_kg_rag_assistant.crew import HybridKgRagAssistant

# Additional warning suppressions for runtime
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince.*")
warnings.filterwarnings("ignore", message=".*No path_separator found.*")
warnings.filterwarnings("ignore", message=".*model_fields.*")
warnings.filterwarnings("ignore", message=".*Using extra keyword arguments.*")
warnings.filterwarnings("ignore", message=".*Extra keys.*")
warnings.filterwarnings("ignore")  # Suppress all remaining warnings


def run():
    """
    Run the crew.
    """

    query = input("Enter your query: ")
    inputs = {
        "query": query,
    }

    try:
        result = HybridKgRagAssistant().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    print(f"Result: {result}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "query": "Tell me about mosdac mission and its objectives",
    }
    try:
        HybridKgRagAssistant().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        HybridKgRagAssistant().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "query": "Tell me about mosdac mission and its objectives",
    }

    try:
        HybridKgRagAssistant().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
