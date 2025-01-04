from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import (
    ChatCompletionClient,
)

MAGENTIC_ONE_CODER_DESCRIPTION = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills."
# using coding in llama: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#-code-interpreter-
MAGENTIC_ONE_CODER_SYSTEM_MESSAGE = """Environment: ipython"""

class LLamaCoderAgent(AssistantAgent):
    """An agent, used by MagenticOne, ollama and llama3 that provides coding assistance using an LLM model client.

    The prompts and description are sealed, to replicate the original MagenticOne configuration. See AssistantAgent if you wish to modify these values.
    """

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
    ):
        super().__init__(
            name,
            model_client,
            description=MAGENTIC_ONE_CODER_DESCRIPTION,
            system_message=MAGENTIC_ONE_CODER_SYSTEM_MESSAGE,
        )
