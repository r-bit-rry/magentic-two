import os
from typing import List

from autogen_agentchat.agents import CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.base import ChatAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core.models import ChatCompletionClient

from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
# from LlamaCoderAgent import LLamaCoderAgent
from RssFeedRetrieverAgent import RssFeedRetrieverAgent
from TextWebSurferAgent import TextWebSurferAgent


class MagenticOne(MagenticOneGroupChat):
    def __init__(self, client: ChatCompletionClient, hil_mode: bool = False):
        # fs = FileSurfer("FileSurfer", model_client=client)
        # ws = MultimodalWebSurfer("WebSurfer", model_client=client)
        ws = TextWebSurferAgent("WebSurfer", model_client=client)
        coder = MagenticOneCoderAgent("Coder", model_client=client)
        executor = CodeExecutorAgent(
            "Executor", code_executor=LocalCommandLineCodeExecutor()
            # "Executor", code_executor=DockerCommandLineCodeExecutor(),
        )
        agents: List[ChatAgent] = [ws, coder, executor]
        if hil_mode:
            user_proxy = UserProxyAgent("User")
            agents.append(user_proxy)
        super().__init__(agents, model_client=client, max_turns=20)
