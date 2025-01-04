import base64
from collections.abc import AsyncGenerator
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from ollama import AsyncClient, ChatResponse
from autogen_core import FunctionCall, Image
from autogen_core._cancellation_token import CancellationToken
from autogen_core.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage
)
from autogen_core.models import (
    ChatCompletionClient,
    ModelInfo,
    ModelFamily
)
from autogen_core.models._types import CreateResult, RequestUsage
from autogen_core.tools._base import Tool

from structured_output import Ledger
from utils import extract_inline_json_schema

def create_local_completion_client(
        **kwargs: Any
) -> ChatCompletionClient:
    # If model capabilities were provided, deserialize them as well
    "Mimic OpenAI API using Local LLM Server."
    return create_ollama_completion_client_from_env(
        # model="llama3.2-vision:11b-instruct-q8_0",
        model="qwen2.5:14b-instruct-q8_0",
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://127.0.0.1:11434",
        model_capabilities={
            "vision": False,  # Replace with True if the model has vision capabilities.
            "function_calling": True,  # Replace with True if the model has function calling capabilities.
            "json_output": True,  # Replace with True if the model has JSON output capabilities.
        },
        max_tokens=128000,
    )


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2-vision:11b-instruct-q8_0"
    temperature: float = 0.1
    top_p: float = 0.9
    num_ctx: int = 32000

class OllamaChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        config: OllamaConfig = OllamaConfig(),
        **kwargs: Any
    ):
        self.config = config
        self.kwargs = kwargs
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.client = AsyncClient()

    @property
    def capabilities(self) -> ModelInfo:
        return ModelInfo(
            vision=self.kwargs.get("model_capabilities", {}).get("vision", False),
            function_calling=self.kwargs.get("model_capabilities", {}).get(
                "function_calling", False
            ),
            json_output=self.kwargs.get("model_capabilities", {}).get(
                "json_output", False
            ),
            family=self.kwargs.get("model_capabilities", {}).get(
                "family", ModelFamily.UNKNOWN
            ),
        )

    @property
    def model_info(self) -> ModelInfo:
        return self.capabilities

    def extract_role_and_content(self, msg) -> (str, Union[str, List[Union[str, Image]]]): # type: ignore
        """Helper function to extract role and content from various message types."""
        if isinstance(msg, SystemMessage):
            return 'system', msg.content
        elif isinstance(msg, UserMessage):
            return 'user', msg.content
        elif isinstance(msg, AssistantMessage):
            return 'assistant', msg.content
        elif isinstance(msg, FunctionExecutionResultMessage):
            return "ipython", msg.content
        elif hasattr(msg, 'role') and hasattr(msg, 'content'):
            return msg.role, msg.content
        else:
            return 'user', str(msg)

    def process_message_content(self, content):
        text_parts = []
        images = []
        format = None
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, Image):
                    image_data = self.encode_image(item)
                    if image_data:
                        images.append(image_data)
                        text_parts.insert(0, "<|image|>")
                else:
                    text_parts.append(str(item))
        else:
            text_parts.append(str(content))

        return '\n'.join(text_parts), images

    def encode_image(self, image: Image) -> Optional[str]:
        """Encodes an Image object to a base64 string."""
        try:
            if hasattr(image, 'path'):
                with open(image.path, 'rb') as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
            elif hasattr(image, 'data'):
                return base64.b64encode(image.data).decode('utf-8')
            elif hasattr(image, 'data_uri'):
                return image.data_uri.split('base64,')[-1]
            else:
                return None
        except Exception as e:
            return None

    async def create(
            self,
            messages: Sequence[LLMMessage],
            *,
            tools: Sequence[Union[Tool, Dict[str, Any]]] = [],
            json_output: Optional[bool] = None,
            extra_create_args: Mapping[str, Any] = {},
            cancellation_token: Optional[CancellationToken] = None,
        ) -> CreateResult:
        """Create a completion using the Ollama API."""

        chat_messages = []
        for msg in messages:
            role, content = self.extract_role_and_content(msg)
            text, images = self.process_message_content(content)
            chat_message = {"role": role, "content": text}
            if images:
                chat_message["images"] = images
            chat_messages.append(chat_message)

        request_data = {
                "model": self.config.model,
                "messages": chat_messages,
                "tools": tools,
                "stream": False,
                "options": {
                    "temperature": extra_create_args.get(
                        "temperature", self.config.temperature
                    ),
                    "top_p": extra_create_args.get("top_p", self.config.top_p),
                    "num_ctx": extra_create_args.get("num_ctx", self.config.num_ctx),
                },
            }
        if json_output:
            request_data["format"] = "json"
        try:
            response = await self.client.chat(
                    model=self.config.model,
                    messages=chat_messages,
                    tools=tools,
                    stream=False,
                    options=request_data["options"],
                    format="json" if json_output else None,
                )
            if isinstance(response, ChatResponse):
                if response.message.content:
                    create_result = CreateResult(
                        finish_reason="stop",
                        content=response.message.content,
                        usage=self._actual_usage,
                        cached=False,
                    )
                elif response.message.tool_calls:
                    create_result = CreateResult(
                        finish_reason="function_calls",
                        content=[
                            FunctionCall(
                                tool_call.function.name,
                                json.dumps(tool_call.function.arguments),
                                tool_call.function.name)
                            for tool_call in response.message.tool_calls
                        ],
                        usage=self._actual_usage,
                        cached=False,
                    )
                self.update_usage(response)
                return create_result
        except Exception as e:
            return CreateResult(
                content=f"Error: Failed to get response from Ollama server: {str(e)}",
                finish_reason="stop",
                usage=self._actual_usage,
                cached=False,
            )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, Dict[str, Any]]] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create a streaming completion using the Ollama API."""

        chat_messages = []
        for msg in messages:
            role, content = self.extract_role_and_content(msg)
            text, images = self.process_message_content(content)
            chat_message = {"role": role, "content": text}
            if images:
                chat_message["images"] = images
            chat_messages.append(chat_message)

        request_data = {
            "model": self.config.model,
            "messages": chat_messages,
            "tools": tools,
            "stream": True,
            "options": {
                "temperature": extra_create_args.get(
                    "temperature", self.config.temperature
                ),
                "top_p": extra_create_args.get("top_p", self.config.top_p),
                "num_ctx": extra_create_args.get("num_ctx", self.config.num_ctx),
            },
        }
        if json_output:
            request_data["format"] = "json"

        try:
            async for response in self.client.chat(
                model=self.config.model,
                messages=chat_messages,
                tools=tools,
                stream=True,
                options=request_data["options"],
                format="json" if json_output else None,
            ):
                if isinstance(response, ChatResponse):
                    create_result = CreateResult(
                        finish_reason="stop",
                        content=response.message.content,
                        usage=self._actual_usage,
                        cached=False,
                    )
                    self.update_usage(response)
                    yield create_result
        except Exception as e:
            yield CreateResult(
                content=f"Error: Failed to stream response from Ollama server: {str(e)}",
                finish_reason="stop",
                usage=self._actual_usage,
                cached=False,
            )

    def update_usage(self, response_data: ChatResponse) -> None:
        """Update the usage statistics based on the response data."""
        try:
            self._actual_usage.prompt_tokens += response_data.prompt_eval_count
            self._actual_usage.completion_tokens += response_data.eval_count
            self._actual_usage.total_tokens = (
                self._actual_usage.prompt_tokens + self._actual_usage.completion_tokens
            )

            self._total_usage = self.actual_usage()
        except:
            pass

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Union[Tool, Dict[str, Any]]] = []) -> int:
        return sum(len(msg.content.split()) for msg in messages) + len(tools) * 5

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Union[Tool, Dict[str, Any]]] = []) -> int:
        max_tokens = self.kwargs.get("max_tokens", 128000)
        used_tokens = self.count_tokens(messages, tools=tools)
        return max_tokens - used_tokens

def create_ollama_completion_client_from_env(env: Optional[Dict[str, str]] = None, **kwargs: Any) -> ChatCompletionClient:
    """Create a model client based on environment variables."""
    if env is None:
        env = dict()
        env.update(os.environ)

    _kwargs = json.loads(env.get("MODEL_CONFIG", "{}"))
    _kwargs.update(kwargs)

    return OllamaChatCompletionClient(
        config=OllamaConfig(
            base_url=_kwargs.pop("base_url", OllamaConfig.base_url),
            model=_kwargs.pop("model", OllamaConfig.model),
            temperature=_kwargs.pop("temperature", OllamaConfig.temperature),
            top_p=_kwargs.pop("top_p", OllamaConfig.top_p),
        ),
        **_kwargs
    )
