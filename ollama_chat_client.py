import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import aiohttp

from autogen_core import Image
from autogen_core._cancellation_token import CancellationToken
from autogen_core.components.models import LLMMessage
from autogen_core.models import SystemMessage, UserMessage, AssistantMessage
from autogen_core.models import (
    ChatCompletionClient,
    ModelCapabilities,
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
        model="llama3.2-vision:11b-instruct-q8_0",
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://127.0.0.1:11434",
        model_capabilities={
            "vision": True,  # Replace with True if the model has vision capabilities.
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
    num_ctx: int = 64000

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

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            vision=self.kwargs.get("model_capabilities", {}).get("vision", False),
            function_calling=self.kwargs.get("model_capabilities", {}).get("function_calling", False),
            json_output=self.kwargs.get("model_capabilities", {}).get("json_output", False),
        )

    def extract_role_and_content(self, msg) -> (str, Union[str, List[Union[str, Image]]]): # type: ignore
        """Helper function to extract role and content from various message types."""
        if isinstance(msg, SystemMessage):
            return 'system', msg.content
        elif isinstance(msg, UserMessage):
            return 'user', msg.content
        elif isinstance(msg, AssistantMessage):
            return 'assistant', msg.content
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
            # try:
            #     format = extract_inline_json_schema(content)
            # except LookupError:
            #     pass
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    # try:
                    #     format = extract_inline_json_schema(content)
                    # except LookupError:
                    #     pass
                elif isinstance(item, Image):
                    image_data = self.encode_image(item)
                    if image_data:
                        images.append(image_data)
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
            # Log or handle the error as needed
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
        outer_format = None
        for msg in messages:
            role, content = self.extract_role_and_content(msg)
            text, images = self.process_message_content(content)
            chat_message = {
                "role": role,
                "content": text
            }
            if images:
                chat_message["images"] = images
            # if format:
            #     outer_format = format
            chat_messages.append(chat_message)

        request_data = {
            "model": self.config.model,
            "messages": chat_messages,
            "stream": extra_create_args.get("stream", False),
            "options": {
                "temperature": extra_create_args.get(
                    "temperature", self.config.temperature
                ),
                "top_p": extra_create_args.get("top_p", self.config.top_p),
                "num_ctx": extra_create_args.get("num_ctx", self.config.num_ctx),
            },
        }
        # if outer_format and json_output:
        #     if speakers:=outer_format.get("properties", {}).get("next_speaker", {}).get("enum"):
        #         format = Ledger.model_json_schema()
        #         format["$defs"]["NextSpeaker"]["answer"]["enum"] = speakers
        #         request_data["format"] = format
        #     else:
        #         request_data["format"] = "json"
        if json_output:
            request_data["format"] = "json"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/chat",
                    json=request_data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return CreateResult(
                            content=f"Error: Ollama API error: {error_text}",
                            finish_reason="stop",
                            usage=self._actual_usage,
                            cached=False,
                        )

                    result = await response.json()
                    self.update_usage(result)
                    return CreateResult(
                        content=result.get("message", {}).get(
                            "content", "Error: No response content"
                        ),
                        finish_reason=result.get("done_reason", "stop"),
                        usage=self._actual_usage,
                        cached=False,
                    )
        except Exception as e:
            return CreateResult(
                content=f"Error: Failed to get response from Ollama server: {str(e)}",
                finish_reason="stop",
                usage=self._actual_usage,
                cached=False
            )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, Dict[str, Any]]] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        """Create a completion using the Ollama API."""

        extra_create_args["stream"] = True
        return self.create(messages, tools=tools, extra_create_args=extra_create_args, cancellation_token=cancellation_token)

    def update_usage(self, response_data: Dict[str, Any]) -> None:
        """Update the usage statistics based on the response data."""
        try:
            self._actual_usage.prompt_tokens += response_data.get("prompt_eval_count", 0)
            self._actual_usage.completion_tokens += response_data.get("eval_count", 0)
            self._actual_usage.total_tokens = (
                self._actual_usage.prompt_tokens + self._actual_usage.completion_tokens
            )

            self._total_usage.prompt_tokens += self._actual_usage.prompt_tokens
            self._total_usage.completion_tokens += self._actual_usage.completion_tokens
            self._total_usage.total_tokens += self._actual_usage.total_tokens
        except:
            pass

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Union[Tool, Dict[str, Any]]] = []) -> int:
        # Simple token counting based on message length
        return sum(len(msg.content.split()) for msg in messages) + len(tools) * 5  # Approximation

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
