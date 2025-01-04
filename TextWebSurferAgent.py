from http.cookiejar import DefaultCookiePolicy
import os
import logging
import requests
from typing import List, Sequence, Tuple, Optional, Dict, Any, Union
import json
from bs4 import BeautifulSoup
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models._types import LLMMessage, SystemMessage, UserMessage, AssistantMessage
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, ChatMessage

from autogen_core.models import ChatCompletionClient, CreateResult
from autogen_core.tools._base import Tool, ToolSchema

REASONING_TOOL_PROMPT = "A short description of the action to be performed and reason for doing so, do not mention the user."

# TOOLS
TOOL_VISIT_URL = {
        "type": "function",
        "function": {
            "name": "visit_url",
            "description": "Navigate directly to a provided URL using the browser's address bar. Prefer this tool over other navigation techniques in cases where the user provides a fully-qualified URL (e.g., choose it over clicking links, or inputing queries into search boxes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": REASONING_TOOL_PROMPT,
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL to visit in the browser.",
                    },
                },
                "required": ["reasoning", "url"],
            },
        },
    }

TOOL_WEB_SEARCH = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a web search on Bing.com with the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": REASONING_TOOL_PROMPT,
                    },
                    "query": {
                        "type": "string",
                        "description": "The web search query to use.",
                    },
                },
                "required": ["reasoning", "query"],
            },
        },
    }

TOOL_READ_PAGE_AND_ANSWER = {
        "type": "function",
        "function": {
            "name": "answer_question",
            "description": "Uses AI to answer a question about the current webpage's content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": REASONING_TOOL_PROMPT,
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to answer.",
                    },
                },
                "required": ["reasoning", "question"],
            },
        },
    }

TOOL_SUMMARIZE_PAGE = {
        "type": "function",
        "function": {
            "name": "summarize_page",
            "description": "Uses AI to summarize the entire page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": REASONING_TOOL_PROMPT,
                    },
                },
                "required": ["reasoning"],
            },
        },
    }

class AllowAllCookiesPolicy(DefaultCookiePolicy):
    """Cookie policy that allows all cookies."""

    def return_ok(self, *args, **kwargs):
        return True


class TextWebSurferAgent(BaseChatAgent):
    """
    TextWebSurferAgent is a sophisticated text-based web surfer that leverages a language model to intelligently navigate and process web content.
    
    It mimics the functionality of the text browser by retrieving web pages as text, accepting all cookies, and utilizing advanced summarization techniques. The agent intelligently decides which tools to use based on user input and context, ensuring efficient and relevant interactions.
    
    Features:
        - **Web Search:** Perform web searches using Bing API to fetch relevant information.
        - **Visit URL:** Retrieve and process content from specified URLs.
        - **Summarize Page:** Generate concise summaries of fetched web pages using the language model.
        - **Tool Decision Making:** Utilize the language model to determine the most appropriate tools to execute based on user queries and context.
    
    Args:
        name (str): The name of the agent.
        model_client (ChatCompletionClient): The language model client used for decision making and summarization.
        description (str, optional): Detailed description of the agent's capabilities.
    """

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        *,
        description: str = (
            "TextWebSurferAgent is a text-based web surfer that retrieves web pages, accepts all cookies, "
            "performs web searches, visits URLs, summarizes pages, and extracts vital information such as categories, summary, and tags. "
            "It intelligently decides which tools to use based on user input and context, leveraging a language model to enhance interactions."
        ),
    ) -> None:
        super().__init__(name=name, description=description)
        self.logger = logging.getLogger(f"{self.name}.TextWebSurferAgent")
        self.session = requests.Session()
        self.model_client = model_client
        # Accept all cookies
        self.session.cookies.set_policy(AllowAllCookiesPolicy())
        self.default_tools: List[ToolSchema] = [
            TOOL_VISIT_URL,
            TOOL_WEB_SEARCH,
            TOOL_READ_PAGE_AND_ANSWER,
            TOOL_SUMMARIZE_PAGE,
        ]

    @property
    def produced_message_types(self) -> Tuple[type[ChatMessage], ...]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        user_content = "\n".join([msg.content for msg in messages if isinstance(msg, TextMessage)])
        self.logger.debug(f"Received user content: {user_content}")

        # Use model_client to decide which tools to use
        tool_commands = await self.decide_tools(user_content, cancellation_token)

        results = []

        for function_call in tool_commands:
            self.logger.debug(
                f"Executing tool: {function_call.name} with params: {function_call.arguments}"
            )
            tool_result = await self.execute_tool(
                function_call.name, json.loads(function_call.arguments)
            )
            results.append(tool_result)

        if not results:
            return Response(
                chat_message=TextMessage(
                    content="No actionable commands found in the messages.",
                    source=self.name,
                )
            )

        combined_results = "\n\n".join(results)
        return Response(
            chat_message=TextMessage(
                content=combined_results,
                source=self.name,
            )
        )

    async def decide_tools(self, user_content: str, cancellation_token: CancellationToken) -> List[FunctionCall]:
        """
        Use the language model to decide which tools to invoke based on the user content.

        Args:
            user_content (str): The aggregated user messages.
            cancellation_token (CancellationToken): Token to handle cancellation requests.

        Returns:
            List[FunctionCall]: A list of FunctionCall containing tool names and their parameters.
        """
        system_prompt = (
            "You are an intelligent assistant that uses the following tools to assist the user:\n\n"
            + "\n".join(
                [
                    f"- **{tool['function']['name']}**: {tool['function']['description']}"
                    for tool in self.default_tools
                ]
            )
            + "\n\n"
            "Based on the user input, decide which tools to use and provide the necessary parameters in JSON format."
        )

        messages: List[LLMMessage] = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_content, source=self.name),
        ]

        try:
            create_result: CreateResult = await self.model_client.create(
                messages=messages,
                tools=self.default_tools,
                cancellation_token=cancellation_token,
                extra_create_args={},
                json_output=True,
            )

            if not isinstance(create_result, CreateResult) or not isinstance(create_result.content, list):
                self.logger.error("Model response is not an CreateResult or List[FunctionCall].")
                return []

            return create_result.content

        except Exception as e:
            self.logger.error(f"Error during tool decision making: {e}")
            return []

    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        if tool_name == "visit_url":
            return self.visit_url(params.get("url", ""))
        elif tool_name == "web_search":
            return self.perform_web_search(params.get("query", ""))
        elif tool_name == "summarize_page":
            return await self.summarize_page()
        elif tool_name == "read_page_and_answer":
            return self.read_page_and_answer(params.get("question", ""))
        elif tool_name == "input_text":
            return self.type_text(params.get("input_field_id"), params.get("text_value", ""))
        else:
            return f"Unknown tool: {tool_name}"

    async def summarize_page(self) -> str:
        self.logger.debug("Executing summarize_page tool.")
        if not hasattr(self, "last_fetched_content") or not self.last_fetched_content:
            return "No page content available to summarize."

        # Prepare messages for the model_client
        messages: List[LLMMessage] = [
            SystemMessage(
                content="You are an assistant that provides concise summaries of web page content."
            ),
            UserMessage(
                content=f"Please summarize the following web page content:\n\n{self.last_fetched_content}"
            ),
        ]

        try:
            # Call the model client with the prepared messages and default tools
            create_result: CreateResult = await self.model_client.create(
                messages=messages,
                tools=self.default_tools,
                cancellation_token=None,  # Replace with an actual CancellationToken if available
                extra_create_args={},      # Add any extra arguments if needed
                json_output=False,        # Set to True if expecting structured output
            )

            # Ensure the response is from the assistant
            if not isinstance(create_result.message, AssistantMessage):
                self.logger.error("Model response is not an AssistantMessage.")
                return "Error summarizing the page."

            # Extract the summary from the model's response
            summary = create_result.message.content.strip()
            return f"Page Summary:\n{summary}"

        except Exception as e:
            self.logger.error(f"Error during summarization: {e}")
            return "Error summarizing the page."

    def perform_web_search(self, query: str) -> str:
        headers = {"Ocp-Apim-Subscription-Key": os.environ.get("BING_API_KEY", "")}
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        params = {
            "q": query,
            "textDecorations": True,
            "textFormat": "HTML",
            "setLang": "en-us",
            "safeSearch": "Off",
            "responseFilter": "Webpages",
            "count": 5,
            "mkt": "en-US",
            "freshness": "Week",
        }

        try:
            response = self.session.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()

            web_pages = search_results.get("webPages", {}).get("value", [])
            if web_pages:
                result_text = f"Search results for '{query}':"
                for idx, result in enumerate(web_pages, start=1):
                    title = result.get("name")
                    url = result.get("url")
                    snippet = result.get("snippet")
                    date_published = result.get("datePublished")
                    date_crawled = result.get("dateLastCrawled")
                    result_text += (
                        f"\n\nResult {idx}:\n"
                        f"Title: {title}\n"
                        f"URL: {url}\n"
                        f"Date Published: {date_published}\n"
                        f"Date Crawled: {date_crawled}\n"
                        f"Snippet: {snippet}"
                    )
                return result_text
            else:
                return f"No results found for query: '{query}'."

        except requests.exceptions.HTTPError as http_err:
            self.logger.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
            return f"Error performing search for query: '{query}'."
        except requests.exceptions.RequestException as req_err:
            self.logger.error(f"Request exception: {req_err}")
            return f"Error performing search for query: '{query}'."
        except ValueError as json_err:
            self.logger.error(f"JSON decode error: {json_err}")
            return f"Error parsing search results for query: '{query}'."

    def fetch_webpage(self, url: str) -> Optional[str]:
        headers = {
            "User-Agent": "Lynx/2.8.8rel.2 libwww-FM/2.14"
        }
        try:
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            self.last_fetched_content = response.text  # Store for summarization
            self.logger.info(f"Successfully fetched URL: {url}")
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching URL {url}: {e}")
            return None

    def visit_url(self, url: str) -> str:
        self.logger.debug(f"Executing visit_url with URL: {url}")
        content = self.fetch_webpage(url)
        if content:
            extracted_info = self.extract_vital_info(content)
            formatted_content = self.format_extracted_info(extracted_info, url)
            return formatted_content
        else:
            return f"Failed to retrieve content from URL: {url}"

    def extract_vital_info(self, html_content: str) -> Dict[str, Any]:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract title
        title = soup.title.string.strip() if soup.title else "No Title"

        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'].strip() if meta_desc and 'content' in meta_desc.attrs else "No Description"

        # Extract categories or tags
        categories = []
        for tag in soup.find_all(['meta', 'a'], attrs={'rel': 'tag'}):
            if tag.name == 'meta' and 'content' in tag.attrs:
                categories.append(tag['content'].strip())
            elif tag.name == 'a':
                categories.append(tag.get_text(strip=True))

        # Extract summary (first 200 words)
        paragraphs = soup.find_all('p')
        summary = ' '.join([p.get_text(strip=True) for p in paragraphs[:5]])
        summary_words = summary.split()
        summary = ' '.join(summary_words[:200]) + ('...' if len(summary_words) > 200 else '')

        return {
            'title': title,
            'description': description,
            'categories': categories,
            'summary': summary,
        }

    def format_extracted_info(self, info: Dict[str, Any], url: str) -> str:
        categories = ', '.join(info['categories']) if info['categories'] else "N/A"
        formatted = (
            f"Web Page: {info['title']}\n"
            f"URL: {url}\n"
            f"Description: {info['description']}\n"
            f"Categories/Tags: {categories}\n"
            f"Summary: {info['summary']}"
        )
        return formatted

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """
        Reset the agent's state by clearing session cookies and any stored content.

        This method ensures that the agent starts fresh by:
            - Clearing all session cookies.
            - Resetting the last fetched content.
            - Clearing chat history if maintained.
        
        Args:
            cancellation_token (CancellationToken): Token to handle cancellation requests.
        """
        self.logger.info("Resetting TextWebSurferAgent state.")

        # Clear session cookies
        self.session.cookies.clear()
        self.logger.debug("Session cookies cleared.")

        # Reset last fetched content
        self.last_fetched_content = None
        self.logger.debug("Last fetched content reset.")

        # Clear chat history if applicable
        if hasattr(self, 'chat_history'):
            self.chat_history.clear()
            self.logger.debug("Chat history cleared.")

        self.logger.info("TextWebSurferAgent has been successfully reset.")

    def read_page_and_answer(self, question: str) -> str:
        """
        Placeholder for the read_page_and_answer tool.
        Implement the logic to read the current page and answer the given question.
        """
        # Implement the logic based on the fetched content
        if not hasattr(self, "last_fetched_content") or not self.last_fetched_content:
            return "No page content available to answer the question."

        # For simplicity, return a mock answer
        return f"Answering the question: '{question}' based on the current page content."

    def type_text(self, input_field_id: int, text_value: str) -> str:
        """
        Placeholder for the input_text tool.
        Implement the logic to type text into a specified input field.
        """
        # Implement the logic to interact with the browser to type text
        return f"Typed '{text_value}' into input field with ID {input_field_id}."
