import os
import logging
import feedparser
from typing import List, Optional, Tuple, Sequence
from datetime import datetime
from dateutil import parser as date_parser

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_core import CancellationToken


class RssFeedRetrieverAgent(BaseChatAgent):
    """An agent that retrieves and filters RSS feed entries based on publication dates and extracts vital information."""

    def __init__(
        self,
        name: str,
        *,
        description: str = "An agent that fetches RSS feeds, filters them by publication dates, and extracts vital information such as categories, summary, and tags.",
    ) -> None:
        super().__init__(name=name, description=description)
        self.logger = logging.getLogger(f"{self.name}.RssFeedRetrieverAgent")

    @property
    def produced_message_types(self) -> Tuple[type[ChatMessage], ...]:
        return (TextMessage,)

    async def on_messages(
        self,
        messages: Sequence[ChatMessage],
        cancellation_token: CancellationToken
    ) -> Response:
        """
        Processes incoming messages containing RSS feed URLs, fetches and filters the feeds.

        Args:
            messages (Sequence[ChatMessage]): The incoming chat messages.
            cancellation_token (CancellationToken): Token for handling cancellations.

        Returns:
            Response: The response containing filtered RSS feed entries.
        """
        feed_urls = []
        for msg in messages:
            if isinstance(msg, TextMessage):
                urls = [line.strip() for line in msg.content.splitlines() if line.strip()]
                feed_urls.extend(urls)

        if not feed_urls:
            return Response(
                chat_message=TextMessage(
                    content="No RSS feed URLs found in the messages.",
                    source=self.name,
                )
            )

        results = []
        for url in feed_urls:
            self.logger.debug(f"Fetching RSS feed from URL: {url}")
            entries = self.fetch_and_filter_rss(url)
            formatted_entries = self.format_entries(entries, url)
            results.append(formatted_entries)

        combined_results = "\n\n".join(results)
        return Response(
            chat_message=TextMessage(
                content=combined_results,
                source=self.name,
            )
        )

    def fetch_and_filter_rss(self, url: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[dict]:
        """
        Fetches and filters RSS feed entries based on publication dates.

        Args:
            url (str): The URL of the RSS feed.
            start_date (Optional[str]): The start date in 'YYYY-MM-DD' format.
            end_date (Optional[str]): The end date in 'YYYY-MM-DD' format.

        Returns:
            List[dict]: A list of filtered RSS feed entries with vital information.
        """
        try:
            feed = feedparser.parse(url)
            if feed.bozo:
                raise ValueError("Failed to parse RSS feed.")
        except Exception as e:
            self.logger.error(f"Error fetching RSS feed from {url}: {e}")
            return []

        filtered_entries = []
        for entry in feed.entries:
            pub_date_str = entry.get('published', '') or entry.get('pubDate', '')
            if not pub_date_str:
                continue

            try:
                pub_date = date_parser.parse(pub_date_str)
            except ValueError:
                continue

            if start_date and end_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                if not (start_dt <= pub_date <= end_dt):
                    continue

            entry_data = {
                'title': entry.get('title', 'No Title'),
                'link': entry.get('link', 'No Link'),
                'published': pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                'summary': entry.get('summary', 'No Summary'),
                'categories': [cat.term for cat in entry.get('tags', [])] if 'tags' in entry else [],
                'tags': [tag.term for tag in entry.get('tags', [])] if 'tags' in entry else [],
            }
            filtered_entries.append(entry_data)

        self.logger.info(f"Retrieved {len(filtered_entries)} entries from RSS feed: {url}")
        return filtered_entries

    def format_entries(self, entries: List[dict], url: str) -> str:
        """
        Formats the filtered RSS feed entries into a readable string.

        Args:
            entries (List[dict]): The filtered RSS feed entries.
            url (str): The RSS feed URL.

        Returns:
            str: A formatted string of RSS entries.
        """
        if not entries:
            return f"No RSS feed entries found for URL: {url} within the specified date range."

        formatted = f"RSS Feed Entries from {url}:\n"
        for idx, entry in enumerate(entries, start=1):
            categories = ", ".join(entry['categories']) if entry['categories'] else "N/A"
            tags = ", ".join(entry['tags']) if entry['tags'] else "N/A"
            formatted += (
                f"\nEntry {idx}:\n"
                f"**Title:** {entry['title']}\n"
                f"**Link:** {entry['link']}\n"
                f"**Published:** {entry['published']}\n"
                f"**Summary:** {entry['summary']}\n"
                f"**Categories:** {categories}\n"
                f"**Tags:** {tags}\n"
            )
        return formatted
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """It it's a no-op as the code executor agent has no mutable state."""
        pass