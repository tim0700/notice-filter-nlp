"""Base collector contract and robots.txt policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from src.schema import NoticeRecord


class BaseCollector(ABC):
    """Abstract base class for public board and API notice collectors."""

    source_name: str = ""
    base_url: str = ""
    user_agent: str = "notice-filter-nlp-bot/0.1"

    def check_robots(self, url: str) -> bool:
        """Return whether crawling the target URL is permitted by robots.txt."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            parser.read()
            return parser.can_fetch(self.user_agent, url)
        except OSError:
            return False

    @abstractmethod
    def fetch(self, limit: int = 100) -> list[NoticeRecord]:
        """Collect up to `limit` normalized NoticeRecord items."""


COLLECTOR_REGISTRY: dict[str, type[BaseCollector]] = {}
