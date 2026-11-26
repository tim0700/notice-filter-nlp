"""Base collector contract, registry, robots.txt policy, and shared parsing helpers."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

from src.schema import NoticeRecord

logger = logging.getLogger(__name__)

COLLECTOR_REGISTRY: dict[str, type[BaseCollector]] = {}

_DATE_FULL_RE = re.compile(
    r"(?P<year>(?:19|20)\d{2})\s*[.\-/년]\s*(?P<month>\d{1,2})\s*[.\-/월]\s*(?P<day>\d{1,2})\s*일?"
)
_DATE_COMPACT_RE = re.compile(r"\b(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12]\d|3[01])\b")
_DATE_SHORT_YEAR_RE = re.compile(r"\b(?P<year>\d{2})\s*[.\-/]\s*(?P<month>\d{1,2})\s*[.\-/]\s*(?P<day>\d{1,2})\b")
_LEADING_BRACKET_RE = re.compile(r"^\s*\[([^\[\]]{1,25})\]\s*")


def register_collector(source_name: str) -> Callable[[type[BaseCollector]], type[BaseCollector]]:
    """Register a BaseCollector subclass under `source_name` in COLLECTOR_REGISTRY."""

    def _decorator(cls: type[BaseCollector]) -> type[BaseCollector]:
        cls.source_name = source_name
        COLLECTOR_REGISTRY[source_name] = cls
        return cls

    return _decorator


def clean_whitespace(text: str | None) -> str:
    """Normalize arbitrary whitespace into single spaces and strip edges."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _format_ymd(year: int, month: int, day: int) -> str:
    """Validate calendar components and format as YYYY-MM-DD."""
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def normalize_date(raw: str | None) -> str:
    """Normalize common Korean date strings into YYYY-MM-DD, or return '' if invalid."""
    text = clean_whitespace(raw)
    if not text:
        return ""

    match = _DATE_FULL_RE.search(text)
    if match:
        return _format_ymd(int(match.group("year")), int(match.group("month")), int(match.group("day")))

    compact = _DATE_COMPACT_RE.search(text)
    if compact:
        return _format_ymd(
            int(compact.group("year")),
            int(compact.group("month")),
            int(compact.group("day")),
        )

    short = _DATE_SHORT_YEAR_RE.search(text)
    if short:
        return _format_ymd(
            2000 + int(short.group("year")),
            int(short.group("month")),
            int(short.group("day")),
        )

    return ""


def extract_deadline_from_text(text: str | None) -> str:
    """Extract a likely deadline date (preferring range end dates) as YYYY-MM-DD."""
    cleaned = clean_whitespace(text)
    if not cleaned:
        return ""

    # Prefer the end of a date range (e.g. "2026-11-01 ~ 2026-12-15").
    if "~" in cleaned:
        after_tilde = cleaned.split("~", 1)[1]
        end_date = normalize_date(after_tilde)
        if end_date:
            return end_date

    # Check explicit deadline markers next.
    marker_match = re.search(
        r"(?:마감|종료|접수기간|신청기간|제출기한|까지)\s*[:：]?\s*([^\n]{0,40})",
        cleaned,
    )
    if marker_match:
        marker_date = normalize_date(marker_match.group(1))
        if marker_date:
            return marker_date

    # Fall back to the last full date present in a short deadline field.
    matches = list(_DATE_FULL_RE.finditer(cleaned))
    if matches:
        last = matches[-1]
        return _format_ymd(int(last.group("year")), int(last.group("month")), int(last.group("day")))

    return normalize_date(cleaned)


def extract_leading_category(title: str) -> str:
    """Extract a leading [tag] from a notice title when present."""
    match = _LEADING_BRACKET_RE.match(title or "")
    return clean_whitespace(match.group(1)) if match else ""


def join_meta_tags(*parts: str | Iterable[str]) -> str:
    """Deduplicate and join non-empty metadata tokens with '|'."""
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(token: str) -> None:
        cleaned = clean_whitespace(token).replace("|", "/")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)

    for part in parts:
        if isinstance(part, str):
            for chunk in part.split("|"):
                _add(chunk)
        elif part is not None:
            for item in part:
                for chunk in str(item).split("|"):
                    _add(chunk)
    return "|".join(ordered)


def extract_query_param(url: str, *param_names: str) -> str:
    """Return the first non-empty query parameter value matching `param_names`."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    for name in param_names:
        values = params.get(name)
        if values and values[0].strip():
            return values[0].strip()
    return ""


def build_notice_id(source: str, raw_id: str | None, fallback_url: str) -> str:
    """Build a canonical unique ID formatted as '<source>_<raw_id>'."""
    cleaned_id = re.sub(r"[^A-Za-z0-9_-]", "", (raw_id or "").strip())
    if cleaned_id:
        return f"{source}_{cleaned_id}"
    digest = hashlib.sha1(fallback_url.encode("utf-8")).hexdigest()[:10]
    return f"{source}_{digest}"


def extract_html_text(element: Tag | BeautifulSoup | None) -> str:
    """Extract clean readable text from an HTML fragment, dropping scripts/styles."""
    if element is None:
        return ""
    clone = BeautifulSoup(str(element), "lxml")
    for tag in clone(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    lines = [clean_whitespace(line) for line in clone.get_text("\n").splitlines()]
    return clean_whitespace(" ".join(line for line in lines if line))


class BaseCollector(ABC):
    """Abstract base class for public board and API notice collectors."""

    source_name: str = ""
    base_url: str = ""
    user_agent: str = "notice-filter-nlp-bot/0.1"

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        delay_seconds: float = 0.25,
    ) -> None:
        """Initialize HTTP session settings and polite crawling delay."""
        self._session = session
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self._robots_cache: dict[str, bool] = {}
        self._has_requested = False

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

    def _is_allowed(self, url: str) -> bool:
        """Check and cache robots.txt permission per target URL."""
        if url not in self._robots_cache:
            self._robots_cache[url] = self.check_robots(url)
        return self._robots_cache[url]

    def _get_session(self) -> requests.Session:
        """Lazily create a configured requests.Session."""
        if self._session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": self.user_agent})
            self._session = session
        return self._session

    def _request(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """Issue a polite HTTP GET request with timeout and error checking."""
        if self._has_requested and self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        self._has_requested = True

        session = self._get_session()
        response = session.get(url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"
        return response

    @abstractmethod
    def fetch(self, limit: int = 100) -> list[NoticeRecord]:
        """Collect up to `limit` normalized NoticeRecord items."""
