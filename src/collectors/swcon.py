"""Collectors for Kyung Hee Univ. Department of Software Convergence boards."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

from src.collectors.base import (
    BaseCollector,
    build_notice_id,
    clean_whitespace,
    extract_deadline_from_text,
    extract_html_text,
    extract_leading_category,
    extract_query_param,
    join_meta_tags,
    normalize_date,
    register_collector,
)
from src.schema import NoticeRecord

logger = logging.getLogger(__name__)

SWCON_BOARDS: dict[str, dict[str, str]] = {
    "swcon_academic": {
        "menu_no": "21300017",
        "default_category": "학사/장학",
        "default_tags": "국제캠퍼스|소프트웨어융합학과",
    },
    "swcon_career": {
        "menu_no": "21300018",
        "default_category": "취업/대외활동",
        "default_tags": "국제캠퍼스|소프트웨어융합학과",
    },
}

_ONCLICK_ID_RE = re.compile(r"(?:fn_view|goView|view|detail)\s*\(\s*['\"]?(\d+)['\"]?")


class SwconCollector(BaseCollector):
    """Collector for swcon.khu.ac.kr BMSR00040 public notice boards."""

    base_url: str = "https://swcon.khu.ac.kr/swcon/user/bbs/BMSR00040/list.do"
    view_base_url: str = "https://swcon.khu.ac.kr/swcon/user/bbs/BMSR00040/view.do"

    def __init__(
        self,
        source_name: str = "swcon_academic",
        fetch_detail: bool = True,
        max_pages: int = 20,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        delay_seconds: float = 0.25,
    ) -> None:
        if source_name not in SWCON_BOARDS:
            raise ValueError(f"Unsupported swcon source_name: {source_name!r}")
        super().__init__(session=session, timeout=timeout, delay_seconds=delay_seconds)
        self.source_name = source_name
        self.board_config = SWCON_BOARDS[source_name]
        self.menu_no = self.board_config["menu_no"]
        self.fetch_detail = fetch_detail
        self.max_pages = max_pages

    def _build_list_url(self, page: int = 1) -> str:
        return f"{self.base_url}?menuNo={self.menu_no}&pageIndex={page}"

    def _extract_detail_url(self, anchor: Tag, list_url: str) -> tuple[str, str]:
        href = clean_whitespace(str(anchor.get("href", "")))
        onclick = clean_whitespace(str(anchor.get("onclick", "")))
        data_id = clean_whitespace(str(anchor.get("data-nttid") or anchor.get("data-id") or ""))

        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            raw_id = data_id
            if not raw_id and onclick:
                match = _ONCLICK_ID_RE.search(onclick)
                if match:
                    raw_id = match.group(1)
            if not raw_id:
                return "", ""
            full_url = f"{self.view_base_url}?menuNo={self.menu_no}&nttId={raw_id}"
            return full_url, raw_id

        full_url = urljoin(list_url, href)
        raw_id = extract_query_param(full_url, "nttId", "boardId", "seq", "idx") or data_id
        return full_url, raw_id

    def _find_title_anchor(self, row: Tag) -> Tag | None:
        for selector in (".subject a", ".title a", ".tit a", "td.tal a", "td.left a", "a"):
            anchor = row.select_one(selector)
            if anchor is not None and clean_whitespace(anchor.get_text(" ", strip=True)):
                return anchor
        return None

    def _parse_list_html(self, html: str, list_url: str) -> list[NoticeRecord]:
        soup = BeautifulSoup(html, "lxml")
        rows = soup.select("table tbody tr")
        if not rows:
            rows = [tr for tr in soup.select("table tr") if not tr.find("th")]
        if not rows:
            rows = soup.select("ul.board_list > li, ul.bbs_list > li, div.board_list li")

        records: list[NoticeRecord] = []
        for row in rows:
            row_text = clean_whitespace(row.get_text(" ", strip=True))
            if not row_text or "등록된 게시물이 없습니다" in row_text or "게시물이 없습니다" in row_text:
                continue

            anchor = self._find_title_anchor(row)
            if anchor is None:
                continue

            title = clean_whitespace(anchor.get_text(" ", strip=True))
            detail_url, raw_id = self._extract_detail_url(anchor, list_url)
            if not title or not detail_url:
                continue

            cate_el = row.select_one(".cate, .category, td.cate, span.cate")
            raw_category = clean_whitespace(cate_el.get_text(" ", strip=True)) if cate_el else ""
            category = (
                raw_category
                or extract_leading_category(title)
                or self.board_config["default_category"]
            )

            date_el = row.select_one(".date, td.date, span.date")
            posted_at = normalize_date(date_el.get_text(" ", strip=True) if date_el else "")
            if not posted_at:
                for cell in row.select("td, span"):
                    candidate = normalize_date(cell.get_text(" ", strip=True))
                    if candidate:
                        posted_at = candidate
                        break

            deadline_el = row.select_one(".deadline, td.deadline, span.deadline")
            deadline = extract_deadline_from_text(
                deadline_el.get_text(" ", strip=True) if deadline_el else title
            )

            meta_parts: list[str] = []
            row_classes = " ".join(row.get("class", []))
            if "notice" in row_classes.lower() or row.select_one(".notice, .badge_notice"):
                meta_parts.append("공지")
            writer_el = row.select_one(".writer, .name, .dept, td.writer")
            if writer_el:
                meta_parts.append(writer_el.get_text(" ", strip=True))
            for tag_el in row.select(".tag, .badge"):
                meta_parts.append(tag_el.get_text(" ", strip=True))
            meta_parts.append(self.board_config["default_tags"])

            notice_id = build_notice_id(self.source_name, raw_id, detail_url)
            records.append(
                NoticeRecord(
                    id=notice_id,
                    source=self.source_name,
                    category=category,
                    title=title,
                    meta_tags=join_meta_tags(*meta_parts),
                    posted_at=posted_at,
                    url=detail_url,
                    content="",
                    deadline=deadline,
                )
            )
        return records

    def _parse_detail_html(self, html: str) -> tuple[str, str, str]:
        soup = BeautifulSoup(html, "lxml")

        content_node: Tag | None = None
        for selector in (
            "div.board_view_con",
            "div.view_content",
            "div.bbs_view_cont",
            "div.board-view-content",
            "div.view_con",
            "div.board_view .content",
            "td.content",
            "article",
        ):
            content_node = soup.select_one(selector)
            if content_node is not None:
                break

        # Guard against unexpected login redirects when no public body exists.
        if content_node is None and soup.select_one("input[type='password']"):
            return "", "", ""

        content = extract_html_text(content_node)

        posted_at = ""
        deadline = ""
        for label_el, value_el in self._iter_detail_pairs(soup):
            label = clean_whitespace(label_el.get_text(" ", strip=True))
            value = clean_whitespace(value_el.get_text(" ", strip=True))
            if not posted_at and any(k in label for k in ("등록일", "작성일", "게시일")):
                posted_at = normalize_date(value)
            if not deadline and any(k in label for k in ("마감", "신청기간", "접수기간", "기한")):
                deadline = extract_deadline_from_text(value)

        if not deadline and content:
            deadline = extract_deadline_from_text(content)

        return content, posted_at, deadline

    @staticmethod
    def _iter_detail_pairs(soup: BeautifulSoup) -> list[tuple[Tag, Tag]]:
        pairs: list[tuple[Tag, Tag]] = []
        for th in soup.select("table th"):
            td = th.find_next_sibling("td")
            if td is not None:
                pairs.append((th, td))
        for dt in soup.select("dl dt"):
            dd = dt.find_next_sibling("dd")
            if dd is not None:
                pairs.append((dt, dd))
        return pairs

    def fetch(self, limit: int = 100) -> list[NoticeRecord]:
        """Collect paginated public notices and optional non-login detail bodies."""
        if limit <= 0:
            return []

        first_list_url = self._build_list_url(1)
        if not self._is_allowed(first_list_url):
            logger.warning("robots.txt disallowed crawling %s", first_list_url)
            return []

        collected: list[NoticeRecord] = []
        seen_ids: set[str] = set()

        for page in range(1, self.max_pages + 1):
            if len(collected) >= limit:
                break
            params: dict[str, Any] = {"menuNo": self.menu_no, "pageIndex": page}
            response = self._request(self.base_url, params=params)
            page_records = self._parse_list_html(response.text, self._build_list_url(page))
            if not page_records:
                break

            added_in_page = 0
            for record in page_records:
                if record.id in seen_ids:
                    continue
                seen_ids.add(record.id)
                collected.append(record)
                added_in_page += 1
                if len(collected) >= limit:
                    break

            if added_in_page == 0:
                break

        if self.fetch_detail:
            for record in collected:
                if not self._is_allowed(record.url):
                    continue
                try:
                    detail_resp = self._request(record.url)
                    if "login" in detail_resp.url.lower():
                        continue
                    content, detail_posted, detail_deadline = self._parse_detail_html(detail_resp.text)
                    record.content = content
                    if not record.posted_at and detail_posted:
                        record.posted_at = detail_posted
                    if not record.deadline and detail_deadline:
                        record.deadline = detail_deadline
                except requests.RequestException as exc:
                    logger.warning("Failed to fetch detail for %s: %s", record.id, exc)

        return collected[:limit]


@register_collector("swcon_academic")
class SwconAcademicCollector(SwconCollector):
    """Collector for swcon_academic (menuNo=21300017)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("source_name", None)
        super().__init__(source_name="swcon_academic", **kwargs)


@register_collector("swcon_career")
class SwconCareerCollector(SwconCollector):
    """Collector for swcon_career (menuNo=21300018)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("source_name", None)
        super().__init__(source_name="swcon_career", **kwargs)
