"""Collectors for Kyung Hee Univ. Future Talent Center (goodjob.khu.ac.kr) public lists."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup, Tag

from src.collectors.base import (
    BaseCollector,
    build_notice_id,
    clean_whitespace,
    extract_deadline_from_text,
    extract_leading_category,
    extract_query_param,
    join_meta_tags,
    normalize_date,
    register_collector,
)
from src.schema import NoticeRecord

logger = logging.getLogger(__name__)

GOODJOB_BOARDS: dict[str, dict[str, str]] = {
    "goodjob_extracurricular": {
        "bo_table": "s4_1",
        "sca": "국제",
        "default_category": "비교과",
        "default_tags": "국제캠퍼스",
    },
    "goodjob_general": {
        "bo_table": "s3_3",
        "sca": "",
        "default_category": "일반채용",
        "default_tags": "",
    },
    "goodjob_recommend": {
        "bo_table": "s3_2",
        "sca": "",
        "default_category": "추천채용",
        "default_tags": "",
    },
}


class GoodjobCollector(BaseCollector):
    """Public list-only collector for goodjob.khu.ac.kr (ADR-02: never logs in, content='')."""

    base_url: str = "https://goodjob.khu.ac.kr/bbs/board.php"

    def __init__(
        self,
        source_name: str = "goodjob_general",
        max_pages: int = 20,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        delay_seconds: float = 0.25,
    ) -> None:
        if source_name not in GOODJOB_BOARDS:
            raise ValueError(f"Unsupported goodjob source_name: {source_name!r}")
        super().__init__(session=session, timeout=timeout, delay_seconds=delay_seconds)
        self.source_name = source_name
        self.board_config = GOODJOB_BOARDS[source_name]
        self.bo_table = self.board_config["bo_table"]
        self.sca = self.board_config["sca"]
        self.max_pages = max_pages

    def _build_params(self, page: int = 1) -> dict[str, Any]:
        params: dict[str, Any] = {"bo_table": self.bo_table}
        if self.sca:
            params["sca"] = self.sca
        if page > 1:
            params["page"] = page
        return params

    def _build_list_url(self, page: int = 1) -> str:
        return f"{self.base_url}?{urlencode(self._build_params(page))}"

    def _find_title_anchor(self, row: Tag) -> Tag | None:
        for selector in (
            ".bo_tit a",
            "td.td_subject a[href*='wr_id']",
            ".subject a[href*='wr_id']",
            ".title a",
            ".subject a",
            "a[href*='wr_id']",
        ):
            anchor = row.select_one(selector)
            if anchor is not None and clean_whitespace(anchor.get_text(" ", strip=True)):
                return anchor
        for anchor in row.select("a"):
            classes = " ".join(anchor.get("class", []))
            if "bo_cate_link" in classes:
                continue
            if clean_whitespace(anchor.get_text(" ", strip=True)):
                return anchor
        return None

    def _extract_deadline(self, row: Tag, title: str) -> str:
        deadline_el = row.select_one("td.td_deadline, .td_deadline, .deadline, span.deadline")
        if deadline_el is not None:
            parsed = extract_deadline_from_text(deadline_el.get_text(" ", strip=True))
            if parsed:
                return parsed

        for el in row.select("td, span"):
            classes = " ".join(el.get("class", []))
            if any(skip in classes for skip in ("td_datetime", "date", "td_num", "td_hit")):
                continue
            text = clean_whitespace(el.get_text(" ", strip=True))
            if "~" in text or any(k in text for k in ("마감", "기한", "접수")):
                parsed = extract_deadline_from_text(text)
                if parsed:
                    return parsed

        if "~" in title or any(k in title for k in ("마감", "까지")):
            return extract_deadline_from_text(title)
        return ""

    def _extract_meta_tags(self, row: Tag, title_anchor: Tag) -> str:
        meta_parts: list[str] = []
        for selector in (
            ".meta",
            ".tags",
            ".tag",
            ".badge",
            ".campus",
            ".job_type",
            "td.td_company",
            "td.td_campus",
            "td.td_job",
            "td.td_name",
            ".sv_member",
        ):
            for el in row.select(selector):
                text = clean_whitespace(el.get_text(" ", strip=True))
                if text:
                    meta_parts.append(text)

        # Inspect generic table cells that are neither number, subject, date, deadline, nor hit count.
        ignored_tokens = {"td_num", "num", "td_subject", "subject", "td_cate", "cate", "category", "td_datetime", "date", "td_deadline", "deadline", "td_hit", "hit"}
        for cell in row.select("td"):
            cell_classes = set(cell.get("class", []))
            if cell_classes & ignored_tokens:
                continue
            if title_anchor in cell.descendants:
                continue
            text = clean_whitespace(cell.get_text(" ", strip=True))
            if not text or text.isdigit() or normalize_date(text):
                continue
            meta_parts.append(text)

        if self.board_config["default_tags"]:
            meta_parts.append(self.board_config["default_tags"])
        return join_meta_tags(*meta_parts)

    def _parse_list_html(self, html: str, list_url: str) -> list[NoticeRecord]:
        soup = BeautifulSoup(html, "lxml")
        rows = soup.select("div.tbl_head01 table tbody tr, table tbody tr")
        if not rows:
            rows = [tr for tr in soup.select("table tr") if not tr.find("th")]
        if not rows:
            rows = soup.select("ul.board_list > li, div.bo_list li, div.list_row")

        records: list[NoticeRecord] = []
        for row in rows:
            row_text = clean_whitespace(row.get_text(" ", strip=True))
            if not row_text or "게시물이 없습니다" in row_text or "등록된 글이 없습니다" in row_text:
                continue

            anchor = self._find_title_anchor(row)
            if anchor is None:
                continue

            title = clean_whitespace(anchor.get_text(" ", strip=True))
            href = clean_whitespace(str(anchor.get("href", "")))
            if not title or not href or href.startswith("#") or href.lower().startswith("javascript:"):
                continue

            detail_url = urljoin(list_url, href)
            wr_id = extract_query_param(detail_url, "wr_id", "id", "seq")
            notice_id = build_notice_id(self.source_name, wr_id, detail_url)

            cate_el = row.select_one(".bo_cate_link, td.td_cate, .cate, .category")
            raw_cate = clean_whitespace(cate_el.get_text(" ", strip=True)) if cate_el else ""
            category = (
                raw_cate
                or extract_leading_category(title)
                or self.board_config["default_category"]
            )

            date_el = row.select_one("td.td_datetime, .td_datetime, .date, span.date")
            posted_at = normalize_date(date_el.get_text(" ", strip=True) if date_el else "")
            if not posted_at:
                for cell in row.select("td, span"):
                    candidate = normalize_date(cell.get_text(" ", strip=True))
                    if candidate:
                        posted_at = candidate
                        break

            deadline = self._extract_deadline(row, title)
            meta_tags = self._extract_meta_tags(row, anchor)

            # Strictly public list only: content must remain "" per ADR-02.
            records.append(
                NoticeRecord(
                    id=notice_id,
                    source=self.source_name,
                    category=category,
                    title=title,
                    meta_tags=meta_tags,
                    posted_at=posted_at,
                    url=detail_url,
                    content="",
                    deadline=deadline,
                )
            )
        return records

    def fetch(self, limit: int = 100) -> list[NoticeRecord]:
        """Collect public list metadata only (never logs in, never fetches detail pages)."""
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
            params = self._build_params(page)
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

        return collected[:limit]


@register_collector("goodjob_extracurricular")
class GoodjobExtracurricularCollector(GoodjobCollector):
    """Collector for goodjob_extracurricular (s4_1, international campus filter)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("source_name", None)
        super().__init__(source_name="goodjob_extracurricular", **kwargs)


@register_collector("goodjob_general")
class GoodjobGeneralCollector(GoodjobCollector):
    """Collector for goodjob_general (s3_3)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("source_name", None)
        super().__init__(source_name="goodjob_general", **kwargs)


@register_collector("goodjob_recommend")
class GoodjobRecommendCollector(GoodjobCollector):
    """Collector for goodjob_recommend (s3_2)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("source_name", None)
        super().__init__(source_name="goodjob_recommend", **kwargs)
