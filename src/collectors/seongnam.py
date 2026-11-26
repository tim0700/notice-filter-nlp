"""Collector for Seongnam Youth Portal (dept.seongnam.go.kr/youth)."""

from __future__ import annotations

import logging
import re
import ssl
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter

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

_ONCLICK_ID_RE = re.compile(
    r"(?:fn_view|fnView|goDetail|goView|view|detail)\s*\(\s*['\"]?([A-Za-z0-9_-]+)['\"]?"
)


class _LegacyTLSAdapter(HTTPAdapter):
    """TLS adapter allowing SECLEVEL=1 for legacy Korean municipal HTTPS servers."""

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> Any:
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


@register_collector("seongnam_youth")
class SeongnamYouthCollector(BaseCollector):
    """Collector for dept.seongnam.go.kr/youth public policy/business/notice endpoints."""

    source_name: str = "seongnam_youth"
    base_url: str = "https://dept.seongnam.go.kr/youth/contents.do?mId=0301000000"
    api_base_url: str = "https://dept.seongnam.go.kr/youth/sn-api/api"
    portal_base_url: str = "https://dept.seongnam.go.kr/youth"

    def __init__(
        self,
        fetch_detail: bool = True,
        use_sn_api: bool = True,
        max_pages: int = 20,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        delay_seconds: float = 0.25,
    ) -> None:
        super().__init__(session=session, timeout=timeout, delay_seconds=delay_seconds)
        self.fetch_detail = fetch_detail
        self.use_sn_api = use_sn_api
        self.max_pages = max_pages

    def _get_session(self) -> requests.Session:
        """Mount LegacyTLSAdapter on HTTPS so dept.seongnam.go.kr handshake succeeds."""
        if self._session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": self.user_agent})
            session.mount("https://", _LegacyTLSAdapter())
            self._session = session
        return self._session

    def check_robots(self, url: str) -> bool:
        """Check robots.txt via configured session (treating HTTP 404 as allowed per standard)."""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            resp = self._get_session().get(robots_url, timeout=self.timeout)
        except requests.RequestException:
            return False

        if resp.status_code == 404:
            return True
        if resp.status_code >= 400:
            return False

        parser = RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser.can_fetch(self.user_agent, url)

    def _build_list_url(self, page: int = 1) -> str:
        sep = "&" if "?" in self.base_url else "?"
        return f"{self.base_url}{sep}pageIndex={page}"

    def _extract_detail_url(self, anchor: Tag, list_url: str) -> tuple[str, str]:
        href = clean_whitespace(str(anchor.get("href", "")))
        onclick = clean_whitespace(str(anchor.get("onclick", "")))
        data_id = clean_whitespace(str(anchor.get("data-idx") or anchor.get("data-id") or ""))

        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            raw_id = data_id
            for js_source in (href, onclick):
                if not raw_id and js_source:
                    match = _ONCLICK_ID_RE.search(js_source)
                    if match:
                        raw_id = match.group(1)
            if not raw_id:
                return "", ""
            sep = "&" if "?" in self.base_url else "?"
            full_url = f"{self.base_url}{sep}idx={raw_id}"
            return full_url, raw_id

        full_url = urljoin(list_url, href)
        raw_id = extract_query_param(full_url, "idx", "nttId", "seq", "plcyId", "id") or data_id
        return full_url, raw_id

    @staticmethod
    def _find_title_anchor(row: Tag) -> Tag | None:
        for selector in (".subject a", ".title a", ".tit a", "h3 a", "h4 a", "td.tal a", "a"):
            anchor = row.select_one(selector)
            if anchor is not None and clean_whitespace(anchor.get_text(" ", strip=True)):
                return anchor
        return None

    def _extract_deadline(self, row: Tag, title: str) -> str:
        deadline_el = row.select_one(".deadline, td.deadline, span.deadline, .period, .apply_period")
        if deadline_el is not None:
            parsed = extract_deadline_from_text(deadline_el.get_text(" ", strip=True))
            if parsed:
                return parsed

        for el in row.select("td, span, p, li"):
            classes = " ".join(el.get("class", []))
            if "date" in classes and "deadline" not in classes:
                continue
            text = clean_whitespace(el.get_text(" ", strip=True))
            if "~" in text or any(k in text for k in ("마감", "신청기간", "모집기간", "접수기간")):
                parsed = extract_deadline_from_text(text)
                if parsed:
                    return parsed

        if "~" in title or any(k in title for k in ("마감", "까지")):
            return extract_deadline_from_text(title)
        return ""

    def _parse_list_html(self, html: str, list_url: str) -> list[NoticeRecord]:
        soup = BeautifulSoup(html, "lxml")
        rows = soup.select("table tbody tr")
        if not rows:
            rows = [tr for tr in soup.select("table tr") if not tr.find("th")]
        if not rows:
            rows = soup.select(
                "ul.board_list > li, ul.bbs_list > li, ul.card_list > li, "
                "ul.policy_list > li, div.card_list > div.card, div.list_item"
            )

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

            cate_el = row.select_one(".cate, .category, td.cate, span.cate, .badge_cate")
            raw_cate = clean_whitespace(cate_el.get_text(" ", strip=True)) if cate_el else ""
            category = raw_cate or extract_leading_category(title) or "청년정책"

            date_el = row.select_one(".date, td.date, span.date")
            posted_at = normalize_date(date_el.get_text(" ", strip=True) if date_el else "")
            if not posted_at:
                for el in row.select("td, span"):
                    candidate = normalize_date(el.get_text(" ", strip=True))
                    if candidate:
                        posted_at = candidate
                        break

            deadline = self._extract_deadline(row, title)

            meta_parts: list[str] = ["성남시"]
            for selector in (".tag", ".tags", ".badge", ".target", ".dept", ".region", "td.dept"):
                for el in row.select(selector):
                    meta_parts.append(el.get_text(" ", strip=True))

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

    def _parse_detail_html(self, html: str) -> tuple[str, str, str, list[str]]:
        soup = BeautifulSoup(html, "lxml")

        content_node: Tag | None = None
        for selector in (
            "div.board_view_con",
            "div.view_cont",
            "div.bbs_view_cont",
            "div.view_content",
            "div.contents_view",
            "div.detail_content",
            "div.board_view .content",
            "td.content",
            "article",
        ):
            content_node = soup.select_one(selector)
            if content_node is not None:
                break

        content = extract_html_text(content_node)
        posted_at = ""
        deadline = ""
        extra_tags: list[str] = []

        for label_el, value_el in self._iter_detail_pairs(soup):
            label = clean_whitespace(label_el.get_text(" ", strip=True))
            value = clean_whitespace(value_el.get_text(" ", strip=True))
            if not value:
                continue
            if not posted_at and any(k in label for k in ("등록일", "작성일", "게시일")):
                posted_at = normalize_date(value)
            if not deadline and any(k in label for k in ("신청기간", "접수기간", "모집기간", "마감", "기한")):
                deadline = extract_deadline_from_text(value)
            if any(k in label for k in ("지원대상", "대상", "담당부서", "지역", "분야")):
                extra_tags.append(value)

        if not deadline and content:
            deadline = extract_deadline_from_text(content)

        return content, posted_at, deadline, extra_tags

    def _post_api(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Issue a polite POST JSON request against Seongnam Youth sn-api."""
        if self._has_requested and self.delay_seconds > 0:
            import time

            time.sleep(self.delay_seconds)
        self._has_requested = True

        resp = self._get_session().post(
            f"{self.api_base_url}{path}",
            json=payload,
            headers={"Accept": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_business_item(self, item: dict[str, Any]) -> NoticeRecord | None:
        title = clean_whitespace(str(item.get("title", "")))
        seq = str(item.get("businessSeq", "") or "").strip()
        if not title or not seq:
            return None

        category = clean_whitespace(str(item.get("receptTypeCmmnClNm") or "청년지원사업"))
        raw_desc = str(item.get("businessDescriptionHtml") or item.get("businessDescription") or "")
        if "<" in raw_desc and ">" in raw_desc:
            desc = extract_html_text(BeautifulSoup(raw_desc, "lxml"))
        else:
            desc = clean_whitespace(raw_desc)
        target = clean_whitespace(str(item.get("target") or ""))
        content_parts = [p for p in (target, desc) if p and p != target[:len(p)]]
        content = clean_whitespace(" ".join(content_parts)) if desc != target else desc

        posted_at = normalize_date(str(item.get("announceStartDate") or item.get("registDt") or ""))
        deadline = normalize_date(str(item.get("endDate") or ""))
        meta_tags = join_meta_tags(
            "성남시",
            "분당구" if "분당" in (title + " " + content) else "",
            category,
            str(item.get("supportTypeCmmnClNm") or ""),
            str(item.get("addSearchKeyword") or ""),
        )
        url = f"{self.portal_base_url}/apply/business/detail/{seq}"
        return NoticeRecord(
            id=build_notice_id(self.source_name, f"biz_{seq}", url),
            source=self.source_name,
            category=category,
            title=title,
            meta_tags=meta_tags,
            posted_at=posted_at,
            url=url,
            content=content,
            deadline=deadline,
        )

    def _parse_notice_item(self, item: dict[str, Any]) -> NoticeRecord | None:
        title = clean_whitespace(str(item.get("title", "")))
        nid = str(item.get("noticeId", "") or "").strip()
        if not title or not nid:
            return None

        raw_content = str(item.get("content") or "")
        content = (
            extract_html_text(BeautifulSoup(raw_content, "lxml"))
            if "<" in raw_content and ">" in raw_content
            else clean_whitespace(raw_content)
        )
        posted_at = normalize_date(str(item.get("registDt") or ""))
        deadline = extract_deadline_from_text(content) if content else ""
        author = clean_whitespace(str(item.get("author") or ""))
        category = clean_whitespace(str(item.get("category") or "공지사항"))
        url = f"{self.portal_base_url}/news/news/youth/detail/{nid}"
        return NoticeRecord(
            id=build_notice_id(self.source_name, f"notice_{nid}", url),
            source=self.source_name,
            category=category,
            title=title,
            meta_tags=join_meta_tags("성남시", author),
            posted_at=posted_at,
            url=url,
            content=content,
            deadline=deadline,
        )

    def _fetch_sn_api(self, limit: int) -> list[NoticeRecord]:
        collected: list[NoticeRecord] = []
        seen_ids: set[str] = set()

        # 1. Notice list
        try:
            notice_res = self._post_api("/notice/list", {"pageNo": 1, "pageSize": min(50, limit)})
            for item in notice_res.get("data") or []:
                if not isinstance(item, dict):
                    continue
                rec = self._parse_notice_item(item)
                if rec and rec.id not in seen_ids:
                    seen_ids.add(rec.id)
                    collected.append(rec)
                    if len(collected) >= limit:
                        return collected
        except requests.RequestException as exc:
            logger.warning("Seongnam notice API failed: %s", exc)

        # 2. Youth business/support programs
        for page in range(1, self.max_pages + 1):
            if len(collected) >= limit:
                break
            try:
                biz_res = self._post_api("/business/list", {"pageNo": page, "pageSize": 20})
            except requests.RequestException as exc:
                logger.warning("Seongnam business API failed (page=%d): %s", page, exc)
                break

            data_obj = biz_res.get("data") or {}
            items = data_obj.get("response") if isinstance(data_obj, dict) else data_obj
            if not items:
                break

            added = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                rec = self._parse_business_item(item)
                if rec and rec.id not in seen_ids:
                    seen_ids.add(rec.id)
                    collected.append(rec)
                    added += 1
                    if len(collected) >= limit:
                        break
            if added == 0:
                break

        return collected[:limit]

    def fetch(self, limit: int = 100) -> list[NoticeRecord]:
        """Collect public notices/programs via sn-api JSON (with HTML board fallback)."""
        if limit <= 0:
            return []

        first_list_url = self._build_list_url(1)
        if not self._is_allowed(first_list_url):
            logger.warning("robots.txt disallowed crawling %s", first_list_url)
            return []

        if self.use_sn_api:
            api_records = self._fetch_sn_api(limit)
            if api_records:
                return api_records

        collected: list[NoticeRecord] = []
        seen_ids: set[str] = set()

        for page in range(1, self.max_pages + 1):
            if len(collected) >= limit:
                break
            response = self._request(self.base_url, params={"pageIndex": page})
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
                    content, detail_posted, detail_deadline, extra_tags = self._parse_detail_html(
                        detail_resp.text
                    )
                    record.content = content
                    if not record.posted_at and detail_posted:
                        record.posted_at = detail_posted
                    if not record.deadline and detail_deadline:
                        record.deadline = detail_deadline
                    if extra_tags:
                        record.meta_tags = join_meta_tags(record.meta_tags, *extra_tags)
                except requests.RequestException as exc:
                    logger.warning("Failed to fetch detail for %s: %s", record.id, exc)

        return collected[:limit]
