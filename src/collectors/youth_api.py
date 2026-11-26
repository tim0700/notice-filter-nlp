"""Collector for YouthCenter (온통청년) Open API supporting JSON and XML."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import defusedxml.ElementTree as ET
import requests

from src.collectors.base import (
    BaseCollector,
    build_notice_id,
    clean_whitespace,
    extract_deadline_from_text,
    join_meta_tags,
    normalize_date,
    register_collector,
)
from src.config import load_env
from src.schema import NoticeRecord

logger = logging.getLogger(__name__)


def _first_non_empty(item: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_whitespace(item.get(key, ""))
        if value and value != "-":
            return value
    return ""


@register_collector("youth_api")
class YouthApiCollector(BaseCollector):
    """Optional YouthCenter Open API collector with graceful skip when keys are missing."""

    source_name: str = "youth_api"
    base_url: str = "https://www.youthcenter.go.kr/opi/youthPlcyList.do"
    content_api_url: str = "https://www.youthcenter.go.kr/opi/youthCntntList.do"

    def __init__(
        self,
        max_pages: int = 10,
        page_size: int = 50,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        delay_seconds: float = 0.25,
    ) -> None:
        super().__init__(session=session, timeout=timeout, delay_seconds=delay_seconds)
        self.max_pages = max_pages
        self.page_size = page_size

    def _extract_json_items(self, data: Any) -> list[dict[str, str]]:
        raw_items: Any = None
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            for key in ("youthPolicyList", "youthContentList", "items", "data"):
                if key in data:
                    raw_items = data[key]
                    break
            if raw_items is None and isinstance(data.get("result"), dict):
                result = data["result"]
                raw_items = result.get("youthPolicyList") or result.get("youthContentList") or result.get("items")
            if raw_items is None and isinstance(data.get("empsInfo"), dict):
                raw_items = data["empsInfo"].get("emp")

            if isinstance(raw_items, dict):
                raw_items = (
                    raw_items.get("youthPolicy")
                    or raw_items.get("youthContent")
                    or raw_items.get("item")
                    or [raw_items]
                )

        if not isinstance(raw_items, list):
            return []

        normalized: list[dict[str, str]] = []
        for entry in raw_items:
            if isinstance(entry, dict):
                normalized.append({str(k): clean_whitespace(str(v)) if v is not None else "" for k, v in entry.items()})
        return normalized

    def _extract_xml_items(self, text: str) -> list[dict[str, str]]:
        root = ET.fromstring(text)
        err_msg = root.findtext(".//errMsg") or root.findtext(".//resultMessage")
        if err_msg and not any(root.findall(tag) for tag in (".//youthPolicy", ".//emp", ".//youthContent", ".//item")):
            logger.warning("Youth API returned XML message: %s", clean_whitespace(err_msg))
            return []

        elements: list[Any] = []
        for tag in (".//youthPolicy", ".//emp", ".//youthContent", ".//item"):
            found = root.findall(tag)
            if found:
                elements = found
                break

        items: list[dict[str, str]] = []
        for elem in elements:
            row = {child.tag: clean_whitespace(child.text) for child in elem}
            if row:
                items.append(row)
        return items

    def _parse_payload(self, text: str, content_type: str = "") -> list[dict[str, str]]:
        stripped = text.strip()
        if not stripped:
            return []

        looks_json = "json" in content_type.lower() or stripped.startswith(("{", "["))
        if looks_json:
            try:
                return self._extract_json_items(json.loads(stripped))
            except json.JSONDecodeError:
                if not stripped.startswith("<"):
                    logger.warning("Failed to decode Youth API JSON payload.")
                    return []

        try:
            return self._extract_xml_items(stripped)
        except ET.ParseError:
            logger.warning("Failed to parse Youth API XML payload.")
            return []

    def _item_to_record(self, item: dict[str, str], is_content: bool = False) -> NoticeRecord | None:
        title = _first_non_empty(item, ("polyBizSjnm", "plcyNm", "cntntNm", "title", "sj"))
        if not title:
            return None

        raw_id = _first_non_empty(item, ("bizId", "plcyNo", "plcyId", "cntntId", "id", "rnum"))
        default_view = (
            f"https://www.youthcenter.go.kr/youthCntnt/view.do?cntntId={raw_id or '0'}"
            if is_content
            else f"https://www.youthcenter.go.kr/youthPlcy/view.do?plcyId={raw_id or '0'}"
        )
        url = _first_non_empty(item, ("rfcSiteUrla1", "aplyUrlAddr", "url", "link")) or default_view
        notice_id = build_notice_id(self.source_name, raw_id, url)

        default_category = "청년콘텐츠" if is_content else "청년정책"
        category = (
            _first_non_empty(item, ("plcyTpNm", "lclsfNm", "mclsfNm", "cntntClsfNm", "category"))
            or default_category
        )

        content_parts = [
            clean_whitespace(item.get(key, ""))
            for key in (
                "polyItcnCn",
                "plcyExplnCn",
                "sporCn",
                "plcySprtCn",
                "prcpCn",
                "rqutProcCn",
                "cntntCn",
                "content",
            )
        ]
        content = clean_whitespace(" ".join(part for part in content_parts if part and part != "-"))

        meta_tags = join_meta_tags(
            *(
                item.get(key, "")
                for key in (
                    "polyBizSecd",
                    "polyBizTy",
                    "rgtrInstCdNm",
                    "sprvsnInstCdNm",
                    "ageInfo",
                    "plcyKywdNm",
                    "region",
                )
            )
        )
        posted_at = normalize_date(
            _first_non_empty(item, ("frstRegDt", "lastMdfcnDt", "regDt", "posted_at"))
        )
        deadline = extract_deadline_from_text(
            _first_non_empty(item, ("rqutPrdCn", "aplyYmd", "deadline"))
        )

        return NoticeRecord(
            id=notice_id,
            source=self.source_name,
            category=category,
            title=title,
            meta_tags=meta_tags,
            posted_at=posted_at,
            url=url,
            content=content,
            deadline=deadline,
        )

    def _fetch_endpoint(
        self,
        endpoint_url: str,
        api_key: str,
        limit: int,
        seen_ids: set[str],
        is_content: bool = False,
    ) -> list[NoticeRecord]:
        collected: list[NoticeRecord] = []
        for page in range(1, self.max_pages + 1):
            remaining = limit - len(collected)
            if remaining <= 0:
                break
            params = {
                "openApiVlak": api_key,
                "display": min(self.page_size, remaining),
                "pageIndex": page,
            }
            try:
                response = self._request(endpoint_url, params=params)
            except requests.RequestException as exc:
                logger.warning("Youth API request failed (%s): %s", endpoint_url, exc)
                break

            content_type = response.headers.get("Content-Type", "")
            raw_items = self._parse_payload(response.text, content_type=content_type)
            if not raw_items:
                break

            added_in_page = 0
            for raw in raw_items:
                record = self._item_to_record(raw, is_content=is_content)
                if record is None or record.id in seen_ids:
                    continue
                seen_ids.add(record.id)
                collected.append(record)
                added_in_page += 1
                if len(collected) >= limit:
                    break

            if added_in_page == 0:
                break
        return collected

    def fetch(self, limit: int = 100) -> list[NoticeRecord]:
        """Load API keys from env/.env and collect YouthCenter notices, or skip with warning."""
        load_env()
        policy_key = os.getenv("YOUTH_POLICY_API_KEY", "").strip()
        content_key = os.getenv("YOUTH_CONTENT_API_KEY", "").strip()

        if not policy_key and not content_key:
            logger.warning("YOUTH_POLICY_API_KEY is empty; skipping youth_api collection.")
            return []

        if limit <= 0:
            return []

        records: list[NoticeRecord] = []
        seen_ids: set[str] = set()

        if policy_key:
            records.extend(
                self._fetch_endpoint(
                    self.base_url,
                    api_key=policy_key,
                    limit=limit,
                    seen_ids=seen_ids,
                    is_content=False,
                )
            )
        else:
            logger.warning(
                "YOUTH_POLICY_API_KEY is empty; collecting only from YOUTH_CONTENT_API_KEY."
            )

        if content_key and len(records) < limit:
            records.extend(
                self._fetch_endpoint(
                    self.content_api_url,
                    api_key=content_key,
                    limit=limit - len(records),
                    seen_ids=seen_ids,
                    is_content=True,
                )
            )

        return records[:limit]
