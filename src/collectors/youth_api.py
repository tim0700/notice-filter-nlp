"""Collector for YouthCenter (온통청년) Open API v2 supporting JSON and XML."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import defusedxml.ElementTree as ET
import requests
from bs4 import BeautifulSoup

from src.collectors.base import (
    BaseCollector,
    build_notice_id,
    clean_whitespace,
    extract_deadline_from_text,
    extract_html_text,
    join_meta_tags,
    normalize_date,
    register_collector,
)
from src.config import load_env
from src.schema import NoticeRecord

logger = logging.getLogger(__name__)


_API_KEY_MASK_RE = re.compile(r"(apiKeyNm=)[^&\s'\"]+")


def _mask_api_key(text: str) -> str:
    """Redact YouthCenter API keys before logging URLs or request exception strings."""
    return _API_KEY_MASK_RE.sub(r"\1***", str(text))


def _first_non_empty(item: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_whitespace(item.get(key, ""))
        if value and value != "-":
            return value
    return ""


@register_collector("youth_api")
class YouthApiCollector(BaseCollector):
    """Optional YouthCenter Open API v2 collector with graceful skip when keys are missing."""

    source_name: str = "youth_api"
    base_url: str = "https://www.youthcenter.go.kr/go/ythip/getPlcy"
    content_api_url: str = "https://www.youthcenter.go.kr/go/ythip/getContent"

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
                raw_items = (
                    result.get("youthPolicyList")
                    or result.get("youthContentList")
                    or result.get("items")
                )
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
                normalized.append(
                    {
                        str(k): str(v).strip() if v is not None else ""
                        for k, v in entry.items()
                    }
                )
        return normalized

    def _extract_xml_items(self, text: str) -> list[dict[str, str]]:
        root = ET.fromstring(text)
        err_msg = root.findtext(".//errMsg") or root.findtext(".//resultMessage")
        if err_msg and not any(
            root.findall(tag)
            for tag in (".//youthPolicy", ".//emp", ".//youthContent", ".//item")
        ):
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
        title = _first_non_empty(item, ("plcyNm", "polyBizSjnm", "pstTtl", "cntntNm", "title", "sj"))
        if not title:
            return None

        raw_id = _first_non_empty(
            item,
            ("plcyNo", "bizId", "plcyId", "pstSn", "cntntId", "id", "rnum"),
        )
        default_view = (
            f"https://www.youthcenter.go.kr/youthCntnt/view.do?cntntId={raw_id or '0'}"
            if is_content or "pstSn" in item
            else f"https://www.youthcenter.go.kr/youthPolicy/ythPlcyTotalSearch/ythPlcyDetail/{raw_id or '0'}"
        )
        url = (
            _first_non_empty(item, ("refUrlAddr1", "aplyUrlAddr", "rfcSiteUrla1", "url", "link"))
            or default_view
        )
        notice_id = build_notice_id(self.source_name, raw_id, url)

        default_category = "청년콘텐츠" if (is_content or "pstSn" in item) else "청년정책"
        category = (
            _first_non_empty(
                item,
                ("lclsfNm", "mclsfNm", "plcyTpNm", "pstSeNm", "cntntClsfNm", "category"),
            )
            or default_category
        )

        content_parts: list[str] = []
        for key in (
            "plcyExplnCn",
            "plcySprtCn",
            "plcyAplyMthdCn",
            "polyItcnCn",
            "sporCn",
            "prcpCn",
            "rqutProcCn",
            "pstWholCn",
            "cntntCn",
            "content",
        ):
            raw_val = str(item.get(key, "") or "").strip()
            if not raw_val or raw_val == "-":
                continue
            cleaned_part = (
                extract_html_text(BeautifulSoup(raw_val, "lxml"))
                if "<" in raw_val and ">" in raw_val
                else clean_whitespace(raw_val)
            )
            if cleaned_part:
                content_parts.append(cleaned_part)
        content = clean_whitespace(" ".join(content_parts))

        age_min = clean_whitespace(item.get("sprtTrgtMinAge", ""))
        age_max = clean_whitespace(item.get("sprtTrgtMaxAge", ""))
        age_span = f"만 {age_min}~{age_max}세" if age_min and age_max and (age_min != "0" or age_max != "0") else ""

        meta_tags = join_meta_tags(
            item.get("sprvsnInstCdNm", ""),
            item.get("operInstCdNm", ""),
            item.get("rgtrInstCdNm", ""),
            item.get("plcyKywdNm", ""),
            item.get("mclsfNm", ""),
            item.get("polyBizSecd", ""),
            item.get("ageInfo", "") or age_span,
            item.get("region", ""),
        )
        posted_at = normalize_date(
            _first_non_empty(item, ("frstRegDt", "lastMdfcnDt", "regDt", "posted_at"))
        )
        deadline = extract_deadline_from_text(
            _first_non_empty(item, ("aplyYmd", "rqutPrdCn", "bizPrdEndYmd", "deadline"))
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
        extra_params: dict[str, Any] | None = None,
    ) -> list[NoticeRecord]:
        collected: list[NoticeRecord] = []
        for page in range(1, self.max_pages + 1):
            remaining = limit - len(collected)
            if remaining <= 0:
                break
            params: dict[str, Any] = {
                "apiKeyNm": api_key,
                "pageNum": page,
                "pageSize": min(self.page_size, remaining),
                "rtnType": "json",
            }
            if extra_params:
                params.update(extra_params)

            try:
                response = self._request(endpoint_url, params=params)
            except requests.RequestException as exc:
                logger.warning(
                    "Youth API request failed (%s): %s",
                    endpoint_url,
                    _mask_api_key(str(exc)),
                )
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
        """Load API keys from env/.env and collect YouthCenter v2 notices, or skip with warning."""
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
            # Query Seongnam (zipCd 41135) first for high local relevance, then general nationwide policies.
            seongnam_limit = min(limit, max(20, limit // 3))
            records.extend(
                self._fetch_endpoint(
                    self.base_url,
                    api_key=policy_key,
                    limit=seongnam_limit,
                    seen_ids=seen_ids,
                    is_content=False,
                    extra_params={"zipCd": "41135"},
                )
            )
            if len(records) < limit:
                records.extend(
                    self._fetch_endpoint(
                        self.base_url,
                        api_key=policy_key,
                        limit=limit - len(records),
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
