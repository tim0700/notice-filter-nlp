"""Offline unit tests for Phase 1 notice collectors and runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

import src.collectors as collectors_pkg
from src.collectors import (
    COLLECTOR_REGISTRY,
    BaseCollector,
    GoodjobCollector,
    GoodjobExtracurricularCollector,
    GoodjobGeneralCollector,
    GoodjobRecommendCollector,
    SeongnamYouthCollector,
    SwconAcademicCollector,
    SwconCareerCollector,
    SwconCollector,
    YouthApiCollector,
    collect_all,
)
from src.collectors.base import (
    build_notice_id,
    extract_deadline_from_text,
    join_meta_tags,
    normalize_date,
)
from src.collectors.runner import main as runner_main
from src.config import CANONICAL_COLUMNS
from src.schema import NoticeRecord, validate_notices_df


class _FakeResponse:
    """Minimal offline replacement for requests.Response."""

    def __init__(
        self,
        text: str,
        url: str = "https://example.com/",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.text = text
        self.url = url
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> Any:
        return json.loads(self.text)


class _FakeSession:
    """Offline requests.Session mock dispatching URL+params routes."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
    ) -> _FakeResponse:
        self.calls.append((url, params))
        return self.handler(url, params)

    def post(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - safety guard
        raise AssertionError("POST/login requests are forbidden in public collectors.")


def test_registry_and_package_exports() -> None:
    expected_sources = {
        "swcon_academic",
        "swcon_career",
        "goodjob_extracurricular",
        "goodjob_general",
        "goodjob_recommend",
        "seongnam_youth",
        "youth_api",
    }
    assert expected_sources.issubset(COLLECTOR_REGISTRY.keys())
    for source_name in expected_sources:
        cls = COLLECTOR_REGISTRY[source_name]
        assert issubclass(cls, BaseCollector)
        instance = cls()
        assert instance.source_name == source_name

    assert hasattr(collectors_pkg, "collect_all")
    assert SwconAcademicCollector is COLLECTOR_REGISTRY["swcon_academic"]
    assert SwconCareerCollector is COLLECTOR_REGISTRY["swcon_career"]
    assert GoodjobExtracurricularCollector is COLLECTOR_REGISTRY["goodjob_extracurricular"]
    assert GoodjobGeneralCollector is COLLECTOR_REGISTRY["goodjob_general"]
    assert GoodjobRecommendCollector is COLLECTOR_REGISTRY["goodjob_recommend"]
    assert SeongnamYouthCollector is COLLECTOR_REGISTRY["seongnam_youth"]
    assert YouthApiCollector is COLLECTOR_REGISTRY["youth_api"]


def test_base_robots_and_normalization_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    class _AllowedParser:
        def __init__(self) -> None:
            self.url = ""

        def set_url(self, url: str) -> None:
            self.url = url

        def read(self) -> None:
            return None

        def can_fetch(self, user_agent: str, url: str) -> bool:
            return "blocked" not in url

    monkeypatch.setattr("src.collectors.base.RobotFileParser", _AllowedParser)
    collector = GoodjobGeneralCollector()
    assert collector.check_robots("https://goodjob.khu.ac.kr/bbs/board.php?bo_table=s3_3") is True
    assert collector.check_robots("https://goodjob.khu.ac.kr/blocked") is False

    class _ErrorParser(_AllowedParser):
        def read(self) -> None:
            raise OSError("robots unreachable")

    monkeypatch.setattr("src.collectors.base.RobotFileParser", _ErrorParser)
    assert collector.check_robots("https://goodjob.khu.ac.kr/bbs/board.php") is False

    assert normalize_date("2026.11.20") == "2026-11-20"
    assert normalize_date("2026년 12월 5일") == "2026-12-05"
    assert normalize_date("20261215") == "2026-12-15"
    assert normalize_date("invalid-date") == ""

    assert extract_deadline_from_text("2026.11.01 ~ 2026.12.15") == "2026-12-15"
    assert extract_deadline_from_text("마감일: 2026-12-09") == "2026-12-09"
    assert join_meta_tags("국제캠퍼스|AI", "AI|게임", "", "백엔드") == "국제캠퍼스|AI|게임|백엔드"
    assert build_notice_id("swcon_academic", "1001", "https://example.com") == "swcon_academic_1001"
    assert build_notice_id("swcon_academic", "", "https://example.com/a").startswith("swcon_academic_")


def test_swcon_collector_pagination_and_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    page1_html = """
    <html><body>
      <table class="board_list">
        <tbody>
          <tr class="notice">
            <td class="num">공지</td>
            <td class="cate">학사</td>
            <td class="subject"><a href="/swcon/user/bbs/BMSR00040/view.do?menuNo=21300017&nttId=1001">[필독] 졸업논문 심사 안내</a></td>
            <td class="writer">학과사무실</td>
            <td class="date">2026.11.20</td>
          </tr>
          <tr>
            <td class="num">12</td>
            <td class="cate">장학</td>
            <td class="subject"><a href="javascript:view('1002');">SW중심대학 우수장학생 선발</a></td>
            <td class="writer">사업단</td>
            <td class="date">2026-11-18</td>
          </tr>
        </tbody>
      </table>
    </body></html>
    """
    page2_html = """
    <html><body>
      <table class="board_list">
        <tbody>
          <tr>
            <td class="num">11</td>
            <td class="cate">특강</td>
            <td class="subject"><a href="/swcon/user/bbs/BMSR00040/view.do?menuNo=21300018&nttId=2001">생성형 AI 실무 특강</a></td>
            <td class="writer">소프트웨어융합대학</td>
            <td class="date">2026.11.22</td>
          </tr>
        </tbody>
      </table>
    </body></html>
    """
    detail_1001 = """
    <html><body>
      <table><tr><th>신청기간</th><td>2026-11-20 ~ 2026-12-11</td></tr></table>
      <div class="board_view_con">
        <script>alert('ignore');</script>
        졸업심사 일정을 안내합니다. 캡스톤 결과보고서를 12월 11일까지 제출하세요.
      </div>
    </body></html>
    """
    detail_1002 = """
    <html><body>
      <div class="board_view_con">산학협력 우수장학생을 선발합니다.</div>
    </body></html>
    """

    def handler(url: str, params: dict[str, Any] | None) -> _FakeResponse:
        if "list.do" in url:
            assert params is not None
            if params["menuNo"] == "21300017":
                if params["pageIndex"] == 1:
                    return _FakeResponse(page1_html, url=url)
                return _FakeResponse("<html><body><table><tbody><tr><td>등록된 게시물이 없습니다.</td></tr></tbody></table></body></html>", url=url)
            if params["menuNo"] == "21300018":
                if params["pageIndex"] == 1:
                    return _FakeResponse(page2_html, url=url)
                return _FakeResponse("<html><body></body></html>", url=url)
        if "nttId=1001" in url:
            return _FakeResponse(detail_1001, url=url)
        if "boardId=1002" in url or "nttId=1002" in url:
            return _FakeResponse(detail_1002, url=url)
        if "nttId=2001" in url or "boardId=2001" in url:
            raise requests.RequestException("detail timeout")
        raise AssertionError(f"Unexpected URL: {url}")

    session = _FakeSession(handler)
    academic = SwconCollector(source_name="swcon_academic", session=session, delay_seconds=0)
    monkeypatch.setattr(academic, "check_robots", lambda url: True)
    records = academic.fetch(limit=10)

    assert len(records) == 2
    assert records[0].id == "swcon_academic_1001"
    assert records[0].category == "학사"
    assert records[0].posted_at == "2026-11-20"
    assert records[0].deadline == "2026-12-11"
    assert "졸업심사 일정을 안내합니다" in records[0].content
    assert "alert" not in records[0].content
    assert "공지" in records[0].meta_tags
    assert records[1].id == "swcon_academic_1002"
    assert records[1].url.endswith("menuNo=21300017&boardId=1002")
    assert "산학협력 우수장학생" in records[1].content

    career = SwconCareerCollector(session=session, delay_seconds=0)
    monkeypatch.setattr(career, "check_robots", lambda url: True)
    career_records = career.fetch(limit=5)
    assert len(career_records) == 1
    assert career_records[0].id == "swcon_career_2001"
    assert career_records[0].content == ""  # graceful fallback on detail RequestException

    # robots.txt disallow test
    disallowed = SwconAcademicCollector(session=session, delay_seconds=0)
    monkeypatch.setattr(disallowed, "check_robots", lambda url: False)
    assert disallowed.fetch(limit=5) == []

    # fetch_detail=False skips detail page fetch
    no_detail_session = _FakeSession(handler)
    no_detail = SwconAcademicCollector(fetch_detail=False, session=no_detail_session, delay_seconds=0)
    monkeypatch.setattr(no_detail, "check_robots", lambda url: True)
    no_detail_records = no_detail.fetch(limit=2)
    assert all(rec.content == "" for rec in no_detail_records)
    assert all("view.do" not in call_url for call_url, _ in no_detail_session.calls)


def test_goodjob_collector_public_list_only(monkeypatch: pytest.MonkeyPatch) -> None:
    extra_html = """
    <html><body>
      <div class="tbl_head01"><table><tbody>
        <tr>
          <td class="td_num">10</td>
          <td class="td_subject">
            <a href="/bbs/board.php?bo_table=s4_1&sca=%EA%B5%AD%EC%A0%9C" class="bo_cate_link">국제</a>
            <div class="bo_tit"><a href="/bbs/board.php?bo_table=s4_1&wr_id=3001">[국제] 클라우드 보안 부트캠프</a></div>
          </td>
          <td class="td_job">보안|클라우드</td>
          <td class="td_datetime">2026-11-23</td>
          <td class="td_deadline">~ 2026-12-10</td>
        </tr>
      </tbody></table></div>
    </body></html>
    """
    general_html = """
    <html><body>
      <div class="tbl_head01"><table><tbody>
        <tr>
          <td class="td_num">1</td>
          <td class="td_subject">
            <div class="bo_tit"><a href="/bbs/board.php?bo_table=s3_3&wr_id=4001">[넥슨코리아] 게임 클라이언트 인턴십</a></div>
          </td>
          <td class="td_company">판교|인턴</td>
          <td class="td_datetime">2026.11.24</td>
          <td class="td_deadline">2026-12-09 마감</td>
        </tr>
        <tr>
          <td class="td_num">2</td>
          <td class="td_subject">
            <div class="bo_tit"><a href="/bbs/board.php?bo_table=s3_3&wr_id=4002">[네이버클라우드] 백엔드 신입 개발자</a></div>
          </td>
          <td class="td_company">IT/SW개발</td>
          <td class="td_datetime">2026-11-23</td>
        </tr>
      </tbody></table></div>
    </body></html>
    """

    def handler(url: str, params: dict[str, Any] | None) -> _FakeResponse:
        assert params is not None
        assert "wr_id" not in url  # ADR-02: never fetch detail page
        if params["bo_table"] == "s4_1":
            assert params.get("sca") == "국제"
            if params.get("page", 1) == 1:
                return _FakeResponse(extra_html, url=url)
            return _FakeResponse("<html><body><table><tbody><tr><td>게시물이 없습니다.</td></tr></tbody></table></body></html>", url=url)
        if params["bo_table"] in {"s3_3", "s3_2"}:
            if params.get("page", 1) == 1:
                return _FakeResponse(general_html, url=url)
            return _FakeResponse("<html><body></body></html>", url=url)
        raise AssertionError(f"Unexpected params: {params}")

    session = _FakeSession(handler)
    extra = GoodjobExtracurricularCollector(session=session, delay_seconds=0)
    monkeypatch.setattr(extra, "check_robots", lambda url: True)
    extra_records = extra.fetch(limit=10)
    assert len(extra_records) == 1
    assert extra_records[0].id == "goodjob_extracurricular_3001"
    assert extra_records[0].category == "국제"
    assert extra_records[0].content == ""
    assert extra_records[0].deadline == "2026-12-10"
    assert "보안" in extra_records[0].meta_tags
    assert "국제캠퍼스" in extra_records[0].meta_tags

    general = GoodjobCollector(source_name="goodjob_general", session=session, delay_seconds=0)
    monkeypatch.setattr(general, "check_robots", lambda url: True)
    gen_records = general.fetch(limit=10)
    assert len(gen_records) == 2
    assert gen_records[0].id == "goodjob_general_4001"
    assert gen_records[0].content == ""
    assert gen_records[0].deadline == "2026-12-09"
    assert gen_records[1].deadline == ""  # must not confuse posted_at with deadline

    recommend = GoodjobRecommendCollector(session=session, delay_seconds=0)
    monkeypatch.setattr(recommend, "check_robots", lambda url: True)
    rec_records = recommend.fetch(limit=1)
    assert len(rec_records) == 1
    assert rec_records[0].source == "goodjob_recommend"
    assert rec_records[0].content == ""


def test_seongnam_youth_collector_list_and_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    list_html = """
    <html><body>
      <ul class="policy_list">
        <li>
          <span class="badge_cate">청년복지/자격증</span>
          <h3><a href="/youth/contents.do?mId=0301000000&idx=6001">2026년 성남시 청년 자격증 응시료 지원사업</a></h3>
          <span class="tag">분당구</span>
          <span class="date">2026-11-20</span>
        </li>
        <li>
          <h3><a href="#detail" onclick="fn_view('6002')">성남시 청년 전월세 대출이자 지원</a></h3>
          <span class="date">2026-11-18</span>
        </li>
      </ul>
    </body></html>
    """
    detail_6001 = """
    <html><body>
      <table>
        <tr><th>신청기간</th><td>2026.11.20 ~ 2026.12.15</td></tr>
        <tr><th>지원대상</th><td>만 19~34세 미취업 청년</td></tr>
      </table>
      <div class="view_cont">
        성남시에 주민등록을 둔 청년에게 자격증 응시료를 최대 100만 원 지원합니다.
      </div>
    </body></html>
    """

    def handler(url: str, params: dict[str, Any] | None) -> _FakeResponse:
        if "idx=6001" in url:
            return _FakeResponse(detail_6001, url=url)
        if "idx=6002" in url:
            raise requests.RequestException("network error")
        page = (params or {}).get("pageIndex", 1)
        if page == 1:
            return _FakeResponse(list_html, url=url)
        return _FakeResponse("<html><body></body></html>", url=url)

    session = _FakeSession(handler)
    collector = SeongnamYouthCollector(use_sn_api=False, session=session, delay_seconds=0)
    monkeypatch.setattr(collector, "check_robots", lambda url: True)
    records = collector.fetch(limit=10)

    assert len(records) == 2
    assert records[0].id == "seongnam_youth_6001"
    assert records[0].category == "청년복지/자격증"
    assert records[0].deadline == "2026-12-15"
    assert "자격증 응시료를 최대 100만 원 지원" in records[0].content
    assert "성남시" in records[0].meta_tags
    assert "분당구" in records[0].meta_tags
    assert "만 19~34세 미취업 청년" in records[0].meta_tags

    assert records[1].id == "seongnam_youth_6002"
    assert records[1].content == ""


def test_youth_api_collector_json_xml_and_missing_keys(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("src.collectors.youth_api.load_env", lambda *args, **kwargs: None)
    monkeypatch.delenv("YOUTH_POLICY_API_KEY", raising=False)
    monkeypatch.delenv("YOUTH_CONTENT_API_KEY", raising=False)

    with caplog.at_level(logging.WARNING):
        skipped = YouthApiCollector(delay_seconds=0).fetch(limit=10)
    assert skipped == []
    assert any("YOUTH_POLICY_API_KEY" in rec.message for rec in caplog.records)

    # JSON policy response + XML content response
    json_payload = json.dumps(
        {
            "result": {
                "youthPolicyList": [
                    {
                        "bizId": "P2026001",
                        "polyBizSjnm": "성남시 청년 월세 지원",
                        "plcyTpNm": "주거",
                        "polyItcnCn": "성남시 거주 대학생 대상 월세 지원",
                        "sporCn": "월 최대 20만 원 지원",
                        "polyBizSecd": "경기도 성남시",
                        "ageInfo": "만 19세~34세",
                        "frstRegDt": "2026-11-17",
                        "rqutPrdCn": "2026-11-17 ~ 2026-12-18",
                        "rfcSiteUrla1": "https://www.youthcenter.go.kr/youthPlcy/view.do?plcyId=7001",
                    }
                ]
            }
        }
    )
    xml_content_payload = """<?xml version="1.0" encoding="UTF-8"?>
    <youthContentList>
      <youthContent>
        <cntntId>C9001</cntntId>
        <cntntNm>청년 취업준비 가이드</cntntNm>
        <cntntClsfNm>취업콘텐츠</cntntClsfNm>
        <cntntCn>포트폴리오 작성 요령 안내</cntntCn>
        <regDt>20261121</regDt>
      </youthContent>
    </youthContentList>
    """

    def handler(url: str, params: dict[str, Any] | None) -> _FakeResponse:
        assert params is not None
        if params["pageNum"] > 1:
            return _FakeResponse("{}", url=url, headers={"Content-Type": "application/json"})
        if "getPlcy" in url:
            assert params["apiKeyNm"] == "policy-key"
            if params.get("zipCd") == "41135":
                return _FakeResponse(
                    json_payload,
                    url=url,
                    headers={"Content-Type": "application/json;charset=utf-8"},
                )
            return _FakeResponse("{}", url=url, headers={"Content-Type": "application/json"})
        assert params["apiKeyNm"] == "content-key"
        return _FakeResponse(
            xml_content_payload,
            url=url,
            headers={"Content-Type": "application/xml;charset=utf-8"},
        )

    monkeypatch.setenv("YOUTH_POLICY_API_KEY", "policy-key")
    monkeypatch.setenv("YOUTH_CONTENT_API_KEY", "content-key")
    session = _FakeSession(handler)
    api_records = YouthApiCollector(session=session, delay_seconds=0).fetch(limit=10)

    assert len(api_records) == 2
    assert api_records[0].id == "youth_api_P2026001"
    assert api_records[0].deadline == "2026-12-18"
    assert "월 최대 20만 원 지원" in api_records[0].content
    assert api_records[1].id == "youth_api_C9001"
    assert api_records[1].posted_at == "2026-11-21"

    # Also verify XML policy parsing directly
    xml_policy_payload = """<?xml version="1.0" encoding="UTF-8"?>
    <youthPolicyList>
      <youthPolicy>
        <bizId>X7002</bizId>
        <polyBizSjnm>청년 SW 직무훈련</polyBizSjnm>
        <plcyTpNm>일자리</plcyTpNm>
        <polyItcnCn>AI/백엔드 실무 교육 지원</polyItcnCn>
        <rqutPrdCn>2026.12.20까지</rqutPrdCn>
      </youthPolicy>
    </youthPolicyList>
    """
    monkeypatch.delenv("YOUTH_CONTENT_API_KEY", raising=False)
    xml_session = _FakeSession(
        lambda url, params: _FakeResponse(
            xml_policy_payload
            if (params or {}).get("pageNum", 1) == 1 and (params or {}).get("zipCd") == "41135"
            else "<youthPolicyList/>",
            url=url,
            headers={"Content-Type": "application/xml"},
        )
    )
    xml_records = YouthApiCollector(session=xml_session, delay_seconds=0).fetch(limit=5)
    assert len(xml_records) == 1
    assert xml_records[0].id == "youth_api_X7002"
    assert xml_records[0].deadline == "2026-12-20"


def test_live_dom_regressions_goodjob_div_table_and_seongnam_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Goodjob div_tb_tr layout (s3_2 recommend with both col_subject and col_subject_sub)
    div_html = """
    <html><body>
      <div class="div_tb_tr bo_notice">
        <div class="div_td col_num"><strong>공지</strong></div>
        <div class="div_td">국제</div>
        <div class="div_td">중견/중소기업</div>
        <div class="div_td col_subject ta_l"><a href="https://goodjob.khu.ac.kr/bbs/board.php?bo_table=s3_2&wr_id=2171">[(주)코스모웍스] 연구개발 신입 모집</a></div>
        <div class="div_td col_subject_sub"><a href="https://goodjob.khu.ac.kr/bbs/board.php?bo_table=s3_2&wr_id=2171">[(주)코스모웍스]</a></div>
        <div class="div_td col_date">26-10-21</div>
        <div class="div_td w12">2026.11.29.(일) 23:50까지</div>
      </div>
    </body></html>
    """
    gj_session = _FakeSession(
        lambda url, params: _FakeResponse(div_html if (params or {}).get("page", 1) == 1 else "<html></html>", url=url)
    )
    gj = GoodjobRecommendCollector(session=gj_session, delay_seconds=0)
    monkeypatch.setattr(gj, "check_robots", lambda url: True)
    gj_records = gj.fetch(limit=5)
    assert len(gj_records) == 1
    assert gj_records[0].title == "[(주)코스모웍스] 연구개발 신입 모집"
    assert gj_records[0].deadline == "2026-11-29"
    assert gj_records[0].posted_at == "2026-10-21"
    assert "국제" in gj_records[0].meta_tags
    assert gj_records[0].content == ""

    # 2. Seongnam sn-api JSON + 404 robots.txt check
    class _SnSession:
        def get(self, url: str, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse("Not Found", url=url, status_code=404)

        def post(self, url: str, json: dict[str, Any] | None = None, **kwargs: Any) -> _FakeResponse:
            if "notice/list" in url:
                return _FakeResponse(
                    '{"success":true,"data":[{"noticeId":49,"category":"GENERAL","title":"성남 청년 해커톤","content":"첫째 문장입니다. 둘째 문장입니다.","author":"정책기획과","registDt":"2026-10-06"}]}',
                    url=url,
                )
            if (json or {}).get("pageNo") == 1:
                return _FakeResponse(
                    '{"success":true,"data":{"response":[{"businessSeq":281,"title":"성남 자격증 지원","receptTypeCmmnClNm":"교육","startDate":"2026-11-02","endDate":"2026-11-26","target":"성남시 청년","businessDescription":"자격증 응시료를 지원합니다."}]}}',
                    url=url,
                )
            return _FakeResponse('{"success":true,"data":{"response":[]}}', url=url)

    sn = SeongnamYouthCollector(session=_SnSession(), delay_seconds=0)  # type: ignore[arg-type]
    assert sn.check_robots("https://dept.seongnam.go.kr/youth/contents.do?mId=0301000000") is True
    sn_records = sn.fetch(limit=10)
    assert len(sn_records) == 2
    assert sn_records[0].id == "seongnam_youth_notice_49"
    assert sn_records[1].id == "seongnam_youth_biz_281"
    assert sn_records[1].deadline == "2026-11-26"


def test_runner_collect_all_resilience_dedup_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _OkCollectorA(BaseCollector):
        source_name = "mock_a"

        def fetch(self, limit: int = 100) -> list[NoticeRecord]:
            return [
                NoticeRecord(
                    id="shared_1",
                    source="mock_a",
                    category="학사",
                    title="첫 번째 공지",
                    meta_tags="국제캠퍼스",
                    posted_at="2026-11-25",
                    url="https://example.com/1",
                    content="본문 A",
                    deadline="2026-12-01",
                ),
                NoticeRecord(
                    id="only_a",
                    source="mock_a",
                    category="장학",
                    title="두 번째 공지",
                    meta_tags="",
                    posted_at="2026-11-24",
                    url="https://example.com/2",
                ),
            ][:limit]

    class _FailingCollector(BaseCollector):
        source_name = "mock_fail"

        def fetch(self, limit: int = 100) -> list[NoticeRecord]:
            raise requests.ConnectionError("simulated network outage")

    class _OkCollectorB(BaseCollector):
        source_name = "mock_b"

        def fetch(self, limit: int = 100) -> list[NoticeRecord]:
            return [
                NoticeRecord(
                    id="shared_1",
                    source="mock_b",
                    category="중복",
                    title="중복 무시되어야 할 공지",
                    meta_tags="",
                    posted_at="2026-11-25",
                    url="https://example.com/dup",
                ),
                NoticeRecord(
                    id="only_b",
                    source="mock_b",
                    category="채용",
                    title="세 번째 공지",
                    meta_tags="IT/백엔드",
                    posted_at="2026-11-26",
                    url="https://example.com/3",
                ),
            ][:limit]

    custom_registry = {
        "mock_a": _OkCollectorA,
        "mock_fail": _FailingCollector,
        "mock_b": _OkCollectorB,
    }
    output_csv = tmp_path / "raw_notices.csv"

    df = collect_all(
        limit_per_source=10,
        output_path=output_csv,
        delay_seconds=0,
        registry=custom_registry,
    )

    assert list(df.columns) == CANONICAL_COLUMNS
    assert list(df["id"]) == ["shared_1", "only_a", "only_b"]
    assert df.loc[0, "source"] == "mock_a"
    validate_notices_df(df)

    saved = pd.read_csv(output_csv, encoding="utf-8", keep_default_na=False)
    assert list(saved.columns) == CANONICAL_COLUMNS
    assert len(saved) == 3

    # CLI entrypoint test without hitting live network
    cli_out = tmp_path / "cli_notices.csv"
    monkeypatch.setattr(
        "src.collectors.runner.COLLECTOR_REGISTRY",
        {"mock_a": _OkCollectorA},
    )
    exit_code = runner_main(["--output", str(cli_out), "--limit-per-source", "1", "--delay", "0"])
    assert exit_code == 0
    cli_df = pd.read_csv(cli_out, encoding="utf-8", keep_default_na=False)
    assert len(cli_df) == 1
    assert cli_df.loc[0, "id"] == "shared_1"
