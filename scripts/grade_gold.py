"""Fast interactive CLI for Human Gold grading (150-item Test Set & 20-item Summary Eval)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.summarizer import TextRankSummarizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _clean(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _save_records(records: list[dict[str, Any]], out_path: Path, columns: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=columns).to_csv(out_path, index=False, encoding="utf-8")


def grade_test_set(
    blind_csv: Path = Path("data/splits/test_blind.csv"),
    gold_csv: Path = Path("data/splits/test_gold.csv"),
) -> None:
    """Interactively label the blind Hold-out Test Set (1=관심, 0=제외) with resume support."""
    df = pd.read_csv(blind_csv, encoding="utf-8")
    columns = ["id", "human_label", "human_note"]

    saved_map: dict[str, dict[str, Any]] = {}
    if gold_csv.is_file() and gold_csv.stat().st_size > 0:
        existing = pd.read_csv(gold_csv, encoding="utf-8")
        for row in existing.to_dict(orient="records"):
            nid = _clean(row.get("id"))
            lbl = row.get("human_label")
            if nid and lbl in (0, 1, "0", "1"):
                saved_map[nid] = {
                    "id": nid,
                    "human_label": int(lbl),
                    "human_note": _clean(row.get("human_note")),
                }

    rows = df.to_dict(orient="records")
    total = len(rows)

    def _flush() -> None:
        ordered = [saved_map[_clean(r["id"])] for r in rows if _clean(r["id"]) in saved_map]
        _save_records(ordered, gold_csv, columns)

    idx = 0
    while idx < total and _clean(rows[idx]["id"]) in saved_map:
        idx += 1

    print(f"=== Human Gold 채점 ({len(saved_map)}/{total} 완료, 저장 경로: {gold_csv}) ===")
    print("명령어: [1] 관심  [0] 제외  [v] 전체 본문  [b] 이전 수정  [q] 저장 후 종료\n")

    while idx < total:
        row = rows[idx]
        nid = _clean(row.get("id"))
        content = _clean(row.get("content"))
        snippet = content[:200] + ("..." if len(content) > 200 else "")
        print("-" * 72)
        print(f"[{idx + 1}/{total}] ({_clean(row.get('source'))}) [{_clean(row.get('category'))}]")
        print(f"제목: {_clean(row.get('title'))}")
        print(f"태그: {_clean(row.get('meta_tags')) or '-'} | 게시: {_clean(row.get('posted_at')) or '-'} | 마감: {_clean(row.get('deadline')) or '-'}")
        if snippet:
            print(f"본문: {snippet}")

        ans = input("판정 (1=관심 / 0=제외 / v=전체 / b=이전 / q=종료): ").strip().lower()
        if ans == "q":
            _flush()
            print(f"저장 완료 ({len(saved_map)}/{total}).")
            return
        if ans == "v":
            print(f"\n[전체 본문]\n{content or '(본문 없음)'}\nURL: {_clean(row.get('url'))}\n")
            continue
        if ans == "b":
            idx = max(0, idx - 1)
            continue
        if ans in {"0", "1"}:
            saved_map[nid] = {"id": nid, "human_label": int(ans), "human_note": ""}
            _flush()
            idx += 1
        else:
            print("1, 0, v, b, q 중 하나를 입력하세요.")

    _flush()
    print(f"\n모든 {total}건 Hold-out Test Human Gold 채점이 완료되었습니다!")


def grade_summary_set(
    eval_csv: Path = Path("data/splits/summary_eval_20.csv"),
    gold_csv: Path = Path("data/splits/summary_eval_gold.csv"),
) -> None:
    """Interactively compose 2-3 sentence Human reference summaries for ROUGE evaluation."""
    df = pd.read_csv(eval_csv, encoding="utf-8")
    columns = ["id", "human_summary"]
    summarizer = TextRankSummarizer()

    saved_map: dict[str, dict[str, Any]] = {}
    if gold_csv.is_file() and gold_csv.stat().st_size > 0:
        existing = pd.read_csv(gold_csv, encoding="utf-8")
        for row in existing.to_dict(orient="records"):
            nid = _clean(row.get("id"))
            summary = _clean(row.get("human_summary"))
            if nid and summary:
                saved_map[nid] = {"id": nid, "human_summary": summary}

    rows = df.to_dict(orient="records")
    total = len(rows)

    def _flush() -> None:
        ordered = [saved_map[_clean(r["id"])] for r in rows if _clean(r["id"]) in saved_map]
        _save_records(ordered, gold_csv, columns)

    idx = 0
    while idx < total and _clean(rows[idx]["id"]) in saved_map:
        idx += 1

    print(f"=== 요약 정답지 작성 ({len(saved_map)}/{total} 완료, 저장 경로: {gold_csv}) ===")
    print("입력 방법: 번호 조합 (예: 1,2,4) 또는 직접 요약 문장 입력 | [b] 이전 | [q] 저장 후 종료\n")

    while idx < total:
        row = rows[idx]
        nid = _clean(row.get("id"))
        sentences = summarizer.split_sentences(_clean(row.get("content")))
        print("-" * 72)
        print(f"[{idx + 1}/{total}] ({_clean(row.get('source'))}) {_clean(row.get('title'))}")
        for i, sent in enumerate(sentences, start=1):
            print(f"  [{i}] {sent}")

        ans = input("핵심 문장 번호(예: 1,2,3) 또는 요약문 직접 입력 (b/q): ").strip()
        if ans.lower() == "q":
            _flush()
            print(f"저장 완료 ({len(saved_map)}/{total}).")
            return
        if ans.lower() == "b":
            idx = max(0, idx - 1)
            continue
        if not ans:
            print("문장 번호 또는 요약 텍스트를 입력하세요.")
            continue

        parts = [p.strip() for p in ans.split(",")]
        if all(p.isdigit() for p in parts):
            indices = [int(p) - 1 for p in parts if 1 <= int(p) <= len(sentences)]
            if not indices:
                print(f"1~{len(sentences)} 범위의 문장 번호를 입력하세요.")
                continue
            summary = " ".join(sentences[i] for i in indices)
        else:
            summary = ans

        saved_map[nid] = {"id": nid, "human_summary": summary}
        _flush()
        idx += 1

    _flush()
    print(f"\n모든 {total}건 요약 정답지 작성이 완료되었습니다!")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Human Gold grading tool.")
    parser.add_argument("--mode", choices=["test", "summary"], default="test")
    args = parser.parse_args()
    if args.mode == "test":
        grade_test_set()
    else:
        grade_summary_set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
