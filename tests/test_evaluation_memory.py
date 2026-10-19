"""Offline tests for independent reflection memory behavior."""

from datetime import date
from pathlib import Path

import pytest

from src.agents.schemas import Draft, Reflection
from src.memory.evaluation_memory import (
    EvaluationMemory,
    normalize_lesson,
    note_priority,
    reflect_and_remember,
    select_lessons,
    sort_date,
    verify_lesson,
)


def reflection(*lessons):
    return Reflection(
        weakest_section="Fundamentals",
        unsupported_claims=[],
        missing_data=["Peer valuation"],
        failed_tools=[],
        what_to_change=["Verify report calculations"],
        confidence="medium",
        lessons=list(lessons),
    )


def draft():
    return Draft(
        ticker="NVDA", thesis="Research thesis",
        bull_case=["Demand"], bear_case=["Competition"],
        fundamentals="Fundamentals", news_and_catalysts="News",
        market_and_macro="Macro", conclusion="Conclusion",
    )


def note(ticker, sector, when, lesson):
    return {"ticker": ticker, "sector": sector,
            "date": when, "lesson": lesson}


def test_normalize_lesson():
    assert (
        normalize_lesson("  For banks,\n  check NIM. ")
        == "For banks, check NIM."
    )


@pytest.mark.parametrize("content,reason", [
    ("", "empty"),
    ("   ", "empty"),
    ("short", "too_short"),
    ("x" * 241, "too_long"),
    ("Use get_options_chain during earnings review", "unknown_explicit_tool"),
    ("Call search_private_data for company research", "unknown_explicit_tool"),
])
def test_reject_invalid_lessons(content, reason):
    assert verify_lesson(content) == reason


def test_permit_known_tool_reference():
    assert (
        verify_lesson("Call get_financials for cash flow statements") is None
    )


def test_permit_plain_language_lesson_without_tool():
    assert verify_lesson("For banks, examine net interest margin") is None


def test_sort_date_invalid_oldest():
    assert sort_date({"date": "not a date"}) == date.min


def test_prioritize_ticker_over_more_recent_sector():
    notes = [
        note("*", "", "2026-10-16", "Read news headlines."),
        note("JPM", "Technology", "2026-10-16", "Inspect sector growth."),
        note("NVDA", "Technology", "2026-10-01", "Review GPU demand."),
    ]
    selected = select_lessons(notes, "NVDA", "Technology")
    assert [r["ticker"] for r in selected] == ["NVDA", "JPM", "*"]


def test_recent_first_within_relevance_category():
    notes = [
        note("NVDA", "Technology", "2026-10-01", "Older lesson for NVDA"),
        note("NVDA", "Technology", "2026-10-15", "Newer lesson for NVDA"),
        note("NVDA", "Technology", "2026-10-10", "Middle lesson NVDA"),
    ]
    selected = select_lessons(notes, "NVDA", "Technology", limit=2)
    assert [r["date"] for r in selected] == ["2026-10-15", "2026-10-10"]


def test_missing_sector_allows_ticker_and_general_only():
    notes = [
        note("JPM", "Banks", "2026-10-15", "Look for net interest margin"),
        note(
            "NVDA", "Technology", "2026-10-02", "Check key customers closely"
        ),
        note("*", "", "2026-10-15", "Check citations for every claim"),
    ]
    selected = select_lessons(notes, "NVDA", None)
    assert [n["ticker"] for n in selected] == ["NVDA", "*"]


def test_ignores_irrelevant_ticker_and_sector():
    notes = [note("XOM", "Energy", "2026-10-15", "Inspect oil prices")]
    assert select_lessons(notes, "NVDA", "Technology") == []


def test_zero_limit_selects_nothing():
    assert select_lessons([note("NVDA", "T", "2026-10-01", "Older lesson")],
                          "NVDA", "T", limit=0) == []


def test_negative_limit_is_invalid():
    with pytest.raises(ValueError, match="negative"):
        select_lessons([], "NVDA", None, limit=-1)


def test_case_insensitive_ticker_sector():
    assert note_priority(note("nvda", "Technology", "2026-10-01", "Whatever"),
                         "NVDA", "technology") == 0


def test_unavailable_file_returns_empty(tmp_path):
    memory = EvaluationMemory(tmp_path / "memory.json")
    assert memory.load() == []
    assert memory.relevant("NVDA", None) == []


def test_preview_does_not_write_file(tmp_path):
    path = tmp_path / "memory.json"
    result = EvaluationMemory(path).add_reflection(
        "NVDA", "Technology",
        reflection("Verify quarterly earnings before reporting growth"),
        today=date(2026, 10, 16),
    )
    assert len(result.added_notes) == 1
    assert not result.saved
    assert not path.exists()


def test_persist_writes_expected_record(tmp_path):
    memory = EvaluationMemory(tmp_path / "memory.json")
    result = memory.add_reflection(
        "nvda", "Technology",
        reflection("Verify quarterly earnings before reporting growth"),
        today=date(2026, 10, 16), persist=True,
    )
    assert result.saved
    assert memory.load()[0]["ticker"] == "NVDA"
    assert memory.load()[0]["date"] == "2026-10-16"


def test_duplicate_detection_ignores_case_and_whitespace(tmp_path):
    memory = EvaluationMemory(tmp_path / "notes.json")
    memory.add_reflection("NVDA", "Tech",
        reflection("Check financial statements for changes"), persist=True)
    result = memory.add_reflection("NVDA", "Tech",
        reflection(" CHECK   financial statements for changes "),
        persist=True)
    assert not result.saved
    assert result.rejected[0].reason == "duplicate"
    assert len(memory.load()) == 1


def test_duplicate_inside_one_reflection(tmp_path):
    memory = EvaluationMemory(tmp_path / "notes.json")
    result = memory.add_reflection("NVDA", None,
        reflection("Check peer valuations before publishing",
                   " check peer valuations before publishing  "))
    assert len(result.added_notes) == 1
    assert result.rejected[0].reason == "duplicate"


def test_unknown_tool_name_is_rejected_in_reflection(tmp_path):
    memory = EvaluationMemory(tmp_path / "notes.json")
    result = memory.add_reflection("NVDA", "Tech",
        reflection("Call get_unavailable_peers before planning next run",
                   "Call get_financials to verify income statements"))
    assert len(result.added_notes) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0].reason == "unknown_explicit_tool"


def test_max_records_keeps_latest_written(tmp_path):
    memory = EvaluationMemory(tmp_path / "notes.json", max_records=2)
    first = reflection("Check filing A before planning next run",
                       "Check filing B before planning next run")
    memory.add_reflection("NVDA", None, first, persist=True)
    memory.add_reflection("NVDA", None,
        reflection("Check filing C before planning next run"), persist=True)
    assert [n["lesson"] for n in memory.load()] == [
        "Check filing B before planning next run",
        "Check filing C before planning next run",
    ]


def test_malformed_json_errors_instead_of_overwriting(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("broken {", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        EvaluationMemory(path).add_reflection("NVDA", None,
            reflection("Inspect annual reports before investing"),
            persist=True,
        )
    assert path.read_text() == "broken {"


def test_wrong_json_shape_fails_closed(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text('{"lesson":"unexpected"}', encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        EvaluationMemory(path).load()


def test_persist_false_does_not_modify_existing_memory(tmp_path):
    memory = EvaluationMemory(tmp_path / "notes.json")
    memory.add_reflection("NVDA", None,
        reflection("Compare customer concentration before reports"),
        persist=True,
    )
    previous = memory.path.read_bytes()
    memory.add_reflection("JPM", None,
        reflection("Check bank liquidity before investing"), persist=False)
    assert memory.path.read_bytes() == previous


def test_reflect_and_remember_uses_injected_callback(tmp_path):
    calls = []

    def fake_reflect(current, findings, tools):
        calls.append((current.ticker, len(findings), len(tools)))
        return reflection("Confirm tool provenance before publishing")

    result = reflect_and_remember(
        draft(), [], [{"ok": False}], ticker="NVDA", sector="Technology",
        memory=EvaluationMemory(tmp_path / "notes.json"),
        reflect_fn=fake_reflect, today=date(2026, 10, 16),
    )
    assert calls == [("NVDA", 0, 1)]
    assert len(result.added_notes) == 1
    assert not result.saved


def test_reflection_rejects_ticker_mismatch(tmp_path):
    with pytest.raises(ValueError, match="ticker"):
        reflect_and_remember(
            draft(), [], [], ticker="JPM", sector=None,
            memory=EvaluationMemory(tmp_path / "notes.json"),
            reflect_fn=lambda *args: reflection(
                "Validate SEC risk references"
            ),
        )


def test_invalid_memory_capacity():
    with pytest.raises(ValueError, match="positive"):
        EvaluationMemory(Path("unused"), max_records=0)
