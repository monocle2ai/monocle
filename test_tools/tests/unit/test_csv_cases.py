import os
import pytest
from monocle_test_tools import csv_cases


def _write(tmp_path, text):
    p = tmp_path / "cases.csv"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_read_rows_parses_header_and_strips(tmp_path):
    path = _write(tmp_path, "case_id, id ,workflow_name\n a1 ,tid1, wf1 \n")
    rows = csv_cases.read_rows(path)
    assert rows == [{"case_id": "a1", "id": "tid1", "workflow_name": "wf1"}]


def test_parse_multivalue_variants():
    assert csv_cases.parse_multivalue("") is None
    assert csv_cases.parse_multivalue("major") == "major"
    assert csv_cases.parse_multivalue("major|minor") == ["major", "minor"]
    assert csv_cases.parse_multivalue('["a","b"]') == ["a", "b"]


def test_parse_int_and_float():
    assert csv_cases.parse_int("", "token_limit", "c1") is None
    assert csv_cases.parse_int("5000", "token_limit", "c1") == 5000
    assert csv_cases.parse_float("12.5", "duration_ms", "c1") == 12.5
    with pytest.raises(csv_cases.CsvCaseError):
        csv_cases.parse_int("five", "token_limit", "c1")


def test_parse_extra_json():
    assert csv_cases.parse_extra_json("", "c1") == []
    steps = csv_cases.parse_extra_json(
        '[{"method":"has_attribute","kwargs":{"attribute_name":"model","expected":"gpt-4o"}}]', "c1"
    )
    assert steps == [{"method": "has_attribute", "kwargs": {"attribute_name": "model", "expected": "gpt-4o"}}]
    with pytest.raises(csv_cases.CsvCaseError):
        csv_cases.parse_extra_json('{"method":"x"}', "c1")   # not an array
    with pytest.raises(csv_cases.CsvCaseError):
        csv_cases.parse_extra_json('[{"kwargs":{}}]', "c1")  # missing method


def test_row_to_case_valid_eval_row():
    row = {"case_id": "cc_t01", "id": "642d", "workflow_name": "wf",
           "fact_name": "traces", "expected": "major|minor"}
    case = csv_cases._row_to_case(row, line=2)
    assert case.case_id == "cc_t01"
    assert case.id == "642d"
    assert case.workflow_name == "wf"
    assert case.expected == ["major", "minor"]
    assert case.fact_name == "traces"


def test_row_to_case_defaults_fact_name():
    row = {"case_id": "c", "id": "t", "workflow_name": "wf", "expected": "ok"}
    assert csv_cases._row_to_case(row, 2).fact_name == "traces"


@pytest.mark.parametrize("row,needle", [
    ({"case_id": "", "id": "t", "workflow_name": "wf", "expected": "ok"}, "case_id"),
    ({"case_id": "c", "id": "", "workflow_name": "wf", "expected": "ok"}, "id"),
    ({"case_id": "c", "id": "t", "workflow_name": "", "expected": "ok"}, "workflow_name"),
    ({"case_id": "c", "id": "t", "workflow_name": "wf", "fact_name": "bogus", "expected": "ok"}, "fact_name"),
    ({"case_id": "c", "id": "t", "workflow_name": "wf"}, "no assertions"),
    ({"case_id": "c", "id": "t", "workflow_name": "wf",
      "extra_json": '[{"method":"not_a_method"}]'}, "not_a_method"),
])
def test_row_to_case_errors(row, needle):
    with pytest.raises(csv_cases.CsvCaseError) as exc:
        csv_cases._row_to_case(row, line=7)
    assert needle in str(exc.value)


def test_row_to_case_non_eval_only_is_valid():
    row = {"case_id": "tool_01", "id": "t", "workflow_name": "wf", "called_tool": "search_web"}
    case = csv_cases._row_to_case(row, 2)
    assert case.called_tool == "search_web"
    assert case.expected is None


def test_load_cases_from_csv_happy(tmp_path):
    path = _write(tmp_path,
        "case_id,id,workflow_name,expected\n"
        "cc_t01,642d,wf_cc,major_hallucination\n"
        "cc_t02,8f12,wf_cc,no_hallucination\n")
    cases = csv_cases.load_cases_from_csv(path)
    assert [c.case_id for c in cases] == ["cc_t01", "cc_t02"]
    assert cases[0].expected == "major_hallucination"


def test_load_cases_duplicate_case_id(tmp_path):
    path = _write(tmp_path,
        "case_id,id,workflow_name,expected\n"
        "dup,t1,wf,ok\n"
        "dup,t2,wf,ok\n")
    with pytest.raises(csv_cases.CsvCaseError) as exc:
        csv_cases.load_cases_from_csv(path)
    assert "duplicate" in str(exc.value).lower()


def test_load_cases_missing_required_column(tmp_path):
    path = _write(tmp_path, "case_id,workflow_name,expected\nc,wf,ok\n")
    with pytest.raises(csv_cases.CsvCaseError) as exc:
        csv_cases.load_cases_from_csv(path)
    assert "id" in str(exc.value)


def test_load_cases_missing_file():
    with pytest.raises(csv_cases.CsvCaseError):
        csv_cases.load_cases_from_csv("/nonexistent/cases.csv")
