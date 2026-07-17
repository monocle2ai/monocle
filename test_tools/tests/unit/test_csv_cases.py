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
