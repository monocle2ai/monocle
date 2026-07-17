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
