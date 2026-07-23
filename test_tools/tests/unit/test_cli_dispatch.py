"""Tests for CLI subcommand dispatch in __main__.py and generate_test.py."""

import sys
from unittest.mock import patch
import pytest


def test_dispatch_help_exits_0():
    """--help should print usage and exit with code 0."""
    from monocle_test_tools.__main__ import main
    with patch.object(sys, "argv", ["monocle_test_tools", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_dispatch_no_args_exits_1():
    """No args should print usage and exit with code 1."""
    from monocle_test_tools.__main__ import main
    with patch.object(sys, "argv", ["monocle_test_tools"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_dispatch_unknown_command_exits_1():
    """Unknown command should print error and exit with code 1."""
    from monocle_test_tools.__main__ import main
    with patch.object(sys, "argv", ["monocle_test_tools", "bogus"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_dispatch_generate_test_without_trace_file_exits_2():
    """generate_test without --trace-file should exit with code 2."""
    from monocle_test_tools.__main__ import main
    with patch.object(sys, "argv", ["monocle_test_tools", "generate_test"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_generate_test_accepts_positional_trace_file():
    """generate_test should accept trace_file as positional argument."""
    from monocle_test_tools.generate_test import main
    with patch.object(sys, "argv", ["generate_test", "trace.json"]):
        result = main()
    assert result != 2  # Not an arg-parse error


def test_generate_test_accepts_trace_file_flag():
    """generate_test should accept --trace-file as named argument."""
    from monocle_test_tools.generate_test import main
    with patch.object(sys, "argv", ["generate_test", "--trace-file", "trace.json"]):
        result = main()
    assert result != 2  # Not an arg-parse error


def test_generate_test_accepts_okahu_trace_id_and_workflow_name():
    """generate_test should accept direct Okahu inputs."""
    from monocle_test_tools.generate_test import main
    with patch.object(
        sys,
        "argv",
        [
            "generate_test",
            "--trace-id",
            "abc123",
            "--workflow-name",
            "my_app",
        ],
    ):
        result = main()
    assert result != 2  # Not an arg-parse error


def test_generate_test_with_only_trace_id_exits_2():
    """trace-id without workflow-name should fail arg validation."""
    from monocle_test_tools.generate_test import main
    with patch.object(sys, "argv", ["generate_test", "--trace-id", "abc123"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_generate_test_rejects_mixed_file_and_okahu_inputs():
    """Supplying both file and Okahu inputs should fail arg validation."""
    from monocle_test_tools.generate_test import main
    with patch.object(
        sys,
        "argv",
        [
            "generate_test",
            "trace.json",
            "--trace-id",
            "abc123",
            "--workflow-name",
            "my_app",
        ],
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2
