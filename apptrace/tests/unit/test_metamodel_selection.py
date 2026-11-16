import os
import unittest
from unittest.mock import patch

import requests
from openai.resources.chat.completions.completions import Completions
from wrapt import ObjectProxy

from monocle_apptrace.instrumentation.common.constants import METAMODELS_ENV
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, get_monocle_instrumentor
from monocle_apptrace.instrumentation.common.utils import build_setup_signature
from monocle_apptrace.instrumentation.common.wrapper_method import (
    DEFAULT_METHODS_LIST,
    METAMODEL_METHODS,
    get_metamodel_methods,
    validate_metamodels,
)
from monocle_apptrace import get_supported_metamodels
from monocle_apptrace.instrumentation.metamodel.openai.methods import OPENAI_METHODS
from monocle_apptrace.instrumentation.metamodel.requests.methods import REQUESTS_METHODS

WRAPPER_METHOD_LOGGER = "monocle_apptrace.instrumentation.common.wrapper_method"


class TestGetMetamodelMethods(unittest.TestCase):

    def test_none_returns_all(self):
        self.assertEqual(get_metamodel_methods(None), DEFAULT_METHODS_LIST)
        self.assertEqual(len(DEFAULT_METHODS_LIST), sum(len(m) for m in METAMODEL_METHODS.values()))

    def test_selection_returns_only_those(self):
        self.assertEqual(get_metamodel_methods(["requests", "openai"]), REQUESTS_METHODS + OPENAI_METHODS)

    def test_string_is_split_on_commas(self):
        self.assertEqual(get_metamodel_methods("requests"), REQUESTS_METHODS)
        self.assertEqual(get_metamodel_methods(" requests, openai "), REQUESTS_METHODS + OPENAI_METHODS)

    def test_supported_names_match_methods(self):
        self.assertEqual(get_supported_metamodels(), sorted(METAMODEL_METHODS))
        self.assertIn("langchain", get_supported_metamodels())

    def test_duplicate_names_load_once(self):
        self.assertEqual(get_metamodel_methods(["requests", "requests"]), REQUESTS_METHODS)

    def test_unknown_name_warns_and_is_skipped(self):
        with self.assertLogs(WRAPPER_METHOD_LOGGER, level="WARNING") as logs:
            self.assertEqual(validate_metamodels(["langchian", "requests"]), ["requests"])
        self.assertIn("Did you mean 'langchain'?", logs.output[0])

    def test_only_unknown_names_fall_back_to_all(self):
        with self.assertLogs(WRAPPER_METHOD_LOGGER, level="WARNING"):
            self.assertIsNone(validate_metamodels("not_a_metamodel"))

    def test_single_string_name(self):
        self.assertEqual(validate_metamodels("openai"), ["openai"])

    def test_signature_tracks_selection(self):
        all_models = build_setup_signature("wf")
        some = build_setup_signature("wf", metamodels=["openai", "requests"])
        same = build_setup_signature("wf", metamodels=["requests", "openai"])
        self.assertNotEqual(all_models, some)
        self.assertEqual(some, same)


class TestSetupWithMetamodels(unittest.TestCase):

    def setUp(self):
        existing = get_monocle_instrumentor()
        if existing is not None:
            try:
                existing.uninstrument()
            except Exception:
                pass
        self.instrumentor = None

    def tearDown(self):
        if self.instrumentor is not None:
            try:
                self.instrumentor.uninstrument()
            except Exception as e:
                print("Uninstrument failed:", e)

    def _is_wrapped(self, func) -> bool:
        # openai's own decorators set __wrapped__, so check for wrapt's proxy instead.
        return isinstance(func, ObjectProxy)

    def test_only_selected_metamodel_is_patched(self):
        self.instrumentor = setup_monocle_telemetry("metamodel_selection_wf", metamodels=["requests"],
                                                    monocle_exporters_list="memory")
        self.assertTrue(self._is_wrapped(requests.Session.request))
        self.assertFalse(self._is_wrapped(Completions.create))
        self.assertEqual(self.instrumentor.metamodels, ["requests"])

    def test_env_var_selects_metamodels(self):
        with patch.dict(os.environ, {METAMODELS_ENV: "requests, openai"}):
            self.instrumentor = setup_monocle_telemetry("metamodel_env_wf", monocle_exporters_list="memory")
        self.assertEqual(self.instrumentor.metamodels, ["requests", "openai"])

    def test_string_argument_is_parsed(self):
        self.instrumentor = setup_monocle_telemetry("metamodel_str_wf", metamodels="requests,openai",
                                                    monocle_exporters_list="memory")
        self.assertEqual(self.instrumentor.metamodels, ["requests", "openai"])

    def test_typo_is_dropped_at_setup(self):
        with self.assertLogs(WRAPPER_METHOD_LOGGER, level="WARNING"):
            self.instrumentor = setup_monocle_telemetry("metamodel_typo_wf", metamodels=["requests", "opneai"],
                                                        monocle_exporters_list="memory")
        self.assertEqual(self.instrumentor.metamodels, ["requests"])


if __name__ == '__main__':
    unittest.main()
