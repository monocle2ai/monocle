import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from opentelemetry.trace.status import StatusCode

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common import wrapper


@contextmanager
def _noop_workflow_type(*args, **kwargs):
    yield


def _span_context_manager(span):
    @contextmanager
    def _cm(*args, **kwargs):
        yield span

    return _cm()


class TestWorkflowSpanPaths(unittest.IsolatedAsyncioTestCase):
    def _base_to_wrap(self, auto_close=False):
        return {
            "package": "pkg",
            "object": "obj",
            "method": "method",
            "output_processor": {
                "is_auto_close": lambda kwargs: auto_close,
            },
        }

    def _common_patches(self):
        return patch.multiple(
            "monocle_apptrace.instrumentation.common.wrapper",
            pre_process_span=MagicMock(),
            post_process_span=MagicMock(side_effect=lambda *args, **kwargs: args[6]),
            get_current_monocle_span=MagicMock(return_value=wrapper.INVALID_SPAN),
        )

    def _span_handler_patches(self, remote_parent_side_effect):
        return patch.multiple(
            "monocle_apptrace.instrumentation.common.wrapper.SpanHandler",
            is_root_span=MagicMock(return_value=False),
            is_remote_parent_span=MagicMock(side_effect=remote_parent_side_effect),
            skip_execution=MagicMock(return_value=(False, None)),
            workflow_type=MagicMock(side_effect=_noop_workflow_type),
        )

    def test_sync_parent_workflow_span_sets_error_and_ends_on_child_failure(self):
        workflow_span = MagicMock(name="workflow_span")
        child_span = MagicMock(name="child_span")
        spans = iter([workflow_span, child_span])
        handler = MagicMock()

        def failing_wrapped(*args, **kwargs):
            raise RuntimeError("boom")

        with (
            self._common_patches(),
            self._span_handler_patches(remote_parent_side_effect=[False, False]),
            patch(
                "monocle_apptrace.instrumentation.common.wrapper.start_as_monocle_span",
                side_effect=lambda *args, **kwargs: _span_context_manager(next(spans)),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                wrapper.monocle_wrapper_span_processor(
                    tracer=MagicMock(),
                    handler=handler,
                    to_wrap=self._base_to_wrap(auto_close=False),
                    wrapped=failing_wrapped,
                    instance=MagicMock(),
                    source_path="/tmp/source.py",
                    add_workflow_span=True,
                    args=(),
                    kwargs={},
                )

        workflow_span.set_status.assert_called_once_with(StatusCode.ERROR, "boom")
        workflow_span.end.assert_called_once()

    async def test_async_parent_workflow_span_sets_error_and_ends_on_child_failure(self):
        workflow_span = MagicMock(name="workflow_span")
        child_span = MagicMock(name="child_span")
        spans = iter([workflow_span, child_span])
        handler = MagicMock()

        async def failing_wrapped(*args, **kwargs):
            raise RuntimeError("boom-async")

        with (
            self._common_patches(),
            self._span_handler_patches(remote_parent_side_effect=[False, False]),
            patch(
                "monocle_apptrace.instrumentation.common.wrapper.start_as_monocle_span",
                side_effect=lambda *args, **kwargs: _span_context_manager(next(spans)),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom-async"):
                await wrapper.amonocle_wrapper_span_processor(
                    tracer=MagicMock(),
                    handler=handler,
                    to_wrap=self._base_to_wrap(auto_close=False),
                    wrapped=failing_wrapped,
                    instance=MagicMock(),
                    source_path="/tmp/source.py",
                    add_workflow_span=True,
                    args=(),
                    kwargs={},
                )

        workflow_span.set_status.assert_called_once_with(StatusCode.ERROR, "boom-async")
        workflow_span.end.assert_called_once()

    async def test_async_iter_parent_workflow_span_sets_error_and_ends_on_child_failure(self):
        workflow_span = MagicMock(name="workflow_span")
        child_span = MagicMock(name="child_span")
        spans = iter([workflow_span, child_span])
        handler = MagicMock()

        async def failing_stream(*args, **kwargs):
            raise RuntimeError("boom-stream")
            yield  # pragma: no cover

        with (
            self._common_patches(),
            self._span_handler_patches(remote_parent_side_effect=[False, False]),
            patch(
                "monocle_apptrace.instrumentation.common.wrapper.start_as_monocle_span",
                side_effect=lambda *args, **kwargs: _span_context_manager(next(spans)),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom-stream"):
                async for _ in wrapper.amonocle_iter_wrapper_span_processor(
                    tracer=MagicMock(),
                    handler=handler,
                    to_wrap=self._base_to_wrap(auto_close=False),
                    wrapped=failing_stream,
                    instance=MagicMock(),
                    source_path="/tmp/source.py",
                    add_workflow_span=True,
                    args=(),
                    kwargs={},
                ):
                    pass

        workflow_span.set_status.assert_called_once_with(StatusCode.ERROR, "boom-stream")
        workflow_span.end.assert_called_once()

    def _stacked_to_wrap(self, intermediate_auto_close=False, leaf_auto_close=True):
        """to_wrap with two stacked processors; the intermediate (has_more) span is non-auto-close."""
        proc_a = {"is_auto_close": lambda kwargs: leaf_auto_close}
        proc_b = {"is_auto_close": lambda kwargs: leaf_auto_close}
        return {
            "package": "pkg",
            "object": "obj",
            "method": "method",
            "output_processor": {"is_auto_close": lambda kwargs: intermediate_auto_close},
            "output_processor_list": [proc_a, proc_b],
        }

    def test_sync_intermediate_nonautoclose_span_is_ended(self):
        intermediate_span = MagicMock(name="intermediate_span")
        leaf_span = MagicMock(name="leaf_span")
        spans = iter([intermediate_span, leaf_span])
        handler = MagicMock()
        handler.should_skip.return_value = False
        wrapped = MagicMock(return_value="ok")

        with (
            self._common_patches(),
            self._span_handler_patches(remote_parent_side_effect=[False] * 10),
            patch(
                "monocle_apptrace.instrumentation.common.wrapper.start_as_monocle_span",
                side_effect=lambda *args, **kwargs: _span_context_manager(next(spans)),
            ),
        ):
            result, _status = wrapper.monocle_wrapper_span_processor(
                tracer=MagicMock(),
                handler=handler,
                to_wrap=self._stacked_to_wrap(intermediate_auto_close=False),
                wrapped=wrapped,
                instance=MagicMock(),
                source_path="/tmp/source.py",
                add_workflow_span=False,
                args=(),
                kwargs={},
            )

        self.assertEqual(result, "ok")
        # Without the fix this intermediate span is never ended -> orphaned tree.
        intermediate_span.end.assert_called_once()

    async def test_async_intermediate_nonautoclose_span_is_ended(self):
        intermediate_span = MagicMock(name="intermediate_span")
        leaf_span = MagicMock(name="leaf_span")
        spans = iter([intermediate_span, leaf_span])
        handler = MagicMock()
        handler.should_skip.return_value = False

        async def wrapped(*args, **kwargs):
            return "ok-async"

        with (
            self._common_patches(),
            self._span_handler_patches(remote_parent_side_effect=[False] * 10),
            patch(
                "monocle_apptrace.instrumentation.common.wrapper.start_as_monocle_span",
                side_effect=lambda *args, **kwargs: _span_context_manager(next(spans)),
            ),
        ):
            result, _status = await wrapper.amonocle_wrapper_span_processor(
                tracer=MagicMock(),
                handler=handler,
                to_wrap=self._stacked_to_wrap(intermediate_auto_close=False),
                wrapped=wrapped,
                instance=MagicMock(),
                source_path="/tmp/source.py",
                add_workflow_span=False,
                args=(),
                kwargs={},
            )

        self.assertEqual(result, "ok-async")
        intermediate_span.end.assert_called_once()

    async def test_async_iter_intermediate_nonautoclose_span_is_ended(self):
        intermediate_span = MagicMock(name="intermediate_span")
        leaf_span = MagicMock(name="leaf_span")
        spans = iter([intermediate_span, leaf_span])
        handler = MagicMock()
        handler.should_skip.return_value = False

        async def streaming_wrapped(*args, **kwargs):
            yield "chunk-1"
            yield "chunk-2"

        with (
            self._common_patches(),
            self._span_handler_patches(remote_parent_side_effect=[False] * 10),
            patch(
                "monocle_apptrace.instrumentation.common.wrapper.start_as_monocle_span",
                side_effect=lambda *args, **kwargs: _span_context_manager(next(spans)),
            ),
        ):
            items = []
            async for item in wrapper.amonocle_iter_wrapper_span_processor(
                tracer=MagicMock(),
                handler=handler,
                to_wrap=self._stacked_to_wrap(intermediate_auto_close=False),
                wrapped=streaming_wrapped,
                instance=MagicMock(),
                source_path="/tmp/source.py",
                add_workflow_span=False,
                args=(),
                kwargs={},
            ):
                items.append(item)

        self.assertEqual(items, ["chunk-1", "chunk-2"])
        # The exact nested-subgraph-streaming case that was orphaned before the fix.
        intermediate_span.end.assert_called_once()

    def test_remote_parent_span_triggers_workflow_recursion_path(self):
        workflow_span = MagicMock(name="workflow_span")
        child_span = MagicMock(name="child_span")
        spans = iter([workflow_span, child_span])
        handler = MagicMock()
        wrapped = MagicMock(return_value="ok")

        with (
            self._common_patches(),
            self._span_handler_patches(remote_parent_side_effect=[True, False]),
            patch(
                "monocle_apptrace.instrumentation.common.wrapper.start_as_monocle_span",
                side_effect=lambda *args, **kwargs: _span_context_manager(next(spans)),
            ),
        ):
            result, _status = wrapper.monocle_wrapper_span_processor(
                tracer=MagicMock(),
                handler=handler,
                to_wrap=self._base_to_wrap(auto_close=True),
                wrapped=wrapped,
                instance=MagicMock(),
                source_path="/tmp/source.py",
                add_workflow_span=False,
                args=(),
                kwargs={},
            )

        self.assertEqual(result, "ok")
        wrapped.assert_called_once()
        workflow_span.set_status.assert_called_once_with(StatusCode.OK)

    def test_is_remote_parent_span_true_when_parent_is_remote(self):
        span = MagicMock()
        span.parent = MagicMock(is_remote=True)
        self.assertTrue(SpanHandler.is_remote_parent_span(span))

    def test_is_remote_parent_span_false_when_parent_missing_or_not_remote(self):
        no_parent_span = MagicMock()
        no_parent_span.parent = None

        local_parent_span = MagicMock()
        local_parent_span.parent = MagicMock(is_remote=False)

        self.assertFalse(SpanHandler.is_remote_parent_span(no_parent_span))
        self.assertFalse(SpanHandler.is_remote_parent_span(local_parent_span))
