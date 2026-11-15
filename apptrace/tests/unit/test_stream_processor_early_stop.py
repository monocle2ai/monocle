"""
Offline unit tests for when BaseStreamProcessor emits the stream span.

Each test checks that exactly one span is emitted however the consumer ends
the stream: drained, closed, `with` exit, or a plain `break`.
"""
import asyncio
import gc

import pytest

from monocle_apptrace.instrumentation.common.stream_processor import BaseStreamProcessor


class _Processor(BaseStreamProcessor):
    def handle_event(self, item, state):
        return False

    def handle_chunk(self, item, state):
        state.add_content(item)
        return True

    def handle_completion(self, item, state):
        return False


class _SyncStream:
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


class _AsyncStream:
    def __init__(self, items):
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


ITEMS = ["a", "b", "c", "d"]


def _wrap(response):
    """Run response through process_stream and return (stream, emitted spans)."""
    emitted = []
    wrapper = _Processor().process_stream(True, response, emitted.append)
    return (wrapper or response), emitted


class TestSyncEarlyStop:
    def test_drained_then_closed_emits_once(self):
        stream, emitted = _wrap(_SyncStream(ITEMS))
        with stream:
            assert list(stream) == ITEMS
        assert len(emitted) == 1
        assert emitted[0].output_text == "abcd"

    def test_break_emits_right_away(self):
        stream, emitted = _wrap(_SyncStream(ITEMS))
        for item in stream:
            if item == "b":
                break
        assert len(emitted) == 1
        assert emitted[0].output_text == "ab"

    def test_with_exit_emits_once(self):
        stream, emitted = _wrap(_SyncStream(ITEMS))
        with stream:
            it = iter(stream)
            next(it)
            assert emitted == []
        assert len(emitted) == 1

    def test_native_generator_dropped_emits(self):
        stream, emitted = _wrap(item for item in ITEMS)
        for item in stream:
            if item == "b":
                break
        del stream
        gc.collect()
        assert len(emitted) == 1


class TestAsyncEarlyStop:
    @pytest.mark.asyncio
    async def test_break_emits_after_loop_ticks(self):
        stream, emitted = _wrap(_AsyncStream(ITEMS))
        async for item in stream:
            if item == "b":
                break
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(emitted) == 1
        assert emitted[0].output_text == "ab"

    @pytest.mark.asyncio
    async def test_async_with_exit_emits_once(self):
        stream, emitted = _wrap(_AsyncStream(ITEMS))
        async with stream:
            async for item in stream:
                if item == "b":
                    break
        assert len(emitted) == 1
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(emitted) == 1

    @pytest.mark.asyncio
    async def test_native_async_generator_dropped_emits(self):
        async def gen():
            for item in ITEMS:
                yield item

        stream, emitted = _wrap(gen())
        async for item in stream:
            if item == "b":
                break
        del stream
        gc.collect()
        assert len(emitted) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
