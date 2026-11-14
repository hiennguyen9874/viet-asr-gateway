"""Admission contention must not prevent delivery of completed chunk results."""

import asyncio
import time
from contextlib import aclosing

import numpy as np
import pytest

from app.chunking import Span
from app.errors import APIError
from app.transcription import _until, run_chunks
from scripts.scoring import wer


def test_completed_chunk_is_delivered_while_replacement_admission_is_blocked():
    async def main():
        class Job:
            def __init__(self, text):
                self.future = asyncio.get_running_loop().create_future()
                self.future.set_result(text)

            def __await__(self):
                return self.future.__await__()

            def cancel(self):
                self.future.cancel()

        class Lane:
            def __init__(self):
                self.calls = 0
                self.blocked = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def submit(self, wav, deadline):
                self.calls += 1
                if self.calls == 3:
                    self.blocked.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.cancelled.set()
                return Job(str(self.calls))

        lane = Lane()
        spans = [Span(i * 1601, (i + 1) * 1601) for i in range(3)]
        async with aclosing(run_chunks(lane, np.zeros(4803, dtype=np.float32), spans,
                                      time.monotonic() + 5, 2)) as texts:
            assert await anext(texts) == (spans[0], "1")
            # The second result is available even though replenishing the window is blocked.
            assert await asyncio.wait_for(anext(texts), 0.2) == (spans[1], "2")
            await asyncio.wait_for(lane.blocked.wait(), 0.2)
        assert lane.cancelled.is_set()

    asyncio.run(main())


def test_expired_deadline_rejects_an_already_completed_result():
    async def main():
        result = asyncio.get_running_loop().create_future()
        result.set_result("too late")
        with pytest.raises(APIError) as error:
            await _until(time.monotonic() - 1, result)
        assert error.value.code == "request_timeout"

    asyncio.run(main())


def test_shared_word_error_rate():
    assert wer("one two three", "one four three five") == pytest.approx(2 / 3)
    assert wer("", "") == 0
    assert wer("one", "") == 1


def test_word_error_rate_ignores_case_and_punctuation_on_both_sides():
    assert wer("Ngân sách, dưới 50 triệu.", "ngân sách dưới 50 triệu") == 0
    assert wer("ngân sách dưới 50 triệu", "Ngân sách, dưới 50 triệu.") == 0
    assert wer("“Dạ” – vâng…", "dạ vâng") == 0  # Unicode punctuation, not only ASCII
    assert wer("năm mươi", "50") == 1  # digits vs spelled-out numbers stay a mismatch
