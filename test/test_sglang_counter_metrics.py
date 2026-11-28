"""Check that the compatibility counter waits for layer completion."""

import threading
from concurrent.futures import Future

from common.sglang_utils import load_source

CompatibilityCounter = load_source(
    "ucm/integration/sglang/patch/v0520/layerwise_load_counter.py",
    Future=Future,
).LayerWiseLoadCounter


def test_actual_wait_blocks_until_complete():
    counter = CompatibilityCounter(2)
    index = counter.update_producer()
    counter.set_consumer(index)
    finished = threading.Event()

    def consume():
        counter.wait_until(0)
        finished.set()

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        assert not finished.wait(0.02)
        counter.complete(index, 0)
        assert finished.wait(2)
    finally:
        counter.complete(index, 0)
        thread.join(2)
    assert not thread.is_alive()
