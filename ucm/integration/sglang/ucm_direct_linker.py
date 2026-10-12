"""SGLang L1 external-linker placeholder for the UCM backend.

The versioned registry patch can select this class without modifying SGLang's
built-in Mooncake or Mori paths. Real UCM external storage is not implemented.
An explicit no-op mode exists solely to exercise the patch and server startup
path: it reports no external hits and does not persist cache data.
"""

from __future__ import annotations

import os

from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker

_STORE_UNIMPLEMENTED = (
    "UCM SGLang L1 external storage is not implemented yet. "
    "Select --unified-cache-external-linker-backend mooncake or mori, or set "
    "UCM_SGLANG_ALLOW_NOOP_LINKER=1 for the startup-only no-op linker."
)


class _NoopLayerDoneCounter:
    """Counter protocol implementation for a linker that never queues a load."""

    def set_consumer(self, index: int) -> None:
        return None

    def wait_until(self, threshold: int) -> None:
        return None


class UcmDirectLinker(UnifiedCacheLinker):
    """Reserved linker implementation until ``UcmDirectStoreAdapter`` is ready."""

    def __init__(self, server_args, params, *, components):
        if os.environ.get("UCM_SGLANG_ALLOW_NOOP_LINKER", "").lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            raise NotImplementedError(_STORE_UNIMPLEMENTED)
        self.layer_done_counter = _NoopLayerDoneCounter()

    def lookup(self, rid, transfers):
        return []

    def load(self, rid, transfers):
        return False

    def start_layer_wise_loading(self):
        return -1

    def cancel_queued_load(self, rid):
        return False

    def num_completed_loads(self):
        return 0

    def pop_completed_load(self):
        raise RuntimeError("UCM no-op linker has no completed loads.")

    def offload(self, transfers):
        return False

    def num_completed_offloads(self):
        return 0

    def pop_completed_offload(self):
        raise RuntimeError("UCM no-op linker has no completed offloads.")

    def reset(self):
        return None

    def close(self):
        return None
