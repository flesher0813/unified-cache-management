"""SGLang L1 external-linker placeholder for the UCM backend.

The versioned registry patch can select this class without modifying SGLang's
built-in Mooncake or Mori paths. Its storage adapter is deliberately not
implemented yet, so constructing this linker fails before it can attach to a
cache or start transfer workers.
"""

from __future__ import annotations

from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker

_STORE_UNIMPLEMENTED = (
    "UCM SGLang L1 external storage is not implemented yet. "
    "Select --unified-cache-external-linker-backend mooncake or mori."
)


class UcmDirectLinker(UnifiedCacheLinker):
    """Reserved linker implementation until ``UcmDirectStoreAdapter`` is ready."""

    def __init__(self, server_args, params, *, components):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def lookup(self, rid, transfers):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def load(self, rid, transfers):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def start_layer_wise_loading(self):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def cancel_queued_load(self, rid):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def num_completed_loads(self):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def pop_completed_load(self):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def offload(self, transfers):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def num_completed_offloads(self):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def pop_completed_offload(self):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def reset(self):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)

    def close(self):
        raise NotImplementedError(_STORE_UNIMPLEMENTED)
