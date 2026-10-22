"""UCM implementation of SGLang's direct L1 external-cache linker.

This module owns the SGLang side only: it materializes device-pool views,
converts ``PoolTransfer`` objects to neutral UCM records, and drives SGLang's
layer-ready and completion queues. Store setup and I/O stay in the adapter.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import get_model, get_parallel
from sglang.srt.utils import freeze_gc, get_device_module

from ucm.integration.sglang.ucm_direct_store_adapter import (
    UcmDirectStoreAdapter,
    UcmLayerTransfer,
    UcmPoolMetadata,
)

logger = logging.getLogger(__name__)
device_module = get_device_module()
_MIN_TIMING_LOG_MS = 10.0


class LayerWiseLoadCounter:
    """Completion counter consumed by SGLang's layer-transfer hooks."""

    def __init__(self, num_layers: int, tp_rank: int):
        self.num_layers = num_layers
        self.tp_rank = tp_rank
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        self._lock = threading.Lock()

    def update_producer(self) -> int:
        with self._lock:
            self.producer_index += 1
            index = self.producer_index
            self.futures[index] = [Future() for _ in range(self.num_layers)]
            return index

    def set_consumer(self, index: int) -> None:
        with self._lock:
            self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        with self._lock:
            futures = self.futures.get(index)
            if futures is None:
                raise RuntimeError(f"UCM completion references unknown batch {index}.")
            if not 0 <= layer < len(futures):
                raise IndexError(f"UCM completion has invalid layer {layer}.")
            future = futures[layer]
        if not future.done():
            future.set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        with self._lock:
            futures = tuple(self.futures.get(index, ()))
        for future in futures:
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        with self._lock:
            index = self.consumer_index
            futures = self.futures.get(index)
        if futures is None:
            return
        if not 0 <= threshold < len(futures):
            raise IndexError(
                f"UCM wait has invalid layer {threshold} for batch {index}."
            )
        future = futures[threshold]
        pending_before_wait = not future.done()
        wait_start = time.perf_counter()
        try:
            future.result()
        except BaseException as error:
            raise RuntimeError("UCM layer-wise KV load failed.") from error
        finally:
            wait_ms = (time.perf_counter() - wait_start) * 1000
            if wait_ms >= _MIN_TIMING_LOG_MS:
                logger.info(
                    "UCM linker wait_until tp_rank=%d batch=%d layer=%d "
                    "pending_before_wait=%s wait_ms=%.3f",
                    self.tp_rank,
                    index,
                    threshold,
                    pending_before_wait,
                    wait_ms,
                )
            if threshold == self.num_layers - 1:
                with self._lock:
                    self.futures.pop(index, None)

    def reset(self) -> None:
        with self._lock:
            self.producer_index = -1
            self.consumer_index = -1
            self.futures.clear()


@dataclass(frozen=True)
class _PoolBundle:
    """Physical pools persisted together as one UCM block payload.

    A bundle is deliberately stricter than ``indices_from_pool`` alone: its
    entries must also cover the same logical-layer set.  DSA's KV and INDEXER
    entries satisfy this, so one page/layer becomes one composite Store block.
    """

    name: str
    source_pool: PoolName
    layers: tuple[int, ...]
    entries: tuple[DevicePoolEntry, ...]


@dataclass(frozen=True)
class _LayerPointerLayout:
    """Fixed buffer addresses, strides and fragment schema for one pool layer."""

    bases: np.ndarray
    strides: np.ndarray
    schema: tuple[tuple[int, int], ...]


@dataclass
class _TransferBuildTimings:
    schema_ms: float = 0.0
    locations_ms: float = 0.0
    layer_meta_ms: float = 0.0
    # Kept in the log for before/after comparison; array I/O no longer
    # converts address rows to Python lists.
    row_convert_ms: float = 0.0
    address_assemble_ms: float = 0.0
    layer_meta_calls: int = 0


class UcmDirectLinker(UnifiedCacheLinker):
    """Link SGLang's L1 cache pools to a UCM Store adapter.

    ``adapter`` is injectable for focused tests. In a normal process it is
    created from the UCM configuration.
    """

    def __init__(
        self,
        server_args: Any,
        params: CacheInitParams,
        *,
        components,
        adapter: UcmDirectStoreAdapter | None = None,
    ):
        del server_args
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        self._bundles = self._build_pool_bundles()
        self._layer_pointer_layouts = self._build_layer_pointer_layouts()
        self._closed = False
        self._gc_frozen = False
        self._pending_loads: dict[str, list[PoolTransfer]] = {}
        self._pending_lock = threading.Lock()
        self._load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object, float] | None
        ] = Queue()
        self._completed_loads: Queue[list[str]] = Queue()
        self._offload_queue: Queue[tuple[list[PoolTransfer], object] | None] = Queue()
        self._offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}

        self._tp_rank = 0
        self._tp_size = get_parallel().tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._tp_rank = torch.distributed.get_rank(group=tp_group)
            self._tp_size = torch.distributed.get_world_size(group=tp_group)
        self._offload_owner = not self.pool_group.rank_replicated or self._tp_rank == 0
        self._log_pool_topology()
        try:
            self.adapter = adapter or self._create_adapter(params)
            self.adapter.register_device_pools(self._pool_metadata())
        except Exception:
            logger.exception(
                "UCM linker initialization failed: layers=%d tp=%d/%d replicated=%s.",
                self.num_layers,
                self._tp_rank,
                self._tp_size,
                self.pool_group.rank_replicated,
            )
            raise

        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers, self._tp_rank)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )

        self._load_thread = threading.Thread(
            target=self._load_worker, daemon=True, name=f"ucm-load-tp{self._tp_rank}"
        )
        self._offload_thread = threading.Thread(
            target=self._offload_worker,
            daemon=True,
            name=f"ucm-offload-tp{self._tp_rank}",
        )
        self._load_thread.start()
        self._offload_thread.start()

    def _create_adapter(self, params: CacheInitParams) -> UcmDirectStoreAdapter:
        identity = {
            "model": getattr(get_model(), "model_path", None),
            "tp_rank": self._tp_rank,
            "tp_size": self._tp_size,
            "pp_rank": params.pp_rank,
            "pp_size": params.pp_size,
            "attn_cp_rank": params.attn_cp_rank,
            "attn_cp_size": params.attn_cp_size,
            "rank_replicated": self.pool_group.rank_replicated,
            "device_id": int(device_module.current_device()),
        }
        config = {"config_file": os.environ.get("UNIFIEDCACHE_CONFIG_FILE")}
        return UcmDirectStoreAdapter.from_config(config, identity)

    def _pool_metadata(self) -> list[UcmPoolMetadata]:
        metadata = []
        for bundle in self._bundles:
            for entry in bundle.entries:
                layer_schemas = tuple(
                    (layer, self._entry_layer_schema(entry, layer))
                    for layer in bundle.layers
                )
                metadata.append(
                    UcmPoolMetadata(
                        name=f"{bundle.name}:{entry.name}",
                        pool_name=str(entry.name),
                        source_pool=str(bundle.source_pool),
                        store_group=bundle.name,
                        page_size=entry.page_size,
                        layer_schemas=layer_schemas,
                        device_buffers=tuple(
                            (int(buffer.data_ptr()), int(buffer.nbytes))
                            for buffer in entry.kv_buffer
                            if buffer.data_ptr() and buffer.nbytes
                        ),
                    )
                )
        return metadata

    def _build_pool_bundles(
        self,
    ) -> tuple[_PoolBundle, ...]:
        """Group entries that can share one page/layer Store block schema."""
        grouped: dict[
            tuple[PoolName, tuple[tuple[int, int | tuple[int, ...]], ...]],
            list[DevicePoolEntry],
        ] = {}
        for entry in self.pools.values():
            mapping = tuple(
                (layer, mapped if isinstance(mapped, int) else tuple(mapped))
                for layer, mapped in sorted(entry.layer_mapping.items())
            )
            layers = tuple(layer for layer, _ in mapping)
            if not layers:
                continue
            if any(layer < 0 or layer >= self.num_layers for layer in layers):
                raise ValueError(
                    f"UCM pool {entry.name} has out-of-range layers {layers}."
                )
            grouped.setdefault((entry.indices_from_pool, mapping), []).append(entry)

        bundles = []
        for (source_pool, mapping), entries in grouped.items():
            layers = tuple(layer for layer, _ in mapping)
            ordered = tuple(sorted(entries, key=lambda entry: str(entry.name)))
            by_schema: dict[tuple[int, ...], list[int]] = {}
            for layer in layers:
                schema = tuple(
                    size
                    for entry in ordered
                    for size, _ in self._entry_layer_schema(entry, layer)
                )
                if not schema:
                    raise ValueError(
                        f"UCM pool group {source_pool} has no fragments at layer {layer}."
                    )
                by_schema.setdefault(schema, []).append(layer)
            base_name = (
                f"{source_pool}__{'__'.join(str(entry.name) for entry in ordered)}"
            )
            for index, (_, schema_layers) in enumerate(sorted(by_schema.items())):
                layer_label = "_".join(map(str, schema_layers))
                bundles.append(
                    _PoolBundle(
                        f"{base_name}__shape{index}__layers{layer_label}",
                        source_pool,
                        tuple(schema_layers),
                        ordered,
                    )
                )
        return tuple(bundles)

    def _log_pool_topology(self) -> None:
        logger.info(
            "UCM device-pool topology: layers=%d page_size=%d pools=%d bundles=%d "
            "tp=%d/%d replicated=%s.",
            self.num_layers,
            self.page_size,
            len(self.pools),
            len(self._bundles),
            self._tp_rank,
            self._tp_size,
            self.pool_group.rank_replicated,
        )
        for bundle in self._bundles:
            schema = tuple(
                size
                for layer in bundle.layers
                for entry in sorted(bundle.entries, key=lambda item: str(item.name))
                for size, _ in self._entry_layer_schema(entry, layer)
            )
            logger.info(
                "UCM pool bundle=%s source=%s pools=%s layers=%s "
                "page_block_tensors=%d page_block_bytes=%d.",
                bundle.name,
                bundle.source_pool,
                [str(entry.name) for entry in bundle.entries],
                bundle.layers,
                len(schema),
                sum(schema),
            )

    @staticmethod
    def _entry_layer_schema(
        entry: DevicePoolEntry, layer: int
    ) -> tuple[tuple[int, int], ...]:
        mapped = entry.layer_mapping.get(layer)
        if mapped is None:
            return ()
        buffer_indices = (mapped,) if isinstance(mapped, int) else tuple(mapped)
        return tuple(
            (int(component[index][2]), int(offsets[index]))
            for component, offsets in zip(entry.buffer_meta, entry._component_offsets)
            for index in buffer_indices
        )

    def _build_layer_pointer_layouts(
        self,
    ) -> dict[tuple[PoolName, int], _LayerPointerLayout]:
        layouts: dict[tuple[PoolName, int], _LayerPointerLayout] = {}
        for bundle in self._bundles:
            for entry in bundle.entries:
                for layer in bundle.layers:
                    key = (entry.name, layer)
                    if key in layouts:
                        continue
                    mapped = entry.layer_mapping[layer]
                    buffer_indices = (
                        (mapped,) if isinstance(mapped, int) else tuple(mapped)
                    )
                    # Match DevicePoolEntry.get_prepared_layer_range_meta's
                    # component-then-buffer order exactly.
                    items = [
                        (*component[index], offsets[index])
                        for component, offsets in zip(
                            entry.buffer_meta, entry._component_offsets
                        )
                        for index in buffer_indices
                    ]
                    bases = np.asarray([item[0] for item in items], dtype=np.uint64)
                    strides = np.asarray([item[1] for item in items], dtype=np.uint64)
                    bases.setflags(write=False)
                    strides.setflags(write=False)
                    layouts[key] = _LayerPointerLayout(
                        bases=bases,
                        strides=strides,
                        schema=tuple((int(item[2]), int(item[3])) for item in items),
                    )
        return layouts

    def _entry_layer_rows(
        self,
        entry: DevicePoolEntry,
        transfer: PoolTransfer,
        layer: int,
        locations: np.ndarray,
        timings: _TransferBuildTimings | None = None,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        layout = self._layer_pointer_layouts.get((entry.name, layer))
        if layout is None:
            return np.empty((0, 0), dtype=np.uint64), ()
        page_count = len(transfer.keys or ())
        if not entry.packed and len(layout.schema) != 1:
            raise ValueError(
                f"UCM bundle requires one address row per page for {entry.name}: "
                f"fragments={len(layout.schema)} pages={page_count}."
            )
        meta_start = time.perf_counter() if timings is not None else 0.0
        # Only the page rows change per transfer. Compute the address matrix
        # without rebuilding a Python size/offset list for every page.
        pointer_matrix = (
            locations[:, None] * layout.strides[None, :]
            + layout.bases[None, :]
        )
        if timings is not None:
            timings.layer_meta_ms += (time.perf_counter() - meta_start) * 1000
            timings.layer_meta_calls += 1
        if pointer_matrix.shape[0] != page_count:
            raise ValueError(
                f"UCM bundle requires one address row per page for {entry.name}: "
                f"pointers={pointer_matrix.shape[0]} pages={page_count}."
            )
        return pointer_matrix, layout.schema

    def _page_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        for_io: bool,
        timings: _TransferBuildTimings | None = None,
    ) -> list[UcmLayerTransfer]:
        by_name = {transfer.name: transfer for transfer in transfers}
        result: list[UcmLayerTransfer] = []
        for bundle in self._bundles:
            members = [
                (entry, by_name.get(entry.name))
                for entry in sorted(bundle.entries, key=lambda item: str(item.name))
            ]
            if any(transfer is None or not transfer.keys for _, transfer in members):
                continue
            page_keys = tuple(members[0][1].keys)
            hit_policy = members[0][1].hit_policy
            if any(
                tuple(transfer.keys) != page_keys or transfer.hit_policy != hit_policy
                for _, transfer in members[1:]
            ):
                raise ValueError(
                    f"UCM store bundle {bundle.name} has inconsistent source transfer metadata."
                )

            schema_start = time.perf_counter() if timings is not None else 0.0
            fragments = tuple(
                fragment
                for layer in bundle.layers
                for entry, _ in members
                for fragment in self._entry_layer_schema(entry, layer)
            )
            if not fragments:
                continue
            layer_sizes = tuple(
                sum(
                    size
                    for entry, _ in members
                    for size, _ in self._entry_layer_schema(entry, layer)
                )
                for layer in bundle.layers
            )
            if not layer_sizes or len(set(layer_sizes)) != 1:
                raise ValueError(
                    f"UCM bundle {bundle.name} has inconsistent layer sizes: "
                    f"{layer_sizes}."
                )
            layer_tensor_size_list = tuple(
                size
                for entry, _ in members
                for size, _ in self._entry_layer_schema(entry, bundle.layers[0])
            )
            if timings is not None:
                timings.schema_ms += (time.perf_counter() - schema_start) * 1000
            if for_io:
                locations_start = time.perf_counter() if timings is not None else 0.0
                prepared_locations = []
                for entry, transfer in members:
                    if transfer.host_indices is None:
                        raise ValueError(
                            f"UCM {entry.name} transfer has no device indices."
                        )
                    prepared_locations.append(
                        np.asarray(
                            entry.prepare_locations(transfer.host_indices),
                            dtype=np.uint64,
                        )
                    )
                if timings is not None:
                    timings.locations_ms += (
                        time.perf_counter() - locations_start
                    ) * 1000
                assemble_start = time.perf_counter() if timings is not None else 0.0
                # Each layer slice is contiguous and can be passed to Store
                # without converting every page's pointers to Python lists.
                tensor_count = len(layer_tensor_size_list)
                if len(fragments) != len(bundle.layers) * tensor_count:
                    raise ValueError(
                        f"UCM bundle {bundle.name} has inconsistent fragment counts."
                    )
                addresses = np.empty(
                    (len(bundle.layers), len(page_keys), tensor_count),
                    dtype=np.uint64,
                )
                if timings is not None:
                    timings.address_assemble_ms += (
                        time.perf_counter() - assemble_start
                    ) * 1000
                for layer_index, layer in enumerate(bundle.layers):
                    column = 0
                    for (entry, transfer), locations in zip(
                        members, prepared_locations
                    ):
                        entry_rows, schema = self._entry_layer_rows(
                            entry, transfer, layer, locations, timings
                        )
                        if entry_rows.shape != (len(page_keys), len(schema)) or not schema:
                            raise ValueError(
                                f"UCM bundle {bundle.name} is missing "
                                f"{entry.name} layer={layer}."
                            )
                        assemble_start = (
                            time.perf_counter() if timings is not None else 0.0
                        )
                        next_column = column + len(schema)
                        addresses[layer_index, :, column:next_column] = entry_rows
                        column = next_column
                        if timings is not None:
                            timings.address_assemble_ms += (
                                time.perf_counter() - assemble_start
                            ) * 1000
                    if column != tensor_count:
                        raise ValueError(
                            f"UCM bundle {bundle.name} page address/schema mismatch."
                        )
            else:
                addresses = tuple(() for _ in page_keys)
            result.append(
                UcmLayerTransfer(
                    store_group=bundle.name,
                    source_pool=str(bundle.source_pool),
                    pools=tuple(str(entry.name) for entry in bundle.entries),
                    layers=bundle.layers,
                    shard_size=layer_sizes[0],
                    tensor_size_list=layer_tensor_size_list,
                    page_keys=page_keys,
                    addresses=addresses,
                    fragments=fragments,
                    hit_policy=str(getattr(hit_policy, "value", hit_policy)),
                )
            )
        return result

    def _expand(self, transfers: list[PoolTransfer], **kwargs) -> list[PoolTransfer]:
        return self.pool_group.resolve_transfers(transfers, **kwargs)

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self._expand(transfers)
        if not expanded:
            return []
        kv_transfer = next(
            (transfer for transfer in transfers if transfer.name == PoolName.KV),
            None,
        )
        page_keys = list(kv_transfer.keys or ()) if kv_transfer is not None else []
        if not page_keys:
            return []
        lookup_transfers = self._page_transfers(expanded, for_io=False)
        if not lookup_transfers:
            return []
        result = self.adapter.lookup(lookup_transfers, page_keys)
        self.stats["lookup"] += 1
        if result:
            logger.debug("UCM linker lookup hit: rid=%s pages=%d", rid, result[-1])
        return result

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        expanded = self._expand(transfers, allow_partial=True, allow_missing_kv=True)
        if not expanded:
            return False
        with self._pending_lock:
            if rid in self._pending_loads:
                raise RuntimeError(f"UCM load for rid={rid} is already queued.")
            self._pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # The tree has already made the external hit visible before it invokes
        # this hook. Removing its transfer would leave device slots referenced
        # by that node uninitialized. This matches SGLang's other direct
        # linkers, which only cancel work before the node is published.
        return False

    def num_completed_loads(self) -> int:
        return self._completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self._completed_loads.get_nowait()

    def _freeze_gc_once(self) -> None:
        if not self._gc_frozen:
            freeze_gc("UCM direct linker")
            self._gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        with self._pending_lock:
            if not self._pending_loads:
                return -1
            pending = self._pending_loads
            self._pending_loads = {}
        self._freeze_gc_once()
        counter_index = self.layer_done_counter.update_producer()
        ready_event = device_module.Event()
        ready_event.record()
        self._load_queue.put((counter_index, pending, ready_event, time.perf_counter()))
        self.stats["load"] += len(pending)
        return counter_index

    def _wait_tasks(self, tasks: object) -> None:
        # Wait consumes the CacheStore task handle after returning. Do not call
        # check() afterwards: CacheStore would report -50005 (NotFound) for the
        # already-reaped task even when the transfer completed successfully.
        self.adapter.wait(tasks)

    def _load_worker(self) -> None:
        while True:
            task = self._load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event, queued_at = task
                try:
                    worker_started = time.perf_counter()
                    batch_start = worker_started
                    queue_wait_ms = (worker_started - queued_at) * 1000
                    ready_start = worker_started
                    ready_event.synchronize()
                    ready_wait_ms = (time.perf_counter() - ready_start) * 1000
                    transfer_build_start = time.perf_counter()
                    build_timings = _TransferBuildTimings()
                    page_transfers = [
                        transfer
                        for request in pending.values()
                        for transfer in self._page_transfers(
                            request, for_io=True, timings=build_timings
                        )
                    ]
                    transfer_build_ms = (
                        time.perf_counter() - transfer_build_start
                    ) * 1000
                    transfer_build_other_ms = transfer_build_ms - sum(
                        (
                            build_timings.schema_ms,
                            build_timings.locations_ms,
                            build_timings.layer_meta_ms,
                            build_timings.row_convert_ms,
                            build_timings.address_assemble_ms,
                        )
                    )
                    transfer_prepare_start = time.perf_counter()
                    prepared_transfers = self.adapter.prepare_load(page_transfers)
                    transfer_prepare_ms = (
                        time.perf_counter() - transfer_prepare_start
                    ) * 1000
                    page_count = sum(
                        len(transfer.page_keys) for transfer in page_transfers
                    )
                    if (
                        max(
                            queue_wait_ms,
                            ready_wait_ms,
                            transfer_build_ms,
                            transfer_prepare_ms,
                        )
                        >= _MIN_TIMING_LOG_MS
                    ):
                        logger.info(
                            "UCM linker load batch begin tp_rank=%d batch=%d requests=%d "
                            "transfers=%d transfer_pages=%d queue_wait_ms=%.3f "
                            "ready_event_ms=%.3f transfer_build_ms=%.3f "
                            "transfer_prepare_ms=%.3f build_schema_ms=%.3f "
                            "build_locations_ms=%.3f build_layer_meta_ms=%.3f "
                            "build_row_convert_ms=%.3f build_address_assemble_ms=%.3f "
                            "build_other_ms=%.3f build_layer_meta_calls=%d",
                            self._tp_rank,
                            counter_index,
                            len(pending),
                            len(page_transfers),
                            page_count,
                            queue_wait_ms,
                            ready_wait_ms,
                            transfer_build_ms,
                            transfer_prepare_ms,
                            build_timings.schema_ms,
                            build_timings.locations_ms,
                            build_timings.layer_meta_ms,
                            build_timings.row_convert_ms,
                            build_timings.address_assemble_ms,
                            transfer_build_other_ms,
                            build_timings.layer_meta_calls,
                        )
                    for layer in range(self.num_layers):
                        if page_transfers:
                            # A logical layer can span several pool bundles and
                            # requests; release it only after every shard finishes.
                            layer_start = time.perf_counter()
                            submit_start = layer_start
                            tasks = self.adapter.submit_prepared_load_layer(
                                prepared_transfers, layer
                            )
                            submit_ms = (time.perf_counter() - submit_start) * 1000
                            store_wait_start = time.perf_counter()
                            self._wait_tasks(tasks)
                            store_wait_ms = (
                                time.perf_counter() - store_wait_start
                            ) * 1000
                            layer_ms = (time.perf_counter() - layer_start) * 1000
                            if layer_ms >= _MIN_TIMING_LOG_MS:
                                logger.info(
                                    "UCM linker load layer tp_rank=%d batch=%d layer=%d "
                                    "requests=%d transfers=%d store_tasks=%d submit_ms=%.3f "
                                    "store_wait_ms=%.3f layer_ms=%.3f",
                                    self._tp_rank,
                                    counter_index,
                                    layer,
                                    len(pending),
                                    len(page_transfers),
                                    len(tasks),
                                    submit_ms,
                                    store_wait_ms,
                                    layer_ms,
                                )
                        self.layer_done_counter.complete(counter_index, layer)
                    batch_ms = (time.perf_counter() - batch_start) * 1000
                    if batch_ms >= _MIN_TIMING_LOG_MS:
                        logger.info(
                            "UCM linker load batch end tp_rank=%d batch=%d requests=%d "
                            "total_ms=%.3f",
                            self._tp_rank,
                            counter_index,
                            len(pending),
                            batch_ms,
                        )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("UCM layer-wise load batch failed")
                finally:
                    self._completed_loads.put(list(pending))
            finally:
                self._load_queue.task_done()

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self._expand(transfers, allow_partial=True)
        if not expanded:
            return False
        self._freeze_gc_once()
        if not self._offload_owner:
            self._offload_results.put(True)
            return True
        ready_event = device_module.Event()
        ready_event.record()
        self._offload_queue.put((expanded, ready_event))
        return True

    def _offload_worker(self) -> None:
        while True:
            task = self._offload_queue.get()
            try:
                if task is None:
                    return
                expanded, ready_event = task
                ready_event.synchronize()
                transfers = self._page_transfers(expanded, for_io=True)
                self._wait_tasks(self.adapter.submit_offload(transfers))
                self.stats["offload"] += 1
                self._offload_results.put(True)
            except BaseException:
                logger.exception("UCM offload failed")
                self._offload_results.put(False)
            finally:
                self._offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self._offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self._offload_results.get_nowait()

    @staticmethod
    def _drain(queue: Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    def reset(self) -> None:
        with self._pending_lock:
            self._pending_loads.clear()
        self._load_queue.join()
        self._offload_queue.join()
        self._drain(self._completed_loads)
        self._drain(self._offload_results)
        self.layer_done_counter.reset()

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        self._closed = True
        self._load_queue.put(None)
        self._offload_queue.put(None)
        self._load_thread.join()
        self._offload_thread.join()
        self.adapter.close()
        logger.info("UCM direct linker stats: %s", self.stats)
