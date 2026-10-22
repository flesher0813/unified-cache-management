"""UCM Store V1 adapter used by SGLang's L1 external-cache linker."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UcmPoolMetadata:
    name: str
    pool_name: str
    source_pool: str
    store_group: str
    page_size: int
    layer_schemas: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    device_buffers: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class UcmLayerTransfer:
    store_group: str
    source_pool: str
    pools: tuple[str, ...]
    layers: tuple[int, ...]
    shard_size: int
    tensor_size_list: tuple[int, ...]
    page_keys: tuple[str, ...]
    # I/O uses a contiguous (layer, page, fragment) uint64 array. Lookup
    # retains empty per-page tuples because it does not materialize pointers.
    addresses: tuple[tuple[int, ...], ...] | np.ndarray
    fragments: tuple[tuple[int, int], ...]
    hit_policy: str = "all_pages"


@dataclass(frozen=True)
class _StoreTask:
    store: UcmKVStoreBaseV1
    task: Task


@dataclass(frozen=True)
class _PreparedStoreTransfer:
    transfer: UcmLayerTransfer
    store: UcmKVStoreBaseV1
    block_ids: tuple[bytes, ...]
    tensor_count: int
    shards_by_layer: dict[int, tuple[int, ...]]


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


class UcmDirectStoreAdapter:
    """Map SGLang device-address rows to UCM Store V1 operations.

    One Store is created during device-pool registration for each linker-selected
    ``store_group``. A Store block holds every fixed fragment for one page across
    the group's full logical-layer range.
    """

    def __init__(
        self,
        *,
        connector_name: str,
        module_path: str | None,
        store_config: dict[str, Any],
        identity: dict[str, Any],
    ) -> None:
        self._connector_name = connector_name
        self._module_path = module_path
        self._store_config = store_config
        self._identity = dict(identity)
        self._stores: dict[str, UcmKVStoreBaseV1] = {}
        self._schemas: dict[str, tuple[tuple[int, int], ...]] = {}
        self._pool_metadata: dict[str, UcmPoolMetadata] = {}
        self._store_lock = threading.Lock()

    @classmethod
    def from_config(
        cls, config: dict[str, Any], identity: dict[str, Any]
    ) -> "UcmDirectStoreAdapter":
        config_file = config.get("config_file")
        if not config_file:
            raise ValueError(
                "UNIFIEDCACHE_CONFIG_FILE is required for the SGLang UCM linker."
            )
        path = Path(config_file)
        if not path.is_file():
            raise FileNotFoundError(
                f"SGLang UCM configuration file does not exist: {path}"
            )
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        if not isinstance(document, dict):
            raise ValueError("UCM configuration YAML root must be a mapping.")
        connectors = document.get("ucm_connectors")
        if not isinstance(connectors, list) or len(connectors) != 1:
            raise ValueError(
                "SGLang UCM direct linker requires exactly one ucm_connectors entry."
            )
        connector = connectors[0]
        if not isinstance(connector, dict):
            raise ValueError("UCM connector entry must be a mapping.")
        name = connector.get("ucm_connector_name")
        store_config = connector.get("ucm_connector_config")
        if not isinstance(name, str) or not name:
            raise ValueError("UCM connector entry is missing ucm_connector_name.")
        if not isinstance(store_config, dict):
            raise ValueError("UCM connector entry is missing ucm_connector_config.")
        return cls(
            connector_name=name,
            module_path=connector.get("ucm_connector_module_path"),
            store_config=copy.deepcopy(store_config),
            identity=identity,
        )

    def register_device_pools(self, pools: list[UcmPoolMetadata]) -> None:
        for pool in pools:
            previous = self._pool_metadata.get(pool.name)
            if previous is not None and previous != pool:
                raise ValueError(
                    f"Conflicting UCM metadata for device pool {pool.name}."
                )
            self._pool_metadata[pool.name] = pool
        by_group: dict[str, list[UcmPoolMetadata]] = {}
        for pool in self._pool_metadata.values():
            by_group.setdefault(pool.store_group, []).append(pool)
        for group, members in by_group.items():
            try:
                schemas_by_member = {
                    member.name: dict(member.layer_schemas) for member in members
                }
                layer_sets = {tuple(schemas) for schemas in schemas_by_member.values()}
                if len(layer_sets) != 1:
                    raise ValueError("inconsistent logical layers")
                layers = next(iter(layer_sets), ())
                page_sizes = {member.page_size for member in members}
                if len(page_sizes) != 1:
                    raise ValueError(f"inconsistent page sizes: {sorted(page_sizes)}")
                schema = tuple(
                    (int(size), 0)
                    for layer in layers
                    for member in sorted(members, key=lambda item: item.name)
                    for size, _ in schemas_by_member[member.name][layer]
                )
                if not schema:
                    raise ValueError("no layer fragments")
                device_buffers = tuple(
                    dict.fromkeys(
                        buffer for member in members for buffer in member.device_buffers
                    )
                )
                if not device_buffers:
                    raise ValueError("no device buffers")
                logger.info(
                    "UCM register pool group=%s source=%s pools=%s page_size=%d "
                    "layers=%s tensors=%d page_block_bytes=%d buffers=%d "
                    "buffer_bytes=%d.",
                    group,
                    members[0].source_pool,
                    [member.name for member in members],
                    next(iter(page_sizes)),
                    layers,
                    len(schema),
                    sum(size for size, _ in schema),
                    len(device_buffers),
                    sum(size for _, size in device_buffers),
                )
                layer_tensor_size_list = tuple(
                    size
                    for member in sorted(members, key=lambda item: item.name)
                    for size, _ in schemas_by_member[member.name][layers[0]]
                )
                layer_shard_size = sum(layer_tensor_size_list)
                self._ensure_store_schema(
                    group,
                    schema,
                    device_buffers,
                    shard_size=layer_shard_size,
                    tensor_size_list=layer_tensor_size_list,
                )
            except Exception:
                logger.exception(
                    "Failed to register UCM pool group=%s source=%s pools=%s.",
                    group,
                    members[0].source_pool,
                    [member.name for member in members],
                )
                raise

    def _identity_suffix(self) -> str:
        model = str(self._identity.get("model") or "model")
        model_slug = _safe_name(model).strip("._-")[:48] or "model"
        model_fingerprint = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
        parts = [
            f"{model_slug}-{model_fingerprint}",
            f"pp{int(self._identity.get('pp_rank', 0))}",
            f"cp{int(self._identity.get('attn_cp_rank', 0))}",
        ]
        if not self._identity.get("rank_replicated", False):
            parts.append(f"tp{int(self._identity.get('tp_rank', 0))}")
        return "_".join(parts)

    def _store_unique_id(
        self,
        store_group: str,
        *,
        schema: tuple[tuple[int, int], ...],
        shard_size: int,
        tensor_size_list: tuple[int, ...],
        block_size: int,
    ) -> str:
        """Return a bounded, deterministic ID for Cache/Fake shm objects.

        The C++ backends prepend ``uc_shm_cache_`` or ``uc_shm_fake_`` to this
        value.  Hash the complete sharing identity and layout instead of
        passing the long layer-expanded ``store_group`` directly, while
        preserving the full group in persistent block IDs.
        """
        identity = {
            "model": str(self._identity.get("model") or "model"),
            "pp_rank": int(self._identity.get("pp_rank", 0)),
            "pp_size": int(self._identity.get("pp_size", 1)),
            "cp_rank": int(self._identity.get("attn_cp_rank", 0)),
            "cp_size": int(self._identity.get("attn_cp_size", 1)),
            "tp_size": int(self._identity.get("tp_size", 1)),
            "rank_replicated": bool(self._identity.get("rank_replicated", False)),
        }
        # Replicated TP ranks intentionally share one shm object.  Non-
        # replicated ranks require separate namespaces, matching _identity_suffix.
        if not identity["rank_replicated"]:
            identity["tp_rank"] = int(self._identity.get("tp_rank", 0))
        layout = {
            "store_group": str(store_group),
            "schema": [[int(size), int(offset)] for size, offset in schema],
            "shard_size": int(shard_size),
            "tensor_size_list": [int(size) for size in tensor_size_list],
            "block_size": int(block_size),
        }
        canonical = json.dumps(
            {"identity": identity, "layout": layout},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _schema(transfer: UcmLayerTransfer) -> tuple[tuple[int, int], ...]:
        if not transfer.fragments:
            raise ValueError(f"UCM transfer {transfer.store_group} has no fragments.")
        raw_schema = tuple(
            (int(size), int(offset)) for size, offset in transfer.fragments
        )
        if any(size <= 0 or offset < 0 for size, offset in raw_schema):
            raise ValueError(
                f"UCM transfer {transfer.store_group} has an invalid fragment schema."
            )
        # Range offsets select each source layer in SGLang's packed pool. The
        # UCM block represents one full page, so its fixed tensor layout is the
        # ordered fragments for every layer in the page.
        return tuple((size, 0) for size, _ in raw_schema)

    def _namespaced_backends(self, group: str) -> list[str]:
        configured = self._store_config.get("storage_backends")
        if isinstance(configured, str):
            roots = [item for item in configured.split(":") if item]
        elif isinstance(configured, (list, tuple)):
            roots = [str(item) for item in configured]
        else:
            raise ValueError("UCM connector config requires storage_backends.")
        if not roots:
            raise ValueError("UCM connector config has no storage_backends.")
        members = [
            pool for pool in self._pool_metadata.values() if pool.store_group == group
        ]
        if not members:
            raise ValueError(f"UCM store group {group} has no registered pools.")
        pool_order = {"kv": 0, "indexer": 1, "swa": 2, "mamba": 3}
        pool_names = tuple(
            sorted(
                {_safe_name(pool.pool_name).strip("._-").lower() for pool in members},
                key=lambda name: (pool_order.get(name, 4), name),
            )
        )
        if not pool_names or any(not name for name in pool_names):
            raise ValueError(
                f"UCM store group {group} has an empty physical pool name."
            )
        directory_name = "_".join(pool_names)

        # Usually a physical pool set maps to one Store and gets a simple name
        # such as ``kv`` or ``deepseek_v4_c128``. If mapping/schema splits it
        # into multiple Stores, add a layout suffix so each Store has its own
        # directory without introducing model or parallel-rank identity.
        pools_by_group: dict[str, list[UcmPoolMetadata]] = {}
        for pool in self._pool_metadata.values():
            pools_by_group.setdefault(pool.store_group, []).append(pool)
        matching_groups = {
            other_group: other_members
            for other_group, other_members in pools_by_group.items()
            if tuple(
                sorted(
                    {
                        _safe_name(candidate.pool_name).strip("._-").lower()
                        for candidate in other_members
                    },
                    key=lambda name: (pool_order.get(name, 4), name),
                )
            )
            == pool_names
        }
        if len(matching_groups) > 1:
            layer_ids = tuple(
                sorted({layer for pool in members for layer, _ in pool.layer_schemas})
            )
            if layer_ids:
                contiguous = layer_ids == tuple(range(layer_ids[0], layer_ids[-1] + 1))
                layer_label = (
                    f"{layer_ids[0]}-{layer_ids[-1]}"
                    if contiguous
                    else "_".join(map(str, layer_ids))
                )
                directory_name = f"{directory_name}_layers_{layer_label}"
            # Layer ranges normally distinguish split Stores. If the same pool
            # set and layer range still collide, use a stable layout fingerprint.
            if (
                sum(
                    tuple(
                        sorted(
                            {
                                layer
                                for pool in other_members
                                for layer, _ in pool.layer_schemas
                            }
                        )
                    )
                    == layer_ids
                    for other_members in matching_groups.values()
                )
                > 1
            ):
                fingerprint = hashlib.sha256(group.encode("utf-8")).hexdigest()[:10]
                directory_name = f"{directory_name}_layout_{fingerprint}"
        backends = []
        for root in roots:
            backend = Path(root) / directory_name
            backend.mkdir(parents=True, exist_ok=True)
            backends.append(str(backend))
        return backends

    def _ensure_store_schema(
        self,
        store_group: str,
        schema: tuple[tuple[int, int], ...],
        device_buffers: tuple[tuple[int, int], ...] = (),
        *,
        shard_size: int,
        tensor_size_list: tuple[int, ...],
    ) -> UcmKVStoreBaseV1:
        with self._store_lock:
            schema = tuple((int(size), 0) for size, _ in schema)
            if not schema or any(size <= 0 for size, _ in schema):
                raise ValueError(f"UCM store group {store_group} has no fragments.")
            existing_schema = self._schemas.setdefault(store_group, schema)
            if existing_schema != schema:
                raise ValueError(
                    "One UCM store group must have a fixed page/layer schema: "
                    f"group={store_group}."
                )
            existing = self._stores.get(store_group)
            if existing is not None:
                return existing

            tensor_size_list = [int(size) for size in tensor_size_list]
            payload_size = sum(size for size, _ in schema)
            if not tensor_size_list or sum(tensor_size_list) != shard_size:
                raise ValueError(
                    "Store tensor_size_list must describe exactly one shard: "
                    f"group={store_group} tensor_size_list={tensor_size_list} "
                    f"shard_size={shard_size}."
                )
            if shard_size <= 0 or payload_size % shard_size:
                raise ValueError(
                    "Store block_size must be an integer number of shards: "
                    f"group={store_group} shard_size={shard_size} "
                    f"block_size={payload_size}."
                )
            store_config = copy.deepcopy(self._store_config)
            store_config["storage_backends"] = self._namespaced_backends(store_group)
            # ``unique_id`` is used in Cache/Fake POSIX shared-memory names.
            # Keep it bounded and include the actual layout so incompatible
            # Stores cannot accidentally attach to the same shared buffer.
            store_config["unique_id"] = self._store_unique_id(
                store_group,
                schema=schema,
                shard_size=shard_size,
                tensor_size_list=tuple(tensor_size_list),
                block_size=payload_size,
            )
            store_config["tensor_layout"] = "sglang_l1"
            store_config["tensor_size_list"] = tensor_size_list
            # CacheStore treats a scalar tensor_size as higher priority than
            # tensor_size_list. Do not collapse a composite DSA payload such
            # as [indexer_bytes, kv_bytes] into one contiguous tensor here.
            # Pipeline backends that need a scalar set it on their private
            # downstream configuration (for example, Posix uses shard_size).
            store_config["shard_size"] = shard_size
            store_config["block_size"] = payload_size
            store_config["device_id"] = int(self._identity.get("device_id", 0))
            store_config["gpu_kv_buffer_addrs"] = [
                address for address, _ in device_buffers
            ]
            store_config["gpu_kv_buffer_sizes"] = [size for _, size in device_buffers]
            store_config["local_rank_size"] = (
                int(self._identity.get("tp_size", 1))
                if self._identity.get("rank_replicated", False)
                else 1
            )
            logger.info(
                "Creating SGLang UCM Store group=%s fragments=%d "
                "page_shard_bytes=%d page_block_bytes=%d layers=%d buffers=%d.",
                store_group,
                len(tensor_size_list),
                shard_size,
                payload_size,
                payload_size // shard_size,
                len(device_buffers),
            )
            logger.info(f"SGLang UCM Store config: {store_config}")
            store = UcmConnectorFactoryV1.create_connector(
                self._connector_name, store_config, self._module_path
            )
            self._stores[store_group] = store
            return store

    def _ensure_store(self, transfer: UcmLayerTransfer) -> UcmKVStoreBaseV1:
        return self._ensure_store_schema(
            transfer.store_group,
            self._schema(transfer),
            shard_size=transfer.shard_size,
            tensor_size_list=transfer.tensor_size_list,
        )

    def _block_ids(
        self, transfer: UcmLayerTransfer, page_keys: Iterable[str] | None = None
    ) -> list[bytes]:
        keys = transfer.page_keys if page_keys is None else tuple(page_keys)
        prefix = f"{self._identity_suffix()}|{transfer.store_group}|"
        return [hashlib.md5(f"{prefix}{key}".encode("utf-8")).digest() for key in keys]

    @staticmethod
    def _is_trailing(transfer: UcmLayerTransfer) -> bool:
        return str(transfer.hit_policy) == "trailing_pages"

    @staticmethod
    def _apply_hit_policy(
        candidates: list[int], exists: list[bool], transfer: UcmLayerTransfer
    ) -> list[int]:
        prefix = [0]
        for present in exists:
            prefix.append(prefix[-1] + int(bool(present)))
        if not UcmDirectStoreAdapter._is_trailing(transfer):
            return [end for end in candidates if prefix[end] == end]
        trailing = max(1, len(transfer.page_keys))
        return [
            end
            for end in candidates
            if prefix[end] - prefix[max(0, end - trailing)]
            == end - max(0, end - trailing)
        ]

    def lookup(
        self, transfers: list[UcmLayerTransfer], page_keys: list[str]
    ) -> list[int]:
        if not transfers or not page_keys:
            return []
        candidates = list(range(1, len(page_keys) + 1))
        for transfer in transfers:
            try:
                store = self._ensure_store(transfer)
                keys = page_keys[: candidates[-1]]
                exists = [
                    bool(value)
                    for value in store.lookup(self._block_ids(transfer, keys))
                ]
                if len(exists) != len(keys):
                    raise RuntimeError(f"expected={len(keys)} actual={len(exists)}")
                candidates = self._apply_hit_policy(candidates, exists, transfer)
                if not candidates:
                    return []
            except Exception:
                logger.exception(
                    "UCM lookup failed: group=%s source=%s layers=%s pools=%s "
                    "pages=%d.",
                    transfer.store_group,
                    transfer.source_pool,
                    transfer.layers,
                    transfer.pools,
                    len(page_keys),
                )
                raise
        return candidates

    def _prepare_transfers(
        self, transfers: list[UcmLayerTransfer], *, is_offload: bool
    ) -> list[_PreparedStoreTransfer]:
        prepared: list[_PreparedStoreTransfer] = []
        for transfer in transfers:
            try:
                schema = self._schema(transfer)
                if isinstance(transfer.addresses, np.ndarray):
                    if (
                        transfer.addresses.shape
                        != (
                            len(transfer.layers),
                            len(transfer.page_keys),
                            len(transfer.tensor_size_list),
                        )
                        or transfer.addresses.dtype != np.uint64
                        or not transfer.addresses.flags.c_contiguous
                    ):
                        raise ValueError("address/schema mismatch")
                else:
                    if len(transfer.page_keys) != len(transfer.addresses):
                        raise ValueError("mismatched page keys and addresses")
                    if any(len(row) != len(schema) for row in transfer.addresses):
                        raise ValueError("address/schema mismatch")
                tensor_count = len(transfer.tensor_size_list)
                if (
                    not tensor_count
                    or len(schema) != len(transfer.layers) * tensor_count
                ):
                    raise ValueError("layer count/address schema mismatch")
                if transfer.shard_size != sum(transfer.tensor_size_list):
                    raise ValueError("tensor sizes do not match shard size")
                shard_indices: dict[int, list[int]] = {}
                for shard_index, layer in enumerate(transfer.layers):
                    start = shard_index * tensor_count
                    stop = start + tensor_count
                    sizes = tuple(size for size, _ in schema[start:stop])
                    if sizes != transfer.tensor_size_list:
                        raise ValueError(
                            f"UCM group {transfer.store_group} layer={layer} "
                            "does not match the Store shard schema."
                        )
                    shard_indices.setdefault(layer, []).append(shard_index)
                prepared.append(
                    _PreparedStoreTransfer(
                        transfer=transfer,
                        store=self._ensure_store_schema(
                            transfer.store_group,
                            schema,
                            shard_size=transfer.shard_size,
                            tensor_size_list=transfer.tensor_size_list,
                        ),
                        block_ids=tuple(self._block_ids(transfer)),
                        tensor_count=tensor_count,
                        shards_by_layer={
                            layer: tuple(indices)
                            for layer, indices in shard_indices.items()
                        },
                    )
                )
            except Exception:
                logger.exception(
                    "UCM %s prepare failed: group=%s source=%s layers=%s pools=%s "
                    "pages=%d.",
                    "offload" if is_offload else "load",
                    transfer.store_group,
                    transfer.source_pool,
                    transfer.layers,
                    transfer.pools,
                    len(transfer.page_keys),
                )
                raise
        return prepared

    def prepare_load(
        self, transfers: list[UcmLayerTransfer]
    ) -> list[_PreparedStoreTransfer]:
        return self._prepare_transfers(transfers, is_offload=False)

    def _submit_prepared(
        self,
        prepared: list[_PreparedStoreTransfer],
        *,
        is_offload: bool,
        logical_layer: int | None = None,
    ) -> list[_StoreTask]:
        tasks: list[_StoreTask] = []
        for item in prepared:
            transfer = item.transfer
            try:
                operation = item.store.dump_data if is_offload else item.store.load_data
                shard_indices = (
                    range(len(transfer.layers))
                    if logical_layer is None
                    else item.shards_by_layer.get(logical_layer, ())
                )
                for shard_index in shard_indices:
                    # The Store index is local to this bundle; transfer.layers
                    # contains the corresponding SGLang logical layer.
                    start = shard_index * item.tensor_count
                    stop = start + item.tensor_count
                    addresses = (
                        transfer.addresses[shard_index]
                        if isinstance(transfer.addresses, np.ndarray)
                        else [list(row[start:stop]) for row in transfer.addresses]
                    )
                    task = operation(
                        list(item.block_ids),
                        [shard_index] * len(item.block_ids),
                        addresses,
                    )
                    tasks.append(_StoreTask(item.store, task))
            except Exception:
                logger.exception(
                    "UCM %s submit failed: group=%s source=%s layers=%s pools=%s "
                    "pages=%d.",
                    "offload" if is_offload else "load",
                    transfer.store_group,
                    transfer.source_pool,
                    transfer.layers,
                    transfer.pools,
                    len(transfer.page_keys),
                )
                raise
        return tasks

    def submit_load(self, transfers: list[UcmLayerTransfer]) -> list[_StoreTask]:
        return self._submit_prepared(self.prepare_load(transfers), is_offload=False)

    def submit_load_layer(
        self, transfers: list[UcmLayerTransfer], logical_layer: int
    ) -> list[_StoreTask]:
        selected = [
            transfer for transfer in transfers if logical_layer in transfer.layers
        ]
        return self.submit_prepared_load_layer(
            self.prepare_load(selected), logical_layer
        )

    def submit_prepared_load_layer(
        self, prepared: list[_PreparedStoreTransfer], logical_layer: int
    ) -> list[_StoreTask]:
        return self._submit_prepared(
            prepared, is_offload=False, logical_layer=logical_layer
        )

    def submit_offload(self, transfers: list[UcmLayerTransfer]) -> list[_StoreTask]:
        return self._submit_prepared(
            self._prepare_transfers(transfers, is_offload=True), is_offload=True
        )

    def wait(self, tasks: _StoreTask | Iterable[_StoreTask] | None) -> None:
        if tasks is None:
            return
        iterable = (tasks,) if isinstance(tasks, _StoreTask) else tasks
        for item in iterable:
            item.store.wait(item.task)

    def check(self, task: _StoreTask) -> bool:
        return bool(task.store.check(task.task))

    def close(self) -> None:
        with self._store_lock:
            stores = tuple(self._stores.values())
            self._stores.clear()
        for store in stores:
            close = getattr(store, "close", None)
            if callable(close):
                close()
