"""UCM Store V1 adapter used by SGLang's L1 external-cache linker."""

from __future__ import annotations

import copy
import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UcmPoolMetadata:
    name: str
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
    layer: int
    page_keys: tuple[str, ...]
    addresses: tuple[tuple[int, ...], ...]
    fragments: tuple[tuple[int, int], ...]
    hit_policy: str = "all_pages"


@dataclass(frozen=True)
class _StoreTask:
    store: UcmKVStoreBaseV1
    task: Task


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


class UcmDirectStoreAdapter:
    """Map SGLang device-address rows to UCM Store V1 operations.

    One Store is created during device-pool registration for each linker-selected
    ``store_group``. A Store block holds every fixed fragment for one page and
    one logical layer. Thus a DSA KV/indexer group becomes one composite UCM
    block.
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
                composite_schemas = {
                    tuple(
                        (int(size), 0)
                        for member in sorted(members, key=lambda item: item.name)
                        for size, _ in schemas_by_member[member.name][layer]
                    )
                    for layer in layers
                }
                if len(composite_schemas) != 1:
                    raise ValueError("variable per-layer schemas")
                schema = next(iter(composite_schemas), ())
                device_buffers = tuple(
                    dict.fromkeys(
                        buffer for member in members for buffer in member.device_buffers
                    )
                )
                if not device_buffers:
                    raise ValueError("no device buffers")
                logger.info(
                    "UCM register pool group=%s source=%s pools=%s page_size=%d "
                    "layers=%s tensors=%d bytes=%d buffers=%d buffer_bytes=%d.",
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
                self._ensure_store_schema(group, schema, device_buffers)
            except Exception:
                logger.exception(
                    "Failed to register UCM pool group=%s source=%s pools=%s.",
                    group,
                    members[0].source_pool,
                    [member.name for member in members],
                )
                raise

    def _identity_suffix(self) -> str:
        parts = [
            _safe_name(self._identity.get("model") or "model"),
            f"pp{int(self._identity.get('pp_rank', 0))}",
            f"cp{int(self._identity.get('attn_cp_rank', 0))}",
        ]
        if not self._identity.get("rank_replicated", False):
            parts.append(f"tp{int(self._identity.get('tp_rank', 0))}")
        return "_".join(parts)

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
        # Range offsets select a source layer in SGLang's packed pool. The UCM
        # block already represents one layer, so its fixed tensor layout is
        # determined by fragment sizes; each address points at that layer's
        # actual range.
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
        namespace = _safe_name(f"sglang_l1/{self._identity_suffix()}/{group}")
        backends = []
        for root in roots:
            backend = Path(root) / namespace
            backend.mkdir(parents=True, exist_ok=True)
            backends.append(str(backend))
        return backends

    def _ensure_store_schema(
        self,
        store_group: str,
        schema: tuple[tuple[int, int], ...],
        device_buffers: tuple[tuple[int, int], ...] = (),
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

            tensor_size_list = [size for size, _ in schema]
            payload_size = sum(tensor_size_list)
            store_size = payload_size
            if self._store_config.get("io_direct", False):
                store_size = (payload_size + 4095) // 4096 * 4096
            store_config = copy.deepcopy(self._store_config)
            store_config["storage_backends"] = self._namespaced_backends(store_group)
            store_config["unique_id"] = _safe_name(
                f"{self._identity_suffix()}_{store_group}"
            )
            store_config["tensor_layout"] = "sglang_l1"
            store_config["tensor_size_list"] = tensor_size_list
            store_config["tensor_size"] = payload_size
            store_config["shard_size"] = store_size
            store_config["block_size"] = store_size
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
                "Creating SGLang UCM Store group=%s fragments=%d payload=%d "
                "store=%d buffers=%d.",
                store_group,
                len(tensor_size_list),
                payload_size,
                store_size,
                len(device_buffers),
            )
            store = UcmConnectorFactoryV1.create_connector(
                self._connector_name, store_config, self._module_path
            )
            self._stores[store_group] = store
            return store

    def _ensure_store(self, transfer: UcmLayerTransfer) -> UcmKVStoreBaseV1:
        return self._ensure_store_schema(transfer.store_group, self._schema(transfer))

    def _block_ids(
        self, transfer: UcmLayerTransfer, page_keys: Iterable[str] | None = None
    ) -> list[bytes]:
        keys = transfer.page_keys if page_keys is None else tuple(page_keys)
        prefix = f"{self._identity_suffix()}|{transfer.store_group}|l{transfer.layer}|"
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
                    "UCM lookup failed: group=%s source=%s layer=%d pools=%s "
                    "pages=%d.",
                    transfer.store_group,
                    transfer.source_pool,
                    transfer.layer,
                    transfer.pools,
                    len(page_keys),
                )
                raise
        return candidates

    def _submit(
        self, transfers: list[UcmLayerTransfer], *, is_offload: bool
    ) -> list[_StoreTask]:
        tasks: list[_StoreTask] = []
        for transfer in transfers:
            try:
                if len(transfer.page_keys) != len(transfer.addresses):
                    raise ValueError("mismatched page keys and addresses")
                schema = self._schema(transfer)
                if any(len(row) != len(schema) for row in transfer.addresses):
                    raise ValueError("address/schema mismatch")
                store = self._ensure_store(transfer)
                operation = store.dump_data if is_offload else store.load_data
                task = operation(
                    self._block_ids(transfer),
                    [0] * len(transfer.page_keys),
                    [list(row) for row in transfer.addresses],
                )
                tasks.append(_StoreTask(store, task))
            except Exception:
                logger.exception(
                    "UCM %s submit failed: group=%s source=%s layer=%d pools=%s "
                    "pages=%d.",
                    "offload" if is_offload else "load",
                    transfer.store_group,
                    transfer.source_pool,
                    transfer.layer,
                    transfer.pools,
                    len(transfer.page_keys),
                )
                raise
        return tasks

    def submit_load(self, transfers: list[UcmLayerTransfer]) -> list[_StoreTask]:
        return self._submit(transfers, is_offload=False)

    def submit_offload(self, transfers: list[UcmLayerTransfer]) -> list[_StoreTask]:
        return self._submit(transfers, is_offload=True)

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
