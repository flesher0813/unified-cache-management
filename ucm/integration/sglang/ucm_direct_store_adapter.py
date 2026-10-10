"""Store adapter interface reserved for the SGLang L1 integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UcmPoolMetadata:
    name: str
    page_size: int
    layout: tuple[tuple[int, int, int], ...]
    layer_mapping: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class UcmLayerTransfer:
    pool: str
    layer: int
    page_keys: tuple[str, ...]
    addresses: tuple[int, ...]
    fragments: tuple[tuple[int, int, int], ...]


class UcmDirectStoreAdapter:
    """Placeholder API. Store creation and I/O are intentionally unimplemented."""

    @classmethod
    def from_config(cls, config: dict[str, Any], identity: dict[str, Any]):
        raise NotImplementedError

    def register_device_pools(self, pools: list[UcmPoolMetadata]) -> None:
        raise NotImplementedError

    def lookup(self, transfers: list[UcmLayerTransfer]) -> list[int]:
        raise NotImplementedError

    def submit_load(self, transfers: list[UcmLayerTransfer]):
        raise NotImplementedError

    def submit_offload(self, transfers: list[UcmLayerTransfer]):
        raise NotImplementedError

    def wait(self, tasks) -> None:
        raise NotImplementedError

    def check(self, task) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
