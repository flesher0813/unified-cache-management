"""Apply the versioned UCM external-linker additions to SGLang modules."""

from __future__ import annotations

import importlib

from ucm.logger import init_logger

logger = init_logger(__name__)
EXPECTED_VERSION = "0.5.20"


def apply() -> None:
    import importlib.metadata

    version = importlib.metadata.version("sglang")
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"SGLang version mismatch: expected {EXPECTED_VERSION}, got {version}."
        )
    registry_patch = importlib.import_module(
        "ucm.integration.sglang.patch.v0520.registry_patch"
    )
    registry_patch.apply()
    from . import server_args_patch

    server_args_patch.apply()
    logger.info("Applied UCM SGLang linker patch for %s.", EXPECTED_VERSION)
