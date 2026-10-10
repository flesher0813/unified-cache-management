"""Select and apply the UCM patch matching the installed SGLang version."""

from __future__ import annotations

import importlib
import importlib.metadata
import os

from ucm.logger import init_logger

logger = init_logger(__name__)
# Package directory names follow the vLLM convention and must be valid Python
# identifiers. The public SGLang version remains the mapping key.
SUPPORTED = {"0.5.20": "v0520"}


def apply_sglang_patch() -> None:
    if os.environ.get("ENABLE_UCM_SGLANG_PATCH", "").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    try:
        version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "SGLang patch trigger ran without an installed SGLang package."
        ) from error
    expected = os.environ.get("UCM_SGLANG_VERSION", version)
    if expected not in SUPPORTED:
        raise RuntimeError(
            "Set UCM_SGLANG_VERSION to a supported SGLang version; "
            f"supported versions: {', '.join(SUPPORTED)}. Installed package version={version}."
        )
    package = f"ucm.integration.sglang.patch.{SUPPORTED[expected]}"
    try:
        module = importlib.import_module(package)
        module.apply()
    except Exception as error:
        raise RuntimeError(
            f"Failed to apply UCM SGLang patch for {expected}."
        ) from error
    logger.info("UCM SGLang patch applied for version %s.", expected)
