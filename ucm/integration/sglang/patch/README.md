# SGLang L1 external-linker patch

The patch integration is opt-in. Install UCM with its `ucm_patch.pth`, then set
`ENABLE_UCM_SGLANG_PATCH=1` and `UCM_SGLANG_VERSION=0.5.20` before importing
SGLang. It validates the installed package version and applies the versioned
server-argument and registry patches.

The `v0520` package targets SGLang 0.5.20. Its registry patch reproduces the
matching SGLang factory and adds only
the `ucm` backend branch. Mooncake and Mori retain their upstream selection,
initialization, and default behavior. The UCM store adapter in
`ucm_direct_store_adapter.py` intentionally contains interface stubs only, so
selecting `ucm` raises a clear `NotImplementedError` before any cache is
attached or transfer worker is started.

This patch-only step does not implement GPU transfers or external-store
behavior. Runtime patch modules are Python files; no diff files are loaded.
