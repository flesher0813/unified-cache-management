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
selecting `ucm` raises a clear `NotImplementedError` by default before any
cache is attached or transfer worker is started.

For patch and server-startup checks only, set `UCM_SGLANG_ALLOW_NOOP_LINKER=1`.
The no-op linker attaches successfully but always reports an external-cache
miss and never writes external KV data. Edit the local configuration block in
`scripts/launch_sglang_ucm_linker_noop.sh`, then run it to start SGLang with
the UCM backend.

This patch-only step does not implement GPU transfers or external-store
behavior. Runtime patch modules are Python files; no diff files are loaded.
