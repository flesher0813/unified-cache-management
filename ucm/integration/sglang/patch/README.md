# SGLang L1 external-linker patch

The patch integration is opt-in. Install UCM with its `ucm_patch.pth`, then set
`ENABLE_UCM_SGLANG_PATCH=1` and `UCM_SGLANG_VERSION=0.5.20` before importing
SGLang. It validates the installed package version and applies the versioned
server-argument and registry patches.

The `v0520` package targets SGLang 0.5.20. Its registry patch reproduces the
matching SGLang factory and adds only
the `ucm` backend branch. Mooncake and Mori retain their upstream selection,
initialization, and default behavior. The UCM store adapter in
`ucm_direct_store_adapter.py` reads the single `ucm_connectors` entry from
`UNIFIEDCACHE_CONFIG_FILE` and creates one UCM Store per linker-selected
device-pool group. It namespaces configured storage backends by model/parallel
identity and Store group, then stores one logical page/layer payload per UCM
block. DSA KV and INDEXER data share one composite block when they belong to
the same pool group.

Edit the local configuration block in
`scripts/launch_sglang_ucm_linker.sh`, then run it to start SGLang with the
UCM backend and a real Store. The launcher exports the patch and UCM
configuration variables, verifies that `ucm_patch.pth` activated the runtime
patch, and writes the service and patch checks to its log directory.

Runtime patch modules are Python files; no diff files are loaded.
