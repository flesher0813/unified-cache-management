"""Add UCM to SGLang 0.5.20's external-linker factory."""

from __future__ import annotations


def apply() -> None:
    """Patch only the UCM backend; preserve all built-in backend behavior."""
    from sglang.srt.mem_cache import registry

    original = registry.create_unified_radix_cache
    if getattr(original, "_ucm_patched", False):
        return

    def create_unified_radix_cache(ctx, *, cache_class=None):
        # This is SGLang 0.5.20's factory with one additional linker branch.
        # Keeping the complete factory avoids changing setup for Mooncake, Mori,
        # HiCache, hybrid cache components, and platform-specific components.
        from sglang.srt.hardware_backend.mlx.runtime import use_mlx
        from sglang.srt.mem_cache.unified_cache.components import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.runtime_context import get_disagg, get_memory

        server_args, params = ctx.server_args, ctx.params
        if get_disagg().disaggregation_decode_retraction_backup == "host_pool":
            if ctx.is_hybrid_ssm:
                raise ValueError("Host-pool retraction does not support Mamba models.")
            if ctx.is_hybrid_swa and ctx.full_tokens_per_layer == 0:
                raise ValueError(
                    "Host-pool retraction does not support pure-SWA models."
                )

        tree_components = [ComponentType.FULL]
        if ctx.is_hybrid_swa:
            tree_components.append(ComponentType.SWA)
        if ctx.is_hybrid_ssm:
            tree_components.append(ComponentType.MAMBA)

        if hasattr(params.req_to_token_pool, "req_to_c128_sidecar"):
            from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import (
                C128SidecarComponent,
            )

            tree_components.append(ComponentType.C128)
            params.component_registry_override = {
                **(params.component_registry_override or {}),
                ComponentType.C128: C128SidecarComponent,
            }

        params.tree_components = tuple(tree_components)
        if use_mlx() and ctx.is_hybrid_ssm:
            from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
                MlxAuxiliaryStateComponent,
            )

            params.component_registry_override = {
                ComponentType.MAMBA: MlxAuxiliaryStateComponent,
            }

        cache = (cache_class or UnifiedRadixCache)(params)
        if (
            ctx.enable_hierarchical_cache
            or get_disagg().disaggregation_decode_retraction_backup == "host_pool"
        ):
            cache.init_hicache(server_args, params)
            ctx.tp_worker.register_hicache_layer_transfer_counter(
                cache.cache_controller.layer_done_counter
            )
        elif get_memory().enable_unified_cache_external_linker:
            backend = get_memory().unified_cache_external_linker_backend
            if backend == "mooncake":
                from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
                    MooncakeDirectLinker,
                )

                linker_cls = MooncakeDirectLinker
            elif backend == "mori":
                from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
                    UMBPDirectLinker,
                )

                linker_cls = UMBPDirectLinker
            elif backend == "ucm":
                from ucm.integration.sglang.ucm_direct_linker import UcmDirectLinker

                linker_cls = UcmDirectLinker
            else:
                raise ValueError(
                    f"Unknown unified cache external linker backend: {backend!r}"
                )

            cache.init_cache_linker(
                linker_cls(server_args, params, components=set(cache.components))
            )
            counter = cache.linker.layer_done_counter
            kvcache = params.token_to_kv_pool_allocator.get_kvcache()
            kvcache.register_layer_transfer_counter(counter)
            ctx.tp_worker.register_hicache_layer_transfer_counter(counter)
        return cache

    create_unified_radix_cache._ucm_patched = True
    registry.create_unified_radix_cache = create_unified_radix_cache
