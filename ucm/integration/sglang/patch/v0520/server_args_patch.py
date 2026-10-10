"""Inject the UCM linker option before SGLang builds CLI argument groups."""

from __future__ import annotations


def apply() -> None:
    from sglang.srt.arg_groups import arg_utils
    from sglang.srt.server_args import ServerArgs

    # msgspec struct schemas are immutable after class creation. Accept `ucm`
    # through the existing choice field's parser only when the patch is enabled;
    # the direct-linker factory still validates and rejects unsupported values.
    def patch_helper(owner, name):
        original = getattr(owner, name)
        if getattr(original, "_ucm_patched", False):
            return

        def add_cli_args_from_dataclass(parser, cls, *args, **kwargs):
            original(parser, cls, *args, **kwargs)
            action = next(
                (
                    item
                    for item in parser._actions
                    if item.dest == "unified_cache_external_linker_backend"
                ),
                None,
            )
            if action is not None:
                action.choices = set(action.choices or ()) | {"ucm"}
            elif cls is ServerArgs:
                raise RuntimeError("SGLang linker backend CLI action is missing.")

        add_cli_args_from_dataclass._ucm_patched = True
        setattr(owner, name, add_cli_args_from_dataclass)

    patch_helper(arg_utils, "add_cli_args_from_dataclass")
    import sglang.srt.server_args as server_args

    patch_helper(server_args, "add_cli_args_from_dataclass")

    from sglang.srt.arg_groups import hicache_hook

    original = hicache_hook.handle_hicache
    if getattr(original, "_ucm_wrapped", False):
        return

    def handle_hicache(server_args):
        if (
            server_args.enable_unified_cache_external_linker
            and server_args.unified_cache_external_linker_backend == "ucm"
            and server_args.enable_linker_mla_dedup
        ):
            raise ValueError("UCM direct linker does not support MLA deduplication.")
        return original(server_args)

    handle_hicache._ucm_wrapped = True
    hicache_hook.handle_hicache = handle_hicache
