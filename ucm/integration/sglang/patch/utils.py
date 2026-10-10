"""Small import-hook and patch helpers shared by SGLang patch versions."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections import defaultdict

from ucm.logger import init_logger

logger = init_logger(__name__)
_HOOKS = defaultdict(list)


def patch_or_inject(target, name, value):
    setattr(target, name, value)
    return value


class _HookFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _HOOKS:
            return None
        saved = sys.meta_path[:]
        sys.meta_path = [finder for finder in sys.meta_path if finder is not self]
        try:
            spec = importlib.util.find_spec(fullname, path)
        finally:
            sys.meta_path = saved
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader

        class _Loader(importlib.abc.Loader):
            def create_module(self, module_spec):
                create = getattr(loader, "create_module", None)
                return create(module_spec) if create else None

            def exec_module(self, module):
                loader.exec_module(module)
                for hook in _HOOKS[fullname]:
                    hook(module)

        spec.loader = _Loader()
        return spec


_FINDER = _HookFinder()


def when_imported(module_name):
    def register(function):
        _HOOKS[module_name].append(function)
        if module_name in sys.modules:
            function(sys.modules[module_name])
        elif _FINDER not in sys.meta_path:
            sys.meta_path.insert(0, _FINDER)
        return function

    return register
