"""hekaiyue 何恺悦 2024-03-19"""
import importlib
import pkgutil

engines = dict()
for _, module_name, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    engines[module_name] = getattr(module, "Engine")
