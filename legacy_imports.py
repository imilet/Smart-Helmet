import importlib
import sys


def use_legacy_package(target: str) -> None:
    if target not in {"yolo", "face"}:
        raise ValueError(f"不支持的兼容包目标: {target}")

    for module_name in list(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]

    model_names = ["common", "experimental", "yolo", "mobilefacenet"]
    util_names = [
        "datasets",
        "general",
        "torch_utils",
        "google_utils",
        "autoanchor",
        "activations",
        "plots",
        "metrics",
        "utils",
    ]

    models_pkg = importlib.import_module("models")
    utils_pkg = importlib.import_module("utils")

    for name in model_names:
        try:
            module = importlib.import_module(f"{target}.models.{name}")
        except ModuleNotFoundError:
            continue
        sys.modules[f"models.{name}"] = module
        setattr(models_pkg, name, module)

    for name in util_names:
        try:
            module = importlib.import_module(f"{target}.utils.{name}")
        except ModuleNotFoundError:
            continue
        sys.modules[f"utils.{name}"] = module
        setattr(utils_pkg, name, module)
