# David W. Romero, 2025-09-09

"""Utility functions for the MNIST classification experiment."""

import importlib.util
from pathlib import Path
from typing import Any, List

import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
import re


from nvsubquadratic.examples.mnist_classification.mnist_classification_cfg import ExperimentConfig


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """
    Load a configuration from a Python file.

    Args:
        config_path: Path to the configuration file

    Returns:
        The loaded configuration
    """
    # Convert path to module path
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Extract the module path
    module_path = str(path).replace("/", ".").replace("\\", ".")
    if module_path.endswith(".py"):
        module_path = module_path[:-3]  # Remove .py extension

    # Import the module
    spec = importlib.util.spec_from_file_location(module_path, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the get_config function
    if not hasattr(module, "get_config"):
        raise AttributeError(f"Configuration file {config_path} must have a get_config() function")

    # Call the get_config function to get the configuration
    return module.get_config()


def apply_config_overrides(config: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    """
    Apply command-line overrides to a configuration.

    Args:
        config: The base configuration
        overrides: List of overrides in the format "key=value"

    Returns:
        Updated configuration
    """

    # Convert config to a nested container that OmegaConf can resolve.
    # This expands nested DictConfigs and dataclasses so ${...} paths can work.
    def _to_nested_container(obj: Any) -> Any:
        import dataclasses as _dc

        from omegaconf import DictConfig as _DictConfig
        from omegaconf import OmegaConf as _OC

        # Dataclass -> dict (recursively)
        if _dc.is_dataclass(obj):
            result = {}
            for f in _dc.fields(obj):
                result[f.name] = _to_nested_container(getattr(obj, f.name))
            return result

        # OmegaConf DictConfig -> plain container (then post-process)
        if isinstance(obj, _DictConfig):
            plain = _OC.to_container(obj, resolve=False)
            return _to_nested_container(plain)

        # dict -> recurse
        if isinstance(obj, dict):
            return {k: _to_nested_container(v) for k, v in obj.items()}

        # list/tuple -> recurse
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_to_nested_container(v) for v in obj)

        # Dataclass objects that might be nested inside DictConfigs
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _to_nested_container(getattr(obj, k)) for k in obj.__dataclass_fields__.keys()}

        # Primitive or other objects (functions, classes) kept as-is
        return obj

    config_dict = _to_nested_container(config)

    # Process each override on the nested container
    for override in overrides:
        if "=" not in override:
            print(f"Warning: Ignoring invalid override '{override}'. Must be in format 'key=value'.")
            continue

        key, value = override.split("=", 1)

        # Convert value to appropriate type
        try:
            # Try to parse as int
            value = int(value)
        except ValueError:
            try:
                # Try to parse as float
                value = float(value)
            except ValueError:
                # Handle booleans
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                # Otherwise keep as string

        # Apply the override
        key_parts = key.split(".")

        # Navigate to the correct part of the config
        current_dict = config_dict
        for i, part in enumerate(key_parts[:-1]):
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        # Set the value
        current_dict[key_parts[-1]] = value

    # Resolve ${...} interpolations while preserving DictConfig for dot-access
    from omegaconf import DictConfig as _DictConfig
    from omegaconf import OmegaConf as _OC

    resolved_conf: _DictConfig = _OC.create(config_dict, flags={"allow_objects": True})
    _OC.resolve(resolved_conf)

    # Convert dictionary back to dataclass
    # This is a simple approach - for a more robust solution,
    # consider using a library like dacite
    def dict_to_dataclass(data_dict, data_class):
        from omegaconf import DictConfig as _DictConfig

        # Get field names
        fields = {f.name for f in dataclasses.fields(data_class)}

        # Prepare kwargs for the dataclass
        kwargs = {}
        # Support both plain dict and OmegaConf DictConfig
        items_iter = data_dict.items() if not isinstance(data_dict, _DictConfig) else list(data_dict.items())
        for key, value in items_iter:
            if key in fields:
                field_type = next(f.type for f in dataclasses.fields(data_class) if f.name == key)
                # If value is a mapping-like (dict or DictConfig) and field is a dataclass, recursively convert
                if isinstance(value, (dict, _DictConfig)) and hasattr(field_type, "__dataclass_fields__"):
                    kwargs[key] = dict_to_dataclass(value, field_type)
                else:
                    kwargs[key] = value

        # Create new instance
        return data_class(**kwargs)

    # Attempt to create new dataclass instance from the resolved config
    new_config = dict_to_dataclass(resolved_conf, ExperimentConfig)
    return new_config


def verify_no_interpolator_overwrites(config: ExperimentConfig, overrides: List[str]) -> None:
    """
    Prevent overriding fields that are defined as OmegaConf interpolations (e.g., "${...}").

    Args:
        config: The base configuration (dataclass with nested LazyConfigs/DictConfigs)
        overrides: CLI overrides in the form ["key=value", ...]

    Raises:
        ValueError: If any override targets a field whose current value is an interpolation string.
    """

    def _to_nested_container(obj: Any) -> Any:
        import dataclasses as _dc

        from omegaconf import DictConfig as _DictConfig
        from omegaconf import OmegaConf as _OC

        if _dc.is_dataclass(obj):
            return {f.name: _to_nested_container(getattr(obj, f.name)) for f in _dc.fields(obj)}
        if isinstance(obj, _DictConfig):
            plain = _OC.to_container(obj, resolve=False)
            return _to_nested_container(plain)
        if isinstance(obj, dict):
            return {k: _to_nested_container(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_to_nested_container(v) for v in obj)
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _to_nested_container(getattr(obj, k)) for k in obj.__dataclass_fields__.keys()}
        return obj

    nested = _to_nested_container(config)
    interp_re = re.compile(r"^\${[^}]+}$")
    violations: list[str] = []

    for override in overrides or []:
        if "=" not in override:
            continue
        key, _ = override.split("=", 1)
        parts = key.split(".")
        parent = nested
        valid_path = True
        for p in parts[:-1]:
            if isinstance(parent, dict) and p in parent:
                parent = parent[p]
            else:
                valid_path = False
                break
        if not valid_path:
            continue
        last = parts[-1]
        if isinstance(parent, dict) and last in parent:
            current_value = parent[last]
            if isinstance(current_value, str) and interp_re.match(current_value):
                violations.append(key)

    if violations:
        raise ValueError(
            "The following overrides target interpolated fields and are not allowed: " + ", ".join(violations)
        )


def set_global_seed(seed: int):
    """Set the global seed for the program."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # CPU RNG
    torch.cuda.manual_seed_all(seed)  # GPU RNG
    pl.seed_everything(seed, workers=True)
