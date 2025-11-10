# TODO: Add license header here


"""Utility functions for the MNIST classification experiment."""

import dataclasses
import datetime
import importlib.util
import re
from pathlib import Path
import getpass
from typing import Any, List

from rich.tree import Tree

from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


_SHORT_NAME_ALIASES = {
    "weight_decay": "wd",
    "warmup_iterations": "warm_its",
    "iterations": "its",
    "batch_size": "bs",
    "augmentation_scheme": "aug",
    "omega_0": "w0",
    "__target__": "",
    # values
    "advanced_with_cutmix_and_mixup": "adv_cutmix_mixup",
    "advanced": "adv",
}


def get_deterministic_run_name(config_path: str, overrides: List[str] = None, use_timestamp: bool = True) -> str:
    """
    Generate a deterministic run name based on the config file name, current timestamp, and any overrides.

    Args:
        config_path: Path to the configuration file
        overrides: List of overrides in the format "key=value"
        use_timestamp: Whether to include the timestamp in the run name
    Returns:
        A deterministic run name in the format: {config_name}_{timestamp}_{override_hash}
    """
    # Extract config name without extension and directory path
    # Remove the base config/experiments/ folder and the .py extension.
    # And replace / with .
    config_name = config_path.replace("experiments/", "").replace(".py", "").replace("/", "_")

    # Get current timestamp in format: YYYY-MM-DD-HH-MM-SS
    if use_timestamp:
        timestamp = "_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    else:
        timestamp = ""
    # Always append the effective username to avoid collisions across users
    raw_username = getpass.getuser().upper()
    parts = [p for p in raw_username.split(".") if p]
    if len(parts) >= 2:
        username = parts[0][0] + parts[1][0]
    else:
        username = raw_username[:2] if raw_username else "??"

    # Add override hash if overrides are provided
    if overrides and len(overrides) > 0:
        # Filter out debug-related overrides
        filtered_overrides = [override for override in overrides if not override.startswith("debug")]

        # Filter out wandb-related overrides
        filtered_overrides = [override for override in filtered_overrides if not override.startswith("wandb.")]

        # If we still have overrides after filtering
        if filtered_overrides:
            # Remove the first part of the overrides to shorten the name
            processed_overrides = []
            for override in filtered_overrides:
                key, value = override.split("=", 1)
                short_key = key.split(".")[-1]
                # replace short key with short alias if it exists
                short_key = _SHORT_NAME_ALIASES.get(short_key, short_key)
                # replace value with short alias if it exists
                value = _SHORT_NAME_ALIASES.get(value, value)
                processed_overrides.append(f"{short_key}={value}")
            filtered_overrides = processed_overrides
            # Sort overrides for deterministic ordering
            filtered_overrides.sort()
            # Create a string representation of the overrides
            override_str = "_".join(filtered_overrides).replace("=", "_")
            return (
                f"{username.upper()}_{config_name}_{override_str}{timestamp}"  # Final _{timestamp} is handled before.
            )

    # Default return without overrides or if all overrides were filtered out
    return f"{username.upper()}_{config_name}{timestamp}"  # Final _{timestamp} is handled before in the function call


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load a configuration from a Python file.

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

    spec = importlib.util.spec_from_file_location(module_path, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the get_config function
    if not hasattr(module, "get_config"):
        raise AttributeError(f"Configuration file {config_path} must have a get_config() function")

    # Call the get_config function to get the configuration
    return module.get_config()


def apply_config_overrides(config: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    """Apply command-line overrides to a configuration.

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
    new_config = dict_to_dataclass(resolved_conf, type(config))
    return new_config


def verify_no_interpolator_overwrites(config: ExperimentConfig, overrides: List[str]) -> None:
    """Prevent overriding fields that are defined as OmegaConf interpolations (e.g., "${...}").

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


def config_to_dict_for_rich(config: Any) -> Any:
    """Recursively convert a dataclass or LazyConfig object to a dictionary for rich printing."""
    if dataclasses.is_dataclass(config):
        # Convert dataclass to a dictionary
        result = {}
        for f in dataclasses.fields(config):
            value = getattr(config, f.name)
            result[f.name] = config_to_dict_for_rich(value)
        return result
    elif isinstance(config, LazyConfig):
        # For LazyConfig, create a dictionary with its target and keyword arguments
        # and recursively process the kwargs
        kwargs = {k: config_to_dict_for_rich(v) for k, v in config.kwargs.items()}
        return {
            "__target__": config.target,
            **kwargs,
        }
    elif isinstance(config, list):
        # Recursively process lists
        return [config_to_dict_for_rich(v) for v in config]
    elif isinstance(config, dict):
        # Recursively process dictionaries
        return {k: config_to_dict_for_rich(v) for k, v in config.items()}
    else:
        # Return the value as is if it's not a dataclass, LazyConfig, list, or dict
        return config


def add_to_tree(tree: Tree, data: Any, key: str = ""):
    """Add a key-value pair to a rich tree. Used for rich printing of the configuration."""
    # Get a beautiful string representation of the value
    if hasattr(data, "__name__"):
        str_data = data.__name__
    else:
        str_data = str(data)

    # If the data is a dictionary or a list, create a new branch.
    if isinstance(data, dict):
        branch = tree.add(f"[bold]{key}[/bold]")
        for key, value in data.items():
            add_to_tree(branch, value, key)
    elif isinstance(data, list):
        branch = tree.add(f"[bold]{key}[/bold]")
        for i, value in enumerate(data):
            add_to_tree(branch, value, f"item {i}")
    else:
        # If the data is a simple type, display it as a key-value pair.
        tree.add(f"[bold]{key}[/bold]: [green]{str_data}[/green]")
