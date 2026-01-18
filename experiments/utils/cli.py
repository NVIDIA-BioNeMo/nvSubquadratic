# TODO: Add license header here


"""Utility functions for the MNIST classification experiment."""

import dataclasses
import datetime
import getpass
import importlib.util
import re
from pathlib import Path
from typing import Any

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


def get_deterministic_run_name(
    config_path: str, overrides: list[str] | None = None, use_timestamp: bool = True
) -> str:
    """Generate a deterministic run name based on the config file name, current timestamp, and any overrides.

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
                # Truncate float values to 4 decimal places
                try:
                    float_val = float(value)
                    # Check if it's actually a float (not an int)
                    if "." in str(value) or "e" in str(value).lower():
                        value = f"{float_val:.4g}"
                except (ValueError, TypeError):
                    pass
                processed_overrides.append(f"{short_key}={value}")
            filtered_overrides = processed_overrides
            # Sort overrides for deterministic ordering
            filtered_overrides.sort()
            # Create a string representation of the overrides
            override_str = "_".join(filtered_overrides).replace("=", "_")
            run_name = (
                f"{username.upper()}_{config_name}_{override_str}{timestamp}"  # Final _{timestamp} is handled before.
            )
            # Limit total run name length to avoid OSError: File name too long
            # Leave room for timestamp (~20 chars) if it's included
            max_base_len = 180 if use_timestamp else 200
            if len(run_name) > max_base_len + len(timestamp):
                run_name = run_name[: max_base_len - len(timestamp)] + timestamp
            return run_name

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


def apply_config_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
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


def verify_no_interpolator_overwrites(config: ExperimentConfig, overrides: list[str]) -> None:
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


def _serialize_config_value(config: Any) -> Any:
    """Recursively convert a config to a JSON-serializable dictionary.

    Handles:
    - Dataclasses
    - OmegaConf DictConfig objects (with unresolved interpolations preserved as strings)
    - LazyConfig objects
    - Function/callable references (converted to qualified name strings)
    - Object instances (converted to class name strings)
    - Lists and dicts (recursively processed)

    Used by config_to_dict for both WandB logging and rich tree printing.
    """
    # Import here to avoid circular imports
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if config is None:
        return None

    # Handle dataclasses
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        result = {}
        for f in dataclasses.fields(config):
            value = getattr(config, f.name)
            result[f.name] = _serialize_config_value(value)
        return result

    # Handle OmegaConf DictConfig - convert to plain dict with resolved values
    if isinstance(config, DictConfig):
        # Convert to plain dict, resolving any remaining interpolations
        # (config should already be resolved by apply_config_overrides)
        raw_dict = OmegaConf.to_container(config, resolve=True)
        # Recursively process to handle nested non-serializable objects (functions, etc.)
        return _serialize_config_value(raw_dict)

    # Handle OmegaConf ListConfig
    if isinstance(config, ListConfig):
        raw_list = OmegaConf.to_container(config, resolve=True)
        return _serialize_config_value(raw_list)

    # Handle LazyConfig objects
    if isinstance(config, LazyConfig):
        kwargs = {k: _serialize_config_value(v) for k, v in config.kwargs.items()}
        return {"__target__": config.target, **kwargs}

    # Handle regular lists
    if isinstance(config, list):
        return [_serialize_config_value(v) for v in config]

    # Handle regular dicts
    if isinstance(config, dict):
        return {k: _serialize_config_value(v) for k, v in config.items()}

    # Handle functions and callables - convert to qualified name
    if callable(config) and hasattr(config, "__module__") and hasattr(config, "__name__"):
        return f"{config.__module__}.{config.__name__}"

    # Handle class types
    if isinstance(config, type):
        return f"{config.__module__}.{config.__name__}"

    # Handle object instances that aren't basic types
    if hasattr(config, "__class__") and not isinstance(config, (int, float, str, bool, type(None))):
        cls = config.__class__
        # For simple objects, just return class name
        return f"{cls.__module__}.{cls.__name__}"

    # Return basic types as-is (int, float, str, bool, None)
    return config


def config_to_dict(config: Any) -> Any:
    """Recursively convert a config to a JSON-serializable dictionary.

    Handles:
    - Dataclasses
    - OmegaConf DictConfig/ListConfig objects
    - LazyConfig objects
    - Function/callable references (converted to qualified name strings)
    - Object instances (converted to class name strings)
    - Lists and dicts (recursively processed)

    Used for both WandB logging and rich tree printing.
    """
    return _serialize_config_value(config)


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
