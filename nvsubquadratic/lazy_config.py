# TODO: Add license header here
# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Lazy configuration system for deferred object instantiation.

This module provides a lightweight alternative to Hydra/detectron2-style lazy
configs, designed to let the entire network architecture be specified as a
plain Python or OmegaConf config dict that can be serialised, diffed, and
instantiated on demand — without importing heavy framework dependencies.

**Core concept**

A :class:`LazyConfig` holds a reference to a target class or callable together
with its keyword arguments.  Calling the object produces an
:class:`~omegaconf.DictConfig` with a ``__target__`` key:

.. code-block:: python

    cfg = LazyConfig(torch.nn.LayerNorm)(normalized_shape=768, eps=1e-6)
    # cfg == DictConfig({"__target__": "torch.nn.LayerNorm",
    #                     "normalized_shape": 768, "eps": 1e-6})

    norm = instantiate(cfg)      # → LayerNorm(768, eps=1e-6)

:func:`instantiate` resolves ``__target__`` via :func:`importlib.import_module`,
merges any ``**kwargs`` overrides (useful for injecting per-block arguments like
``drop_path_rate``), recursively instantiates nested configs, and evaluates
simple arithmetic strings (e.g. ``"160 * 3"`` → ``480``) found in values.

**Placeholder values**

:data:`PLACEHOLDER` is a singleton sentinel that marks fields whose values are
not yet known at config-construction time (e.g. a hidden dimension resolved
later by OmegaConf interpolation).  :func:`_contains_placeholder` prevents
premature instantiation of configs that still hold placeholders.

**Config I/O**

:func:`save_config` / :func:`load_config` serialise configs to YAML via
OmegaConf.  :func:`to_config` reverse-engineers a ``__target__`` dict from an
already-instantiated object (best-effort; works for simple cases).

Adapted from https://github.com/implicit-long-convs/ccnn_v2.
"""

import ast
import copy
import importlib
import inspect
from typing import Any, Callable, Dict, Type, Union

import torch
from omegaconf import DictConfig, OmegaConf


class _Placeholder:
    """Sentinel object for unset config fields.

    Using a dedicated class instead of ``None`` so that legitimate ``None`` values in configs
    are not confused with missing fields.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "PLACEHOLDER"

    def __bool__(self):
        return False


PLACEHOLDER = _Placeholder()


class LazyConfig:
    """Deferred-instantiation config builder.

    A :class:`LazyConfig` stores a reference to a target class or callable.
    Calling the instance with keyword arguments produces an OmegaConf
    :class:`~omegaconf.DictConfig` that encodes the target and its arguments
    under the ``__target__`` key — the standard format consumed by
    :func:`instantiate`.

    **Two-step usage**::

        # Step 1: declare the config (no import of the target needed at this point)
        cfg = LazyConfig(torch.nn.Dropout)(p=0.5, inplace=True)

        # Step 2: create the object when ready
        dropout = instantiate(cfg)
        isinstance(dropout, torch.nn.Dropout)  # True

    **Nesting**

    LazyConfigs can be nested: passing a ``LazyConfig`` result as a keyword
    argument to another ``LazyConfig`` call is supported.  :func:`instantiate`
    resolves nested configs recursively when ``recursive_instantiate=True``.

    **String targets**

    The ``target`` may be a dotted-path string (e.g.
    ``"torch.nn.LayerNorm"``), a class, or any callable that has
    ``__module__`` and ``__name__`` attributes.

    Args:
        target: The class, callable, or fully-qualified dotted string that
            will be used as ``__target__`` in the resulting config dict.

    Example::

        cfg = LazyConfig(torch.nn.Dropout)(p=0.5, inplace=True)
        module = instantiate(cfg)
        isinstance(module, torch.nn.Dropout)  # True
    """

    def __init__(self, target: Union[Type, Callable, str]):
        """Store the target class or callable reference.

        Args:
            target: A class, callable, or fully-qualified dotted-path string.
        """
        self.target = target

    def __call__(self, **kwargs) -> Union[Dict[str, Any], DictConfig]:
        """Produce a config dict containing ``__target__`` and all keyword args.

        Converts ``target`` to a dotted string (``module.ClassName``) and
        creates an OmegaConf :class:`~omegaconf.DictConfig` so the result
        supports attribute access and OmegaConf interpolation.

        Args:
            **kwargs: Arguments to forward to the target at instantiation time.
                Values may themselves be :class:`LazyConfig` results (nested
                configs), plain scalars, or any OmegaConf-compatible type.

        Returns:
            OmegaConf DictConfig with ``__target__`` set to the resolved
            dotted-path string and one key per kwarg.  Falls back to a plain
            ``dict`` if OmegaConf validation fails (a warning is printed).
        """
        # Convert target to a string if it's a class or function
        if isinstance(self.target, str):
            target_str = self.target
        else:
            if hasattr(self.target, "__module__") and hasattr(self.target, "__name__"):
                target_str = f"{self.target.__module__}.{self.target.__name__}"
            else:
                raise ValueError(f"Target {self.target} is not a valid class or function")

        # Create config dict
        config_dict = {"__target__": target_str}
        config_dict.update(kwargs)

        # Convert to OmegaConf for dot notation access
        try:
            # Create as OmegaConf with permissive validation by default
            return OmegaConf.create(config_dict, flags={"allow_objects": True})
        except Exception as e:
            # If that fails, log the error and return plain dict as last resort
            print(f"Warning: OmegaConf validation failed for {target_str}, using plain dict: {str(e)}")
            return config_dict


def _resolve_target(target_str: str) -> Callable:
    """Resolve a string reference to a class or function.

    Args:
        target_str: String reference to a class or function

    Returns:
        The resolved class or function
    """
    module_path, target_name = target_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    target = getattr(module, target_name)
    return target


def _is_module_class(obj: Any) -> bool:
    """Return True if obj is a class and a subclass of torch.nn.Module."""
    try:
        return inspect.isclass(obj) and issubclass(obj, torch.nn.Module)
    except Exception:
        return False


def _to_dict_with_target(config: DictConfig) -> Dict[str, Any]:
    """Convert an OmegaConf DictConfig to a dictionary while preserving __target__."""
    if not isinstance(config, DictConfig):
        return config

    result = {}
    for k, v in config.items():
        result[k] = v

    return result


def _contains_placeholder(obj: Any) -> bool:
    """Check if a dictionary, list, or value contains any PLACEHOLDER values.

    Args:
        obj: The object to check

    Returns:
        True if obj or any of its nested elements contains a PLACEHOLDER, False otherwise
    """
    if obj is PLACEHOLDER:
        return True
    elif isinstance(obj, dict) or isinstance(obj, DictConfig):
        return any(_contains_placeholder(v) for v in obj.values())
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return any(_contains_placeholder(item) for item in obj)
    return False


def _is_arithmetic_string(value: str) -> bool:
    """Return True if the string looks like a pure arithmetic expression (no letters)."""
    if not isinstance(value, str):
        return False
    # Quick reject if any alphabetic characters or braces remain
    if any(c.isalpha() for c in value) or "${" in value or "}" in value:
        return False
    # Allow digits, whitespace, and arithmetic symbols
    allowed = set("0123456789.+-*/()% ")
    return set(value) <= allowed


def _safe_eval_arith(expr: str) -> Any:
    """Safely evaluate a basic arithmetic expression string to int/float.

    Supports +, -, *, /, //, %, parentheses. No names or function calls.
    """
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):  # Py>=3.8
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Invalid constant in arithmetic expression")
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.FloorDiv):
                return left // right
            if isinstance(n.op, ast.Mod):
                return left % right
            raise ValueError("Unsupported operator in arithmetic expression")
        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator in arithmetic expression")
        if isinstance(n, ast.Tuple):  # Not allowed
            raise ValueError("Tuples not allowed in arithmetic expression")
        raise ValueError("Invalid node in arithmetic expression")

    result = _eval(node)
    # Cast to int if it's an exact integer
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def _eval_arith_in_obj(obj: Any) -> Any:
    """Recursively evaluate arithmetic strings within dicts/lists."""
    if isinstance(obj, dict):
        return {k: _eval_arith_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_eval_arith_in_obj(v) for v in obj]
    if isinstance(obj, DictConfig):
        # Convert to plain container, process, then back
        plain = OmegaConf.to_container(obj, resolve=False)
        processed = _eval_arith_in_obj(plain)
        return OmegaConf.create(processed, flags={"allow_objects": True})
    if isinstance(obj, str) and _is_arithmetic_string(obj):
        try:
            return _safe_eval_arith(obj)
        except Exception:
            return obj
    return obj


def instantiate(
    config: Union[Dict[str, Any], DictConfig, "LazyConfig"],
    *,
    recursive_instantiate: bool = False,
    **kwargs,
) -> Any:
    """Instantiate an object from a :class:`LazyConfig` or ``__target__`` dict.

    **Resolution pipeline**

    1. If ``config`` is a :class:`LazyConfig`, call it with any ``**kwargs``
       to produce a :class:`~omegaconf.DictConfig`.
    2. If the config has no ``__target__`` key, recursively instantiate each
       value and return a :class:`~omegaconf.DictConfig` (useful for nested
       non-lazy sub-configs).
    3. Resolve ``__target__`` to the actual class/callable via
       :func:`importlib.import_module`.
    4. Deep-copy, resolve OmegaConf interpolations, merge ``**kwargs``
       overrides, recursively instantiate nested configs, and evaluate
       arithmetic strings (e.g. ``"128 * 4"`` → ``512``).
    5. Call ``target(**processed_args)`` and return the result.

    **Placeholder handling**

    Nested configs that still contain :data:`PLACEHOLDER` values are left
    as config dicts rather than instantiated.

    Args:
        config: A :class:`LazyConfig` instance, an OmegaConf
            :class:`~omegaconf.DictConfig`, or a plain ``dict`` containing at
            least a ``"__target__"`` key.
        recursive_instantiate: If ``True``, recursively instantiate all nested
            configs found in argument values.  If ``False`` (default), nested
            configs are left as :class:`~omegaconf.DictConfig` objects.
        **kwargs: Extra keyword arguments merged into (and overriding) the
            config's stored arguments before instantiation.  Commonly used to
            inject per-block arguments such as ``drop_path_rate``.

    Returns:
        The instantiated object (result of calling the resolved ``__target__``
        with the processed arguments).

    Raises:
        TypeError: If ``target(**processed_args)`` fails; re-raised with the
            target's signature in the error message for easier debugging.

    Example::

        cfg = LazyConfig(torch.nn.LayerNorm)(normalized_shape=768)
        norm = instantiate(cfg, eps=1e-5)   # override default eps
    """
    _cfg = copy.deepcopy(config)
    # If it's a LazyConfig object, convert it to a config dict first
    if isinstance(_cfg, LazyConfig):
        # Call the LazyConfig to get a config dict with __target__
        _cfg = _cfg(**kwargs)
        # Clear kwargs since they're now in the config
        kwargs = {}

    if not isinstance(_cfg, (dict, DictConfig)):
        # If it's not a config dict, return as is
        return _cfg

    # Ensure config is an OmegaConf object for dot notation access
    if not isinstance(_cfg, DictConfig):
        _cfg = OmegaConf.create(_cfg, flags={"allow_objects": True})

    # Check if it's a lazy config (has __target__)
    if "__target__" not in _cfg:
        # For non-lazy configs, preserve dot notation by returning as OmegaConf
        # while also processing nested dictionaries recursively
        processed_dict = {key: instantiate(value) for key, value in _cfg.items()}
        return OmegaConf.create(processed_dict, flags={"allow_objects": True})

    # Get the target and arguments
    target_str = _cfg.get("__target__")
    # Create a clean copy of args without modifying the original config
    args = {}
    for k, v in _cfg.items():
        if k != "__target__":
            args[k] = v

    # Override with additional kwargs if provided
    args.update(kwargs)

    # Process nested configurations
    processed_args = {}
    for key, value in args.items():
        if isinstance(value, LazyConfig):
            # Nested LazyConfig objects
            if recursive_instantiate:
                processed_args[key] = instantiate(value, recursive_instantiate=recursive_instantiate)
            else:
                # Decide whether to defer or instantiate based on target type
                target = value.target if not isinstance(value.target, str) else _resolve_target(value.target)
                if not _is_module_class(target):
                    # For non-module callables (e.g., init factories), instantiate now
                    processed_args[key] = instantiate(value, recursive_instantiate=recursive_instantiate)
                else:
                    # Pass through as config (DictConfig) without constructing the module
                    processed_args[key] = value()
        elif (isinstance(value, dict) or isinstance(value, DictConfig)) and "__target__" in value:
            # If the nested config contains placeholders, don't instantiate it yet
            if _contains_placeholder(value):
                # Keep as OmegaConf to maintain dot notation access
                if not isinstance(value, DictConfig):
                    processed_args[key] = OmegaConf.create(value, flags={"allow_objects": True})
                else:
                    processed_args[key] = value
            else:
                # If no placeholders, instantiate based on target type and recursion flag
                target = _resolve_target(value.get("__target__"))
                if recursive_instantiate or not _is_module_class(target):
                    processed_args[key] = instantiate(value, recursive_instantiate=recursive_instantiate)
                else:
                    # Leave as config (DictConfig), do not instantiate into a module
                    if not isinstance(value, DictConfig):
                        processed_args[key] = OmegaConf.create(value, flags={"allow_objects": True})
                    else:
                        processed_args[key] = value
        elif isinstance(value, dict) or isinstance(value, DictConfig):
            # For nested dicts without __target__, maintain dot notation access but don't instantiate
            # if they contain placeholders
            if _contains_placeholder(value):
                processed_args[key] = OmegaConf.create(value, flags={"allow_objects": True})
            else:
                if recursive_instantiate:
                    processed_args[key] = instantiate(value, recursive_instantiate=recursive_instantiate)
                else:
                    # Keep as config container for later resolution/instantiation
                    if not isinstance(value, DictConfig):
                        processed_args[key] = OmegaConf.create(value, flags={"allow_objects": True})
                    else:
                        processed_args[key] = value
        elif isinstance(value, list):
            # Handle lists of configs
            if recursive_instantiate:
                # Instantiate the list recursively if requested
                processed_args[key] = [
                    (
                        instantiate(item, recursive_instantiate=recursive_instantiate)
                        if isinstance(item, (dict, DictConfig, LazyConfig)) and not _contains_placeholder(item)
                        else item
                    )
                    for item in value
                ]
            else:
                # Keep list items as configs where applicable
                new_list = []
                for item in value:
                    if isinstance(item, LazyConfig):
                        new_list.append(item())
                    elif (isinstance(item, dict) or isinstance(item, DictConfig)) and "__target__" in item:
                        if not isinstance(item, DictConfig):
                            new_list.append(OmegaConf.create(item, flags={"allow_objects": True}))
                        else:
                            new_list.append(item)
                    else:
                        new_list.append(item)
                processed_args[key] = new_list
        else:
            processed_args[key] = value

    # Evaluate simple arithmetic strings in processed_args (e.g., "160 * 3" -> 480)
    processed_args = _eval_arith_in_obj(processed_args)

    # Resolve the target class/function
    target = _resolve_target(target_str)

    # Instantiate the object
    try:
        return target(**processed_args)
    except TypeError as e:
        # Get the signature of the target for better error message
        sig = inspect.signature(target)
        raise TypeError(
            f"Error instantiating {target_str} with arguments {processed_args}.\n"
            f"Target signature: {sig}\n"
            f"Original error: {str(e)}"
        ) from e


def to_config(obj: Any) -> Dict[str, Any]:
    """Convert an instantiated object to a LazyConfig-compatible dictionary.

    Args:
        obj: The object to convert

    Returns:
        A dictionary with __target__ and arguments
    """
    if not hasattr(obj, "__class__"):
        return obj

    # Get the class
    cls = obj.__class__
    target_str = f"{cls.__module__}.{cls.__name__}"

    # Get init parameters
    if hasattr(cls, "__init__"):
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        # Remove 'self'
        if params and params[0] == "self":
            params = params[1:]
    else:
        params = []

    # Build config dict
    config = {"__target__": target_str}

    # Add params if they exist as attributes
    for param in params:
        if hasattr(obj, param):
            value = getattr(obj, param)
            # Recursively convert nested objects
            if hasattr(value, "__class__") and not isinstance(value, (int, float, str, bool, list, tuple, dict)):
                config[param] = to_config(value)
            else:
                config[param] = value

    return config


def save_config(config: Dict[str, Any], filename: str) -> None:
    """Save a configuration to a file.

    Args:
        config: Configuration dictionary
        filename: File to save to
    """
    # Convert to OmegaConf
    conf = OmegaConf.create(config, flags={"allow_objects": True})
    with open(filename, "w") as f:
        f.write(OmegaConf.to_yaml(conf))


def load_config(filename: str) -> Dict[str, Any]:
    """Load a configuration from a file.

    Args:
        filename: File to load from

    Returns:
        The loaded configuration
    """
    # Load with OmegaConf
    conf = OmegaConf.load(filename)
    # Return as dict to be safe
    return OmegaConf.to_container(conf, resolve=True)
