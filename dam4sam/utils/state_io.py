import os
from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
import numpy as np


def _move_to_cpu(obj: Any, memo: Dict[int, Any] | None = None) -> Any:
    """Recursively move all torch.Tensors in a nested structure to CPU and detach.

    Also normalizes torch.device entries to CPU to avoid device mismatches in comparisons.

    Supported containers: dict, list, tuple, set, OrderedDict. Numpy arrays are left as-is.
    """
    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    # Tensors
    if isinstance(obj, torch.Tensor):
        cpu_tensor = obj.detach().to("cpu")
        memo[obj_id] = cpu_tensor
        return cpu_tensor

    # torch.device
    if isinstance(obj, torch.device):
        cpu_dev = torch.device("cpu")
        memo[obj_id] = cpu_dev
        return cpu_dev

    # Numpy arrays - already CPU resident
    if isinstance(obj, np.ndarray):
        memo[obj_id] = obj
        return obj

    # Dict-like
    if isinstance(obj, dict):
        # Preserve OrderedDict if provided
        new_dict = OrderedDict() if isinstance(obj, OrderedDict) else {}
        memo[obj_id] = new_dict
        for k, v in obj.items():
            # Keys are assumed to be simple/immutable
            new_dict[k] = _move_to_cpu(v, memo)
        return new_dict

    # List
    if isinstance(obj, list):
        new_list = []
        memo[obj_id] = new_list
        new_list.extend(_move_to_cpu(v, memo) for v in obj)
        return new_list

    # Tuple
    if isinstance(obj, tuple):
        # Construct tuple from converted elements
        new_tuple = tuple(_move_to_cpu(v, memo) for v in obj)
        memo[obj_id] = new_tuple
        return new_tuple

    # Set
    if isinstance(obj, set):
        new_set = set(_move_to_cpu(v, memo) for v in obj)
        memo[obj_id] = new_set
        return new_set

    # Everything else: leave as-is (int, float, str, None, bool, custom objects)
    memo[obj_id] = obj
    return obj


def state_to_cpu_copy(inference_state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied CPU version of an inference_state dict.

    All tensors are detached and moved to CPU; torch.device entries are set to CPU.
    """
    return _move_to_cpu(inference_state)


def save_inference_state(inference_state: Dict[str, Any], target_path: str) -> None:
    """Save an inference state to disk after moving all tensors to CPU.

    - Creates parent directories if necessary
    - Uses torch.save for efficient serialization of tensors and nested containers
    """
    cpu_state = state_to_cpu_copy(inference_state)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    # Use torch.save (pickle) which handles complex Python containers
    torch.save(cpu_state, target_path)


def load_inference_state(source_path: str) -> Dict[str, Any]:
    """Load a previously saved inference state (mapped to CPU)."""
    return torch.load(source_path, map_location="cpu")


class DiffResult:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.num_differences: int = 0

    def add(self, line: str) -> None:
        self.lines.append(line)
        self.num_differences += 1

    def extend(self, lines: list[str]) -> None:
        self.lines.extend(lines)

    def __str__(self) -> str:
        return "\n".join(self.lines)


def _compare_tensors(a: torch.Tensor, b: torch.Tensor, path: str, atol: float, rtol: float) -> list[str]:
    msgs: list[str] = []
    if a.shape != b.shape:
        msgs.append(f"{path}: tensor shape differs {tuple(a.shape)} vs {tuple(b.shape)}")
        return msgs
    if a.dtype != b.dtype:
        msgs.append(f"{path}: tensor dtype differs {a.dtype} vs {b.dtype}")
    diff = torch.isclose(a, b, atol=atol, rtol=rtol)
    if not bool(diff.all()):
        abs_diff = (a - b).abs()
        max_abs = float(abs_diff.max().item())
        num_bad = int((~diff).sum().item())
        msgs.append(f"{path}: tensor values differ (max_abs={max_abs:.6g}, num_bad={num_bad}, atol={atol}, rtol={rtol})")
    return msgs


def _compare_ndarrays(a: np.ndarray, b: np.ndarray, path: str, atol: float, rtol: float) -> list[str]:
    msgs: list[str] = []
    if a.shape != b.shape:
        msgs.append(f"{path}: ndarray shape differs {a.shape} vs {b.shape}")
        return msgs
    if a.dtype != b.dtype:
        msgs.append(f"{path}: ndarray dtype differs {a.dtype} vs {b.dtype}")
    close = np.isclose(a, b, atol=atol, rtol=rtol)
    if not np.all(close):
        abs_diff = np.abs(a - b)
        max_abs = float(abs_diff.max())
        num_bad = int(np.size(close) - int(np.sum(close)))
        msgs.append(f"{path}: ndarray values differ (max_abs={max_abs:.6g}, num_bad={num_bad}, atol={atol}, rtol={rtol})")
    return msgs


def _compare_scalars(a: Any, b: Any, path: str, atol: float, rtol: float) -> list[str]:
    msgs: list[str] = []
    if isinstance(a, float) and isinstance(b, float):
        if not (abs(a - b) <= atol + rtol * max(abs(a), abs(b))):
            msgs.append(f"{path}: float values differ {a} vs {b} (atol={atol}, rtol={rtol})")
    else:
        if a != b:
            msgs.append(f"{path}: values differ {a!r} vs {b!r}")
    return msgs


def _compare_any(a: Any, b: Any, path: str, atol: float, rtol: float, visited: set[Tuple[int, int]]) -> list[str]:
    key = (id(a), id(b))
    if key in visited:
        return []
    visited.add(key)

    # Types must match for structural equality
    if type(a) is not type(b):
        return [f"{path}: type differs {type(a).__name__} vs {type(b).__name__}"]

    # Tensors
    if isinstance(a, torch.Tensor):
        return _compare_tensors(a, b, path, atol, rtol)

    # Numpy arrays
    if isinstance(a, np.ndarray):
        return _compare_ndarrays(a, b, path, atol, rtol)

    # Dict-like
    if isinstance(a, dict):
        msgs: list[str] = []
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        for k in sorted(a_keys - b_keys):
            msgs.append(f"{path}.{k}: missing in right")
        for k in sorted(b_keys - a_keys):
            msgs.append(f"{path}.{k}: missing in left")
        for k in sorted(a_keys & b_keys, key=lambda x: str(x)):
            msgs.extend(_compare_any(a[k], b[k], f"{path}.{k}" if path else str(k), atol, rtol, visited))
        return msgs

    # List
    if isinstance(a, list):
        msgs: list[str] = []
        if len(a) != len(b):
            msgs.append(f"{path}: list length differs {len(a)} vs {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            msgs.extend(_compare_any(ai, bi, f"{path}[{i}]", atol, rtol, visited))
        return msgs

    # Tuple
    if isinstance(a, tuple):
        msgs: list[str] = []
        if len(a) != len(b):
            msgs.append(f"{path}: tuple length differs {len(a)} vs {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            msgs.extend(_compare_any(ai, bi, f"{path}[{i}]", atol, rtol, visited))
        return msgs

    # Set
    if isinstance(a, set):
        if a != b:
            return [f"{path}: set differs (len {len(a)} vs {len(b)})"]
        return []

    # torch.device
    if isinstance(a, torch.device):
        return _compare_scalars(str(a), str(b), path, atol, rtol)

    # Scalars/others
    return _compare_scalars(a, b, path, atol, rtol)


def diff_inference_states(left: Dict[str, Any], right: Dict[str, Any], atol: float = 0.0, rtol: float = 0.0) -> DiffResult:
    """Compute a human-readable diff between two inference state dicts."""
    res = DiffResult()
    msgs = _compare_any(left, right, path="", atol=atol, rtol=rtol, visited=set())
    res.extend(msgs)
    res.num_differences = len(msgs)
    return res


