import torch

def _find_tensors_recursive(obj, path=""):
    """Helper function to recursively find all tensors in nested structure"""
    tensors = []
    if isinstance(obj, torch.Tensor):
        tensors.append((path, obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            tensors.extend(_find_tensors_recursive(value, new_path))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            tensors.extend(_find_tensors_recursive(item, new_path))
    return tensors


def _unbatch_tensor_recursive(obj, batch_idx):
    """Helper function to recursively unbatch tensors in nested structure"""
    if isinstance(obj, torch.Tensor):
        return obj[batch_idx:batch_idx + 1]
    elif isinstance(obj, dict):
        return {key: _unbatch_tensor_recursive(value, batch_idx) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_unbatch_tensor_recursive(item, batch_idx) for item in obj]
    else:
        return obj


def unbatch_dict(batched_dict: dict) -> list[dict]:
    """
    Split a batched dictionary into a list of single-batch dictionaries.
    Each tensor is split along the first dimension (batch dimension).
    
    Args:
        batched_dict: Dictionary containing batched tensors
        
    Returns:
        List of dictionaries, each with batch size 1
        
    Raises:
        ValueError: If tensors have different batch sizes
    """
    # Find all tensors recursively
    tensors = _find_tensors_recursive(batched_dict)

    if not tensors:
        raise ValueError("No tensors found in dictionary")

    # Determine batch size and validate consistency
    batch_size = None
    for path, tensor in tensors:
        if batch_size is None:
            batch_size = tensor.shape[0]
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"Tensor {path} has batch size {tensor.shape[0]}, expected {batch_size}")

    # Create list of unbatched dictionaries
    unbatched_list = []

    for batch_idx in range(batch_size):
        unbatched_dict = _unbatch_tensor_recursive(batched_dict, batch_idx)
        unbatched_list.append(unbatched_dict)

    return unbatched_list
