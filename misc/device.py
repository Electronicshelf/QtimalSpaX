import torch


def _mps_available():
    if not hasattr(torch.backends, "mps"):
        return False
    is_built = getattr(torch.backends.mps, "is_built", lambda: False)()
    is_available = torch.backends.mps.is_available()
    return is_built and is_available


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device):
    """
    Normalize device selection for cpu/cuda/mps.
    Accepts a torch.device or a string (cpu/cuda/mps/auto).
    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        return get_default_device()
    device_str = str(device).lower()
    if device_str in ("auto", "default"):
        return get_default_device()
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "mps":
        return torch.device("mps") if _mps_available() else torch.device("cpu")
    return torch.device(device_str)
