def resolve_device(preference: str) -> str:
    """Pick cuda, then mps, then cpu when preference is auto."""
    if preference != "auto":
        return preference
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _torch():
    import torch

    return torch
