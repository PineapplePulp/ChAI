import torch
from PIL import Image
import PIL

import cv2
import numpy as np

def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), PIL.Image.Resampling.LANCZOS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), PIL.Image.Resampling.LANCZOS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def bgr_to_tensor(
    frame_bgr: np.ndarray,
    *,
    normalize: bool = False,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    add_batch: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Convert an OpenCV BGR frame (H x W x 3, uint8) to a PyTorch tensor
    (C x H x W, float32, RGB).  Optionally normalise with mean / std.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Raw image from cv2 (BGR, uint8, H x W x 3).
    normalize : bool, default False
        If True, apply  (tensor - mean) / std  after scaling to [0,1].
    mean, std : tuple of 3 floats, optional
        Normalisation stats in **RGB** order.  If `normalize` is True and
        these are omitted, ImageNet values are used.
    add_batch : bool, default False
        If True, adds a leading batch dim → (1, C, H, W).
    device : torch.device | str | None
        Target device (e.g. "cuda").  If None, tensor stays on CPU.

    Returns
    -------
    torch.Tensor
        The converted (and optionally normalised) tensor.
    """
    # --- 1. BGR → RGB -------------------------------------------------------
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # --- 2. Ensure contiguous memory for zero‑copy conversion --------------
    frame_rgb = np.ascontiguousarray(frame_rgb)

    # --- 3. numpy → torch, reorder to C,H,W --------------------------------
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()  # C,H,W

    # --- 4. Scale to [0,1] --------------------------------------------------
    tensor = tensor.div_(255.0)

    # --- 5. Optional normalisation -----------------------------------------
    if normalize:
        if mean is None or std is None:
            mean = (0.485, 0.456, 0.406)  # ImageNet defaults (RGB)
            std  = (0.229, 0.224, 0.225)
        mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)[:, None, None]
        std  = torch.tensor(std,  dtype=tensor.dtype, device=tensor.device)[:, None, None]
        tensor.sub_(mean).div_(std)

    # --- 6. Optional batch and device move ---------------------------------
    if add_batch:
        tensor = tensor.unsqueeze(0)      # N,C,H,W
    if device is not None:
        tensor = tensor.to(device, non_blocking=True)

    return tensor


def tensor_to_bgr(frame_tensor, *, undo_normalise=False, mean=None, std=None):
    """
    Args
    ----
    frame_tensor : torch.Tensor
        (C,H,W) or (1,C,H,W)   ―  float or half   ―  RGB
    undo_normalise : bool
        True if you previously applied (x - mean) / std
    mean, std : list/tuple of 3 floats
        Same numbers you used for normalising (e.g. ImageNet)
    Returns
    -------
    frame_bgr : np.ndarray   (H,W,3) uint8   BGR  contiguous
    """
    # 1) squeeze batch dimension if present
    if frame_tensor.ndim == 4:
        frame_tensor = frame_tensor[0]

    # 2) move to CPU & float32 for math
    img = frame_tensor.detach()

    # 3) (optional) reverse mean/std normalisation
    if undo_normalise:
        if mean is None or std is None:
            raise ValueError("Supply mean and std to undo normalisation")
        mean = torch.tensor(mean).to(img).view(3,1,1)
        std  = torch.tensor(std).to(img).view(3,1,1)
        img = img * std + mean

    # 4) scale back to 0‑255, clamp, uint8
    img = (img * 255.0)
    # img = img # .to(torch.float16)
    img = img.clamp(0,255).byte()

    # 5) channel‑last & numpy
    img = img.permute(1,2,0).cpu().numpy()                 # H,W,C  RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # → BGR
    img = np.ascontiguousarray(img)                  # ensure OpenCV‑happy
    return img
