"""Standalone CPU sliding-window inference; only dependency: PyTorch.

NCHW float32/float64 input and same-resolution raw logits. See docs/tiled.md.
Generated copies specialize the minimum tile side and channel counts.
"""
import itertools

import torch

MIN_TILE_SIZE = 1
INPUT_CHANNELS = None
OUTPUT_CHANNELS = None


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _pair(value, name, minimum):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a (height, width) pair")
    return tuple(_integer(v, name, minimum) for v in value)


def tile_plan(shape, *, tile_size=(256, 256), overlap=(32, 32), tile_batch_size=1,
              blend="constant", max_image_pixels=16_777_216, max_tiles=65_536,
              max_tile_batch_size=64, max_tile_pixels=1_048_576):
    """Validate before inference/allocation; return a deterministic grid plan.

    Pixel limit counts N*H*W, tile limit includes all N images. Limits are
    caller-controlled workload guards, not byte limits or a process-memory cap.
    Starts are multiples of tile_size-overlap; stop once the image is covered.
    """
    if not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 4:
        raise ValueError("shape must be positive NCHW")
    n, c, h, w = (_integer(v, "shape") for v in shape)
    if INPUT_CHANNELS is not None and c != INPUT_CHANNELS:
        raise ValueError(f"expected {INPUT_CHANNELS} input channels")
    th, tw = _pair(tile_size, "tile_size", MIN_TILE_SIZE)
    oh, ow = _pair(overlap, "overlap", 0)
    if oh >= th or ow >= tw:
        raise ValueError("overlap must be smaller than tile_size on each axis")
    if blend not in ("constant", "gaussian"):
        raise ValueError("blend must be constant or gaussian")
    for name, value in (("max_image_pixels", max_image_pixels), ("max_tiles", max_tiles),
                        ("max_tile_batch_size", max_tile_batch_size),
                        ("max_tile_pixels", max_tile_pixels), ("tile_batch_size", tile_batch_size)):
        _integer(value, name)
    if n * h * w > max_image_pixels:
        raise ValueError("max_image_pixels exceeded (N*H*W)")
    if th * tw > max_tile_pixels:
        raise ValueError("max_tile_pixels exceeded")
    if tile_batch_size > max_tile_batch_size:
        raise ValueError("max_tile_batch_size exceeded")
    sh, sw = th - oh, tw - ow
    rows = 1 + (max(h - th, 0) + sh - 1) // sh
    columns = 1 + (max(w - tw, 0) + sw - 1) // sw
    count = n * rows * columns
    if count > max_tiles:
        raise ValueError("max_tiles exceeded")
    return dict(shape=(n, c, h, w), tile_size=(th, tw), overlap=(oh, ow),
                y_starts=list(range(0, rows * sh, sh)),
                x_starts=list(range(0, columns * sw, sw)), tile_count=count,
                tile_batch_size=tile_batch_size, blend=blend)


def blend_weights(tile_size, blend="constant", *, dtype=torch.float32):
    """Positive HW weights: constant 1 or centered Gaussian, sigma=side/8.

    Normalize Gaussian peak to 1, clamp to 1e-6 after normalization. This is
    deliberately fixed, with no user sigma parameter or half precision.
    """
    th, tw = _pair(tile_size, "tile_size", 1)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("weights require float32 or float64")
    if blend == "constant":
        return torch.ones(th, tw, dtype=dtype)
    if blend != "gaussian":
        raise ValueError("blend must be constant or gaussian")
    y = (torch.arange(th, dtype=dtype) - (th - 1) / 2) / (th / 8)
    x = (torch.arange(tw, dtype=dtype) - (tw - 1) / 2) / (tw / 8)
    weights = torch.exp(-0.5 * (y[:, None].square() + x[None, :].square()))
    return (weights / weights.max()).clamp_min_(1e-6)


@torch.inference_mode()
def tiled_logits(predictor, image, *, tile_size=(256, 256), overlap=(32, 32),
                 tile_batch_size=1, blend="constant", max_image_pixels=16_777_216,
                 max_tiles=65_536, max_tile_batch_size=64, max_tile_pixels=1_048_576):
    """Blend raw logits, then return NCHW on CPU in the input dtype.

    Caller must put nn.Modules in eval mode. No state/mode changes are made.
    Arbitrary callables must be deterministic, batch-independent predictors.
    Finite input/output and matching dtype/resolution are required. Classification
    (argmax or threshold) belongs AFTER this function; it never blends labels.
    """
    if not isinstance(image, torch.Tensor) or image.layout != torch.strided:
        raise ValueError("image must be a strided tensor")
    if image.device.type != "cpu" or image.dtype not in (torch.float32, torch.float64):
        raise ValueError("image must be CPU float32 or float64")
    plan = tile_plan(tuple(image.shape), tile_size=tile_size, overlap=overlap,
                     tile_batch_size=tile_batch_size, blend=blend,
                     max_image_pixels=max_image_pixels, max_tiles=max_tiles,
                     max_tile_batch_size=max_tile_batch_size, max_tile_pixels=max_tile_pixels)
    if not callable(predictor):
        raise ValueError("predictor must be callable")
    if isinstance(predictor, torch.nn.Module):
        if any(module.training for module in predictor.modules()):
            raise ValueError("predictor must be in eval mode")
        for tensor in itertools.chain(predictor.parameters(), predictor.buffers()):
            if tensor.device.type != "cpu" or (tensor.is_floating_point() and tensor.dtype != image.dtype):
                raise ValueError("predictor must match CPU input dtype")
    n, c, h, w = plan['shape']
    th, tw = plan['tile_size']
    weights = blend_weights((th, tw), blend, dtype=image.dtype)
    # One normalization map per image, independent of output channel count.
    normalization = torch.zeros(n, 1, h, w, dtype=image.dtype)
    accumulation = None
    coordinates = itertools.product(range(n), plan['y_starts'], plan['x_starts'])
    while True:
        batch_coords = list(itertools.islice(coordinates, tile_batch_size))
        if not batch_coords:
            break
        tiles = torch.zeros(len(batch_coords), c, th, tw, dtype=image.dtype)
        for i, (sample, y, x) in enumerate(batch_coords):
            vh, vw = min(th, h-y), min(tw, w-x)
            tiles[i, :, :vh, :vw] = image[sample, :, y:y+vh, x:x+vw]
        if not torch.isfinite(tiles).all():
            raise ValueError("image contains nonfinite values")
        logits = predictor(tiles)
        if (not isinstance(logits, torch.Tensor) or logits.layout != torch.strided
                or logits.device.type != "cpu" or logits.dtype != image.dtype
                or logits.ndim != 4 or logits.shape[0] != len(batch_coords)
                or logits.shape[1] < 1 or tuple(logits.shape[-2:]) != (th, tw)):
            raise ValueError("predictor must return CPU NCHW logits matching tile batch, size and dtype")
        if OUTPUT_CHANNELS is not None and logits.shape[1] != OUTPUT_CHANNELS:
            raise ValueError(f"expected {OUTPUT_CHANNELS} output channels")
        if not torch.isfinite(logits).all():
            raise ValueError("predictor returned nonfinite logits")
        if accumulation is None:
            accumulation = torch.zeros(n, logits.shape[1], h, w, dtype=image.dtype)
        elif logits.shape[1] != accumulation.shape[1]:
            raise ValueError("predictor changed output channels between batches")
        for i, (sample, y, x) in enumerate(batch_coords):
            vh, vw = min(th, h-y), min(tw, w-x)
            weight = weights[:vh, :vw]
            accumulation[sample, :, y:y+vh, x:x+vw].add_(logits[i, :, :vh, :vw] * weight)
            normalization[sample, :, y:y+vh, x:x+vw].add_(weight)
        # Do not retain previous tile outputs during the next model invocation.
        del tiles, logits
    if not (normalization > 0).all():
        raise RuntimeError("tile grid left uncovered pixels")
    accumulation.div_(normalization)
    if not torch.isfinite(accumulation).all():
        raise ValueError("logit accumulation overflowed")
    return accumulation
