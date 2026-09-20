"""Strict specification parsing, integer accounting, and deterministic emission."""
import hashlib
import json

from . import __version__


class SpecError(ValueError):
    """A specification cannot be compiled within this candidate family."""


FIELDS = {
    "input_channels", "output_classes", "depth", "max_parameters",
    "width_candidates", "input_height", "input_width",
}


def normalize_spec(raw):
    if not isinstance(raw, dict):
        raise SpecError("specification must be a JSON object")
    missing, extra = FIELDS - raw.keys(), raw.keys() - FIELDS
    if missing or extra:
        raise SpecError(f"schema mismatch: missing={sorted(missing)}, unknown={sorted(extra)}")
    bounds = {
        "input_channels": (1, 1024), "output_classes": (1, 1024),
        "depth": (1, 8), "max_parameters": (1, 10**12),
        "input_height": (1, 65536), "input_width": (1, 65536),
    }
    for key, (low, high) in bounds.items():
        value = raw[key]
        if type(value) is not int or not low <= value <= high:
            raise SpecError(f"{key} must be an integer in [{low}, {high}] (booleans are not integers)")
    widths = raw["width_candidates"]
    if not isinstance(widths, list) or not 1 <= len(widths) <= 256:
        raise SpecError("width_candidates must be a list of 1 to 256 integers")
    if any(type(w) is not int or not 1 <= w <= 1024 for w in widths):
        raise SpecError("each width candidate must be an integer in [1, 1024]")
    for axis in ("input_height", "input_width"):
        if raw[axis] < 2 ** raw["depth"]:
            raise SpecError(f"{axis} must be at least {2 ** raw['depth']} for depth={raw['depth']}")
    return {**raw, "width_candidates": sorted(set(widths))}


def parameter_count(spec, width):
    """Count weights and biases; no tensor allocation or floating-point arithmetic."""
    def block(cin, cout):
        return 9 * cin * cout + cout + 9 * cout * cout + cout
    total = block(spec["input_channels"], width)
    for level in range(1, spec["depth"] + 1):
        total += block(width * 2 ** (level - 1), width * 2 ** level)
    for level in reversed(range(spec["depth"])):
        channels = width * 2 ** level
        total += block(3 * channels, channels)
    return total + width * spec["output_classes"] + spec["output_classes"]


def architecture(spec, width):
    """Resolve every primitive operation and its NCHW shape for the example input."""
    layers = []
    shape = ["N", spec["input_channels"], spec["input_height"], spec["input_width"]]

    def add(name, kind, inputs, output, parameters=0, **attrs):
        layers.append(dict(name=name, kind=kind, inputs=inputs, output=output,
                           parameters=parameters, **attrs))
        return output

    def conv(name, current, channels, kernel):
        output = ["N", channels, *current[2:]]
        return add(name, "conv2d", [current], output,
                   channels * (current[1] * kernel**2 + 1),
                   in_channels=current[1], out_channels=channels,
                   kernel_size=kernel, padding=kernel // 2, bias=True)

    def block(name, current, channels):
        for suffix in ("a", "b"):
            current = conv(f"{name}_{suffix}", current, channels, 3)
            current = add(f"{name}_{suffix}_relu", "relu", [current], current[:])
        return current

    skips = []
    for level in range(spec["depth"] + 1):
        if level:
            shape = add(f"pool_{level}", "max_pool2d", [shape],
                        ["N", shape[1], shape[2] // 2, shape[3] // 2],
                        kernel_size=2, stride=2, ceil_mode=False)
        shape = block(f"enc_{level}", shape, width * 2**level)
        skips.append(shape)
    for level in reversed(range(spec["depth"])):
        skip = skips[level]
        shape = add(f"resize_{level}", "interpolate", [shape],
                    ["N", shape[1], *skip[2:]], mode="nearest")
        shape = add(f"concat_{level}", "concatenate", [skip, shape],
                    ["N", skip[1] + shape[1], *skip[2:]], dim=1)
        shape = block(f"dec_{level}", shape, width * 2**level)
    conv("head", shape, spec["output_classes"], 1)
    return layers


def emit_code(spec, width, layers, digest):
    lines = [
        '"""Generated standalone U-Net. Outputs raw NCHW logits; requires PyTorch.',
        f'Compiler {__version__}; normalized specification SHA-256: {digest}',
        'Do not edit if you need byte-for-byte regeneration."""',
        'import torch', 'from torch import nn', 'from torch.nn import functional as F', '', '',
        'class UNet(nn.Module):',
        '    """Padded double convolutions, floor pooling, nearest resize, skip concatenation."""',
        '    def __init__(self):', '        super().__init__()',
    ]
    for layer in layers:
        if layer["kind"] == "conv2d":
            lines.append(f"        self.{layer['name']} = nn.Conv2d({layer['in_channels']}, {layer['out_channels']}, kernel_size={layer['kernel_size']}, padding={layer['padding']}, bias=True)")
    lines += ['', '    def forward(self, x):',
              f'        if x.ndim != 4 or x.shape[1] != {spec["input_channels"]}:',
              f'            raise ValueError("expected NCHW input with {spec["input_channels"]} channels")',
              f'        if x.shape[0] < 1 or min(x.shape[-2:]) < {2**spec["depth"]}:',
              f'            raise ValueError("batch must be nonempty and height/width must be >= {2**spec["depth"]}")']
    for level in range(spec["depth"] + 1):
        if level:
            lines.append('        x = F.max_pool2d(x, kernel_size=2, stride=2)')
        lines += [f'        x = F.relu(self.enc_{level}_a(x))',
                  f'        x = F.relu(self.enc_{level}_b(x))']
        if level < spec["depth"]:
            lines.append(f'        skip_{level} = x')
    for level in reversed(range(spec["depth"])):
        lines += [f'        x = F.interpolate(x, size=skip_{level}.shape[-2:], mode="nearest")',
                  f'        x = torch.cat((skip_{level}, x), dim=1)',
                  f'        x = F.relu(self.dec_{level}_a(x))',
                  f'        x = F.relu(self.dec_{level}_b(x))']
    lines += ['        return self.head(x)', '']
    return '\n'.join(lines)


def compile_spec(raw):
    """Return (standalone Python source, JSON-serializable architecture report)."""
    spec = normalize_spec(raw)
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    candidates = [{"base_width": w, "parameters": parameter_count(spec, w)}
                  for w in spec["width_candidates"]]
    feasible = [c for c in candidates if c["parameters"] <= spec["max_parameters"]]
    if not feasible:
        raise SpecError(f"no width fits max_parameters={spec['max_parameters']}; smallest candidate width={candidates[0]['base_width']} needs {candidates[0]['parameters']} parameters; increase budget or include smaller widths")
    chosen = feasible[-1]
    layers = architecture(spec, chosen["base_width"])
    assert sum(layer["parameters"] for layer in layers) == chosen["parameters"]
    code = emit_code(spec, chosen["base_width"], layers, digest)
    report = dict(report_schema_version=1, compiler_version=__version__,
                  family="padded-nearest-unet-v1", specification=spec,
                  specification_sha256=digest,
                  generated_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                  selection=chosen, candidates=candidates, layers=layers,
                  output_semantics="raw logits; no sigmoid or softmax",
                  shape_convention="NCHW; N is any positive batch size; shapes use specification height/width")
    return code, report
