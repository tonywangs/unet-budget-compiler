"""Command-line entry point. Compilation never imports PyTorch."""
import argparse
import json
from pathlib import Path
import sys

from . import SpecError, __version__, compile_spec


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SpecError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compile a bounded 2D U-Net specification offline.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("spec", type=Path, help="JSON specification")
    parser.add_argument("--out", type=Path, required=True, help="new or empty artifact directory")
    parser.add_argument("--with-inference", action="store_true", help="also emit standalone CPU tiled inference.py")
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.spec.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        code, report = compile_spec(raw)
        helper = None
        if args.with_inference:
            import hashlib
            helper = (Path(__file__).with_name('inference.py').read_text(encoding='utf-8')
                      .replace('MIN_TILE_SIZE = 1', f'MIN_TILE_SIZE = {2 ** raw["depth"]}')
                      .replace('INPUT_CHANNELS = None', f'INPUT_CHANNELS = {raw["input_channels"]}')
                      .replace('OUTPUT_CHANNELS = None', f'OUTPUT_CHANNELS = {raw["output_classes"]}'))
            report['inference'] = dict(file='inference.py', schema_version=1,
                                     sha256=hashlib.sha256(helper.encode()).hexdigest())
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise SpecError(f"output path must be a new or empty directory: {args.out}")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "model.py").write_text(code, encoding="utf-8")
        if helper is not None:
            (args.out / 'inference.py').write_text(helper, encoding='utf-8')
        (args.out / "architecture.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (SpecError, OSError, ValueError, RecursionError) as exc:
        print(f"unet-budget: {exc}", file=sys.stderr)
        return 2
    print(f"Selected width {report['selection']['base_width']}: {report['selection']['parameters']} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
