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
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.spec.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        code, report = compile_spec(raw)
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise SpecError(f"output path must be a new or empty directory: {args.out}")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "model.py").write_text(code, encoding="utf-8")
        (args.out / "architecture.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (SpecError, OSError, ValueError, RecursionError) as exc:
        print(f"unet-budget: {exc}", file=sys.stderr)
        return 2
    print(f"Selected width {report['selection']['base_width']}: {report['selection']['parameters']} parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
