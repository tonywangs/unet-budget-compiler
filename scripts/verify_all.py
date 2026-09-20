"""Run the milestone checks; optionally preserve plain test output."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, help='save plain output, e.g. results/tests.log')
    args = parser.parse_args()
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        [sys.executable, 'scripts/verify_isolated.py'],
        [sys.executable, 'experiments/train_synthetic.py', '--check', 'results/synthetic.json'],
    ]
    output = []
    failed = False
    for command in commands:
        header = '$ ' + shlex.join(['python'] + command[1:]) + '\n'
        print(header, end='', flush=True)
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        print(result.stdout, end='', flush=True)
        output.extend((header, result.stdout, f'Exit status: {result.returncode}\n\n'))
        failed |= result.returncode != 0
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(''.join(output).rstrip() + '\n', encoding='utf-8')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
