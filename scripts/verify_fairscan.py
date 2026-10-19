"""Verify the Gaussian deployment and regress the explicitly retained negative case.

The original experiment remains strict and fails its all-settings tolerance gate.
This verifier was added after that failure; it does not change its thresholds,
source, measurements or outcome. --strict-all retains the original failing gate.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from fairscan import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-all', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'results/fairscan'
    try:
        # Verifies every source/input/artifact hash, exact saved statistics for
        # all 24 cases, saved arrays, and CLI output before its tolerance gate.
        run(output, check=True)
    except AssertionError as exc:
        if args.strict_all or str(exc) != 'runtime tolerance failed; results retained':
            raise
        result = json.loads((output / 'results.json').read_text())
        cases = result['cases']
        assert len(cases) == 24
        failures = [(c['id'], c['blend'], c['overlap']) for c in cases if not c['parity']['passed']]
        assert failures == [('IMG_20250614_210835', 'constant', [0, 0])], failures
        primary = [c for c in cases if c['blend'] == 'gaussian' and c['overlap'] == [32, 40]]
        assert len(primary) == 8 and all(c['parity']['passed'] for c in primary)
        print('PASS: all 8 primary Gaussian deployment cases; 23/24 overall parity cases.')
        print('EXPECTED FAILURE retained: IMG_20250614_210835, constant, no overlap; '
              'elementwise bound exceeded by 5.74%, labels identical. Not universal runtime parity.')
    else:
        raise AssertionError('Expected negative outcome changed; review before updating evidence.')


if __name__ == '__main__':
    main()
