"""CPU wall time and Linux fresh-process peak RSS, including all raw samples.

Every (workload, method, repetition) runs in a new interpreter. Peak RSS includes
imports, model/input setup, one warmup and three measured calls. It is NOT a
per-call incremental allocation estimate. No timing or memory improvement gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import torch

from unet_budget import compile_spec
from unet_budget.inference import tiled_logits, tile_plan
import unet_budget.inference as inference_module

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = [dict(name='padded_small', shape=[1, 1, 17, 21], width=4, depth=2, tile=[64, 80]),
             dict(name='small', shape=[1, 1, 65, 81], width=4, depth=2, tile=[32, 40]),
             dict(name='large', shape=[1, 1, 513, 641], width=12, depth=3, tile=[128, 160])]
METHODS = ['full', 'nonoverlapping', 'overlap_constant', 'overlap_gaussian']
CONFIG = dict(process_repetitions=3, warmups=1, timed_calls_per_process=3,
              model_seed=2901, input_seed=2902, threads=1, interop_threads=1,
              tile_batch_size=1, dtype='float32', device='cpu',
              workloads=WORKLOADS, methods=METHODS)


def sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def hardware():
    cpu_lines = Path('/proc/cpuinfo').read_text().splitlines()
    models = sorted(set(line.split(':', 1)[1].strip() for line in cpu_lines if line.startswith('model name')))
    return dict(platform=platform.platform(), machine=platform.machine(), cpu_models=models,
                logical_cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
                python=platform.python_version(), torch=torch.__version__,
                numpy=__import__('numpy').__version__, torch_build=torch.__config__.show(),
                thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})


def worker(workload_name, method):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('This RSS experiment requires Linux: ru_maxrss units are KiB.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    workload = next(w for w in WORKLOADS if w['name'] == workload_name)
    n, c, h, w = workload['shape']
    raw = dict(input_channels=c, output_classes=2, depth=workload['depth'], max_parameters=10**9,
               width_candidates=[workload['width']], input_height=h, input_width=w)
    code, report = compile_spec(raw)
    namespace = {}
    exec(code, namespace)
    torch.manual_seed(CONFIG['model_seed'])
    model = namespace['UNet']().eval()
    image = torch.randn(n, c, h, w, generator=torch.Generator().manual_seed(CONFIG['input_seed']))
    tile = workload['tile']
    overlap = [0, 0] if method == 'nonoverlapping' else [t//2 for t in tile]
    kwargs = dict(tile_size=tile, overlap=overlap, tile_batch_size=CONFIG['tile_batch_size'],
                  blend='gaussian' if method == 'overlap_gaussian' else 'constant')
    plan = tile_plan(image.shape, **kwargs)
    output_bytes = n * 2 * h * w * image.element_size()
    allocations = dict(input_bytes=image.numel()*image.element_size(), output_bytes=output_bytes,
                       normalization_bytes=0 if method == 'full' else n*h*w*image.element_size(),
                       tile_input_bytes=0 if method == 'full' else CONFIG['tile_batch_size']*c*tile[0]*tile[1]*image.element_size(),
                       tile_output_bytes=0 if method == 'full' else CONFIG['tile_batch_size']*2*tile[0]*tile[1]*image.element_size(),
                       blending_weights_bytes=0 if method == 'full' else tile[0]*tile[1]*image.element_size())
    setup_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    times = []
    hashes = []
    with torch.inference_mode():
        for iteration in range(CONFIG['warmups'] + CONFIG['timed_calls_per_process']):
            begin = time.perf_counter_ns()
            output = model(image) if method == 'full' else tiled_logits(model, image, **kwargs)
            elapsed = (time.perf_counter_ns() - begin) / 1e9
            if iteration >= CONFIG['warmups']:
                times.append(elapsed)
            assert list(output.shape) == [n, 2, h, w]
            assert torch.isfinite(output).all()
            hashes.append(sha(output))
            del output
    assert len(set(hashes)) == 1
    return dict(workload=workload_name, method=method, seconds=times,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                setup_peak_rss_bytes=setup_peak, output_shape=[n, 2, h, w], output_sha256=hashes[0],
                theoretical_tensor_sizes=allocations, input_sha256=sha(image),
                tile_count=0 if method == 'full' else plan['tile_count'],
                settings=None if method == 'full' else kwargs,
                specification=raw, generated_code_sha256=report['generated_code_sha256'])


def summarize(samples, workloads=WORKLOADS):
    summaries = []
    for workload in workloads:
        group = [s for s in samples if s['workload'] == workload['name']]
        full = [s for s in group if s['method'] == 'full']
        full_rss = statistics.median(s['peak_rss_bytes'] for s in full)
        full_time = statistics.median(t for s in full for t in s['seconds'])
        for method in METHODS:
            selected = [s for s in group if s['method'] == method]
            rss = statistics.median(s['peak_rss_bytes'] for s in selected)
            timing = statistics.median(t for s in selected for t in s['seconds'])
            summaries.append(dict(workload=workload['name'], method=method,
                                  median_seconds=timing, median_peak_rss_bytes=rss,
                                  rss_ratio_to_full=rss/full_rss, runtime_ratio_to_full=timing/full_time,
                                  reduced_median_process_rss=rss < full_rss))
    return summaries


def validate_saved(result, config=CONFIG):
    assert result['config'] == config
    assert len(result['samples']) == len(config['workloads'])*len(METHODS)*config['process_repetitions']
    assert result['summary'] == summarize(result['samples'], config['workloads'])
    for workload in config['workloads']:
        for method in METHODS:
            samples = [s for s in result['samples'] if s['workload'] == workload['name'] and s['method'] == method]
            assert len(samples) == CONFIG['process_repetitions']
            assert sorted(s['repetition'] for s in samples) == list(range(CONFIG['process_repetitions']))
            assert len({s['output_sha256'] for s in samples}) == 1
            for sample in samples:
                assert len(sample['seconds']) == CONFIG['timed_calls_per_process']
                assert all(0 < t < 3600 for t in sample['seconds'])
                assert sample['peak_rss_bytes'] >= sample['setup_peak_rss_bytes'] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=[w['name'] for w in WORKLOADS])
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--output', type=Path, default=ROOT/'results/tiled-benchmark.json')
    parser.add_argument('--check', type=Path, help='validate saved samples and rerun in fresh processes; no speed/RSS gate')
    args = parser.parse_args()
    if args.worker:
        if not args.method:
            parser.error('--worker requires --method')
        print(json.dumps(worker(args.worker, args.method), sort_keys=True))
        return
    samples = []
    # Rotate method order across process repetitions to reduce order confounding.
    for workload in WORKLOADS:
        for repetition in range(CONFIG['process_repetitions']):
            order = METHODS[repetition:] + METHODS[:repetition]
            for method in order:
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', workload['name'], '--method', method]
                run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
                samples.append(dict(json.loads(run.stdout), repetition=repetition))
                print(f"Measured {workload['name']} / {method} / process {repetition+1}", flush=True)
    result = dict(schema_version=1, config=CONFIG, environment=hardware(), samples=samples,
                  summary=summarize(samples),
                  helper_sha256=hashlib.sha256(Path(inference_module.__file__).read_bytes()).hexdigest(),
                  measurement='Linux ru_maxrss * 1024; whole fresh process including import/setup/warmup/timed calls/checks; perf_counter_ns for calls only',
                  limitations=['RSS includes allocator caches, Python, PyTorch and workspaces; tensor sizes are not peak-memory estimates.',
                               'Shared host wall times and RSS vary; no inference of statistical significance from three processes.',
                               'No GPU, real-world images, process-memory cap, or guarantee that tiling saves memory.'])
    validate_saved(result)
    if args.check:
        saved = json.loads(args.check.read_text())
        validate_saved(saved)
        assert saved['helper_sha256'] == result['helper_sha256'], 'helper changed; regenerate benchmark'
        def numerical(records):
            return sorted((s['workload'], s['method'], s['repetition'], s['output_sha256'], s['input_sha256'],
                           s['generated_code_sha256'], s['tile_count'], s['output_shape'], s['theoretical_tensor_sizes'],
                           s['settings'], s['specification']) for s in records)
        assert numerical(saved['samples']) == numerical(samples), 'benchmark numerical/configuration mismatch'
        # Retain and verify the initial two-workload run as well; adding the
        # padded-small workload must not discard earlier raw measurements.
        initial_path = ROOT/'results/tiled-benchmark-initial.json'
        initial = json.loads(initial_path.read_text())
        validate_saved(initial, {**CONFIG, 'workloads': WORKLOADS[1:]})
        assert initial['helper_sha256'] == result['helper_sha256']
        assert numerical(initial['samples']) == numerical([s for s in samples if s['workload'] != 'padded_small'])
        print(f'Saved benchmark structure and numerical outputs verified in {len(samples)} fresh processes; timings/RSS may differ.')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
