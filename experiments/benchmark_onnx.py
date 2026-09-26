"""Balanced paired CPU measurements in fresh runtime-specific processes.

No torch or compiler imports at module scope: ORT worker RSS excludes them.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name('onnx_protocol.json')
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
CONFIG = PROTOCOL['benchmark']
METHODS = ('torch', 'onnx')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(method, directory):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Peak RSS measurement requires Linux (ru_maxrss in KiB).')
    begin = time.perf_counter_ns()
    startup_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    import numpy as np
    if method == 'torch':
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    else:
        import onnxruntime as ort
    imported = time.perf_counter_ns()
    image = np.load(directory/'input.npy', allow_pickle=False)
    manifest = json.loads((directory/'bundle/manifest.json').read_text())
    assert image.dtype == np.float32 and list(image.shape) == manifest['input']['shape']
    assert np.isfinite(image).all()
    if method == 'torch':
        module = load_module(directory/'bundle/model.py', 'standalone_model')
        with torch.device('meta'):
            model = module.UNet().eval()
        model.load_state_dict(torch.load(directory/'weights.pt', weights_only=True, map_location='cpu'), assign=True)
        tensor = torch.from_numpy(image)
        def call():
            return model(tensor)
        context = torch.inference_mode()
        threads = dict(intra_op=torch.get_num_threads(), inter_op=torch.get_num_interop_threads())
    else:
        module = load_module(directory/'bundle/inference.py', 'standalone_inference')
        session, _ = module.load_bundle(directory/'bundle', threads=1)
        def call():
            return session.run(['logits'], {'image': image})[0]
        from contextlib import nullcontext
        context = nullcontext()
        assert not {'torch', 'unet_budget', 'onnx'} & sys.modules.keys()
        options = session.get_session_options()
        threads = dict(intra_op=options.intra_op_num_threads, inter_op=options.inter_op_num_threads,
                       execution_mode=str(options.execution_mode), graph_optimization=str(options.graph_optimization_level),
                       providers=session.get_providers())
    initialized = time.perf_counter_ns()
    setup_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    warmups, seconds, hashes = [], [], []
    with context:
        for index in range(CONFIG['warmups'] + CONFIG['timed_calls']):
            started = time.perf_counter_ns()
            output = call()
            elapsed = (time.perf_counter_ns() - started) / 1e9
            (warmups if index < CONFIG['warmups'] else seconds).append(elapsed)
            actual = output.numpy() if method == 'torch' else output
            assert list(actual.shape) == manifest['output']['shape'] and np.isfinite(actual).all()
            hashes.append(hashlib.sha256(actual.tobytes()).hexdigest())
            if index == CONFIG['warmups'] + CONFIG['timed_calls'] - 1:
                np.save(directory/f'{method}-output.npy', actual, allow_pickle=False)
            del actual, output
    assert len(set(hashes)) == 1
    return dict(method=method, seconds=seconds, warmup_seconds=warmups,
                import_seconds=(imported-begin)/1e9, model_input_setup_seconds=(initialized-imported)/1e9,
                initialization_seconds=(initialized-begin)/1e9,
                startup_peak_rss_bytes=startup_peak,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                setup_peak_rss_bytes=setup_peak, threads=threads,
                output_shape=manifest['output']['shape'], output_sha256=hashes[0],
                input_file_sha256=sha(directory/'input.npy'), checkpoint_sha256=sha(directory/'weights.pt'),
                onnx_sha256=sha(directory/'bundle/model.onnx'),
                generated_source_sha256=sha(directory/'bundle/model.py'),
                compiler_imported='unet_budget' in sys.modules, torch_imported='torch' in sys.modules,
                thread_environment={k: os.environ.get(k) for k in
                                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})


def environment():
    cpu_lines = Path('/proc/cpuinfo').read_text().splitlines()
    cpu_models = sorted({line.split(':', 1)[1].strip() for line in cpu_lines if line.startswith('model name')})
    quota = Path('/sys/fs/cgroup/cpu.max')
    return dict(platform=platform.platform(), python=platform.python_version(), machine=platform.machine(),
                logical_cpus=os.cpu_count(), cpu_models=cpu_models, affinity=sorted(os.sched_getaffinity(0)),
                cgroup_cpu_max=quota.read_text().strip() if quota.exists() else None,
                versions={name: importlib.metadata.version(name) for name in ('torch', 'onnxruntime', 'onnx', 'numpy')},
                torch_build=__import__('torch').__config__.show(),
                onnxruntime_build=__import__('onnxruntime').get_build_info())


def summarize(samples):
    summaries = []
    for workload in CONFIG['workloads']:
        selected = [s for s in samples if s['workload'] == workload['name']]
        ratios = []
        for pair in range(CONFIG['process_pairs']):
            records = {s['method']: s for s in selected if s['pair'] == pair}
            a, b = records['torch'], records['onnx']
            ratios.append(dict(pair=pair, onnx_to_torch_latency=statistics.median(b['seconds'])/statistics.median(a['seconds']),
                               onnx_to_torch_peak_rss=b['peak_rss_bytes']/a['peak_rss_bytes'],
                               onnx_to_torch_initialization=b['initialization_seconds']/a['initialization_seconds']))
        methods = {}
        for method in METHODS:
            records = [s for s in selected if s['method'] == method]
            times = [t for s in records for t in s['seconds']]
            methods[method] = dict(median_seconds=statistics.median(times), min_seconds=min(times), max_seconds=max(times),
                                   median_process_median_seconds=statistics.median(statistics.median(s['seconds']) for s in records),
                                   median_peak_rss_bytes=statistics.median(s['peak_rss_bytes'] for s in records),
                                   median_import_seconds=statistics.median(s['import_seconds'] for s in records),
                                   median_model_input_setup_seconds=statistics.median(s['model_input_setup_seconds'] for s in records),
                                   median_initialization_seconds=statistics.median(s['initialization_seconds'] for s in records),
                                   median_process_wall_seconds=statistics.median(s['process_wall_seconds'] for s in records))
        summaries.append(dict(workload=workload['name'], methods=methods, paired_ratios=ratios,
                              median_paired_latency_ratio=statistics.median(r['onnx_to_torch_latency'] for r in ratios),
                              median_paired_rss_ratio=statistics.median(r['onnx_to_torch_peak_rss'] for r in ratios),
                              slower_onnx_pairs=sum(r['onnx_to_torch_latency'] > 1 for r in ratios)))
    return summaries


def validate(result):
    assert result['protocol'] == PROTOCOL
    samples = result['samples']
    assert len(samples) == len(CONFIG['workloads']) * CONFIG['process_pairs'] * 2
    assert result['summary'] == summarize(samples)
    for workload in CONFIG['workloads']:
        group = [s for s in samples if s['workload'] == workload['name']]
        for method in METHODS:
            records = [s for s in group if s['method'] == method]
            assert sorted(s['pair'] for s in records) == list(range(CONFIG['process_pairs']))
            assert len({s['output_sha256'] for s in records}) == 1
        for pair in range(CONFIG['process_pairs']):
            records = [s for s in group if s['pair'] == pair]
            order = list(METHODS) if pair % 2 == 0 else list(reversed(METHODS))
            assert [s['method'] for s in records] == order
            for index, record in enumerate(records):
                assert record['order_in_pair'] == index
                assert record['torch_imported'] == (record['method'] == 'torch')
                assert not record['compiler_imported']
                assert record['threads']['intra_op'] == record['threads']['inter_op'] == 1
                assert len(record['seconds']) == CONFIG['timed_calls']
                assert len(record['warmup_seconds']) == CONFIG['warmups']
                assert all(0 < t < 3600 for t in record['seconds'] + record['warmup_seconds'])
                assert record['initialization_seconds'] > 0 and record['model_input_setup_seconds'] > 0
                assert record['import_seconds'] > 0 and record['process_wall_seconds'] > record['initialization_seconds']
                assert record['peak_rss_bytes'] >= record['setup_peak_rss_bytes'] > 0
                assert record['setup_peak_rss_bytes'] > record['startup_peak_rss_bytes'] > 0
            for key in ('input_file_sha256', 'checkpoint_sha256', 'onnx_sha256', 'generated_source_sha256', 'output_shape'):
                assert records[0][key] == records[1][key]
    assert len(result['comparisons']) == len(CONFIG['workloads']) * CONFIG['process_pairs']
    tol = PROTOCOL['tolerances']
    for comparison in result['comparisons']:
        assert comparison['finite']
        assert comparison['elementwise_close'] == (comparison['elementwise_violation_count'] == 0)
        meets_bounds = comparison['elementwise_close']
        for key in ('maximum_absolute_error', 'mean_absolute_error', 'label_disagreement_fraction'):
            assert comparison[key] >= 0
            meets_bounds &= comparison[key] <= tol[key]
        assert comparison['within_tolerance'] == meets_bounds
    assert result['tolerance_failures'] == [dict(workload=c['workload'], pair=c['pair'])
                                             for c in result['comparisons'] if not c['within_tolerance']]
    # A structural benchmark check must never silently accept an uninvestigated
    # negative result. Compare every failure with the separately replayed case.
    investigation = json.loads((ROOT/'results/onnx-numerical-investigation.json').read_text())
    assert investigation['protocol_sha256'] == result['protocol_sha256']
    for record in result['comparisons']:
        if not record['within_tolerance']:
            assert record['workload'] == 'larger', 'new failing workload needs investigation'
            assert {k: v for k, v in record.items() if k not in ('workload', 'pair')} == investigation['comparisons_to_torch_float32']['onnx_default'], 'new numerical failure needs investigation'


def experiment():
    import shutil
    import numpy as np
    import torch
    from unet_budget.export import export_model
    from compare_onnx import comparison
    torch.set_num_threads(1)
    samples, comparisons, artifacts = [], [], {}
    spec = json.loads((ROOT/'examples/circles.json').read_text())
    with tempfile.TemporaryDirectory(prefix='unet-onnx-bench-') as temporary:
        temp = Path(temporary)
        for workload in CONFIG['workloads']:
            directory = temp/workload['name']
            directory.mkdir()
            shutil.copyfile(ROOT/'results/synthetic-weights.pt', directory/'weights.pt')
            export_model(spec, directory/'weights.pt', workload['shape'], directory/'bundle')
            image = np.random.default_rng(CONFIG['input_seed']).standard_normal(workload['shape']).astype('float32')
            np.save(directory/'input.npy', image, allow_pickle=False)
            artifacts[workload['name']] = {str(p.relative_to(directory)): dict(bytes=p.stat().st_size, sha256=sha(p))
                                          for p in sorted(directory.rglob('*')) if p.is_file()}
            for pair in range(CONFIG['process_pairs']):
                order = METHODS if pair % 2 == 0 else tuple(reversed(METHODS))
                for index, method in enumerate(order):
                    # A fork/exec directly from this torch-loaded exporter can
                    # inherit its RSS high-water mark. Exec a lightweight stdlib
                    # supervisor first, then fork the actual measured worker.
                    command = [sys.executable, str(Path(__file__).resolve()), '--supervise', method, '--directory', str(directory)]
                    begin = time.perf_counter_ns()
                    run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180,
                                         env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
                    wall = (time.perf_counter_ns()-begin)/1e9
                    samples.append(dict(json.loads(run.stdout), workload=workload['name'], pair=pair,
                                        order_in_pair=index, process_wall_seconds=wall, stderr=run.stderr.strip()))
                    print(f"Measured {workload['name']} pair {pair+1}: {method}", flush=True)
                comparisons.append(dict(workload=workload['name'], pair=pair,
                                        **comparison(np.load(directory/'torch-output.npy'), np.load(directory/'onnx-output.npy'),
                                                     require_close=False)))
                if not comparisons[-1]['within_tolerance']:
                    print(f"Frozen numerical tolerance FAILED: {workload['name']} pair {pair+1}; retained for investigation", flush=True)
    result = dict(schema_version=1, protocol=PROTOCOL, protocol_sha256=sha(PROTOCOL_PATH), environment=environment(),
                  source_hashes={name: sha(ROOT/name) for name in ('experiments/benchmark_onnx.py', 'experiments/compare_onnx.py',
                                  'src/unet_budget/export.py', 'src/unet_budget/onnx_inference.py')},
                  artifacts=artifacts, samples=samples, comparisons=comparisons, summary=summarize(samples),
                  tolerance_failures=[dict(workload=c['workload'], pair=c['pair']) for c in comparisons if not c['within_tolerance']],
                  measurement='Fresh Linux worker from a stdlib-only supervisor: ru_maxrss * 1024; direct calls timed by perf_counter_ns after 3 warmups',
                  limitations=['Shared-host CPU measurements, six paired repetitions; no statistical significance claim.',
                               'RSS includes runtime imports, caches, setup and all calls; not incremental tensor memory.',
                               'Only three workloads, one small trained synthetic model, float32 and one thread.',
                               'Timing excludes preprocessing, validation and output serialization; initialization is reported separately.',
                               'No acceleration guarantee; preserve slower runs and ratios above one.'])
    validate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=METHODS)
    parser.add_argument('--supervise', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'results/onnx-benchmark.json')
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.supervise:
        if not args.directory:
            parser.error('--supervise requires --directory')
        assert not {'torch', 'numpy', 'onnxruntime', 'unet_budget'} & sys.modules.keys()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', args.supervise,
                   '--directory', str(args.directory)]
        run = subprocess.run(command, check=True, capture_output=True, text=True, timeout=170)
        print(run.stdout, end='')
        print(run.stderr, file=sys.stderr, end='')
        return
    if args.worker:
        if not args.directory:
            parser.error('--worker requires --directory')
        print(json.dumps(worker(args.worker, args.directory), sort_keys=True))
        return
    result = experiment()
    if args.check:
        saved = json.loads(args.check.read_text())
        validate(saved)
        for key in ('protocol_sha256', 'source_hashes', 'artifacts', 'comparisons', 'tolerance_failures'):
            assert saved[key] == result[key], f'benchmark reproduction mismatch: {key}'
        timing_keys = {'seconds', 'warmup_seconds', 'import_seconds', 'model_input_setup_seconds', 'initialization_seconds',
                       'peak_rss_bytes', 'setup_peak_rss_bytes', 'startup_peak_rss_bytes', 'process_wall_seconds', 'stderr'}
        numerical = lambda records: [{k: v for k, v in s.items() if k not in timing_keys} for s in records]
        assert numerical(saved['samples']) == numerical(result['samples'])
        print('Saved measurements validated; numerical/configuration evidence (including known failures) reproduced in 36 fresh processes. No timing/RSS equality gate.')
    else:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result['summary'], sort_keys=True))
    print('Frozen-tolerance failures: ' + json.dumps(result['tolerance_failures']))


if __name__ == '__main__':
    main()
