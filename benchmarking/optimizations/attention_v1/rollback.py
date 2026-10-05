"""Restore the original attention sources, preserving the displaced version first.

Dry run: python3 benchmarking/optimizations/attention_v1/rollback.py
On the GPU: .venv/bin/python benchmarking/optimizations/attention_v1/rollback.py --apply --rebuild
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent / 'baseline'
TARGETS = [Path(f'runtime/production_kernels/{role}/attention/kernel.cu') for role in ['target', 'draft']] + [
    Path('runtime/production_kernels/target/attention/kernel_walkthrough.md'), Path('demo/WALKTHROUGH.md')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Restore source after saving the current version')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild both AOT attention extensions with this Python')
    args = parser.parse_args()
    if args.rebuild and not args.apply:
        parser.error('--rebuild requires --apply')
    manifest = json.loads((BASE / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((BASE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Backup checksum failed: {name}')
    for path in TARGETS:
        current = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        print(f'{path}: ' + ('already original' if current == manifest['files'][str(path)] else 'would restore original'))
    if not args.apply:
        print('Verified backup. No files changed. Use --apply --rebuild on the GPU to restore and rebuild.')
        return
    displaced = BASE.parent / ('rollback_displaced_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    displaced.mkdir()
    for path in TARGETS:
        saved = displaced / path
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, saved)
        for binary in (ROOT / path.parent).glob('*.so'):
            shutil.copy2(binary, saved.parent / binary.name)
    for path in TARGETS:
        shutil.copy2(BASE / path, ROOT / path)
    print(f'Restored original sources; displaced sources/binaries saved in {displaced}')
    if args.rebuild:
        import sys
        env = os.environ.copy()
        env.update(BUILD_ROLE='all', BUILD_KERNEL='attention')
        env.setdefault('MAX_JOBS', '2')
        env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace', '--force'], cwd=ROOT, env=env, check=True)
        print('Original attention extensions rebuilt. Restart any model process to load them.')
    else:
        print('Compiled extensions were not changed. Rebuild on the GPU before running inference.')

if __name__ == '__main__':
    main()
