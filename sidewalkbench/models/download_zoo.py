"""Download and verify checkpoints for all nine current benchmark models."""
from argparse import ArgumentParser
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import yaml


DEFAULT_MODELS = (
    'ViNT', 'NoMaD', 'MBRA', 'CityWalker', 'InternVLA_N1',
    'Cosmos3', 'MIMIC', 'S2E', 'FlowPilot',
)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def download(models=DEFAULT_MODELS, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    for model in models:
        if model not in DEFAULT_MODELS:
            raise ValueError(f'Unsupported download model: {model}')
        folder = root / model
        info = yaml.safe_load((folder / 'model_info.yaml').read_text())
        for name, spec in info['checkpoint_files'].items():
            target = folder / name
            if target.exists() and target.stat().st_size == spec['size'] and digest(target) == spec['sha256']:
                print(f'{model}/{name}: verified existing file', flush=True)
                continue
            repository = spec.get('repository', info['checkpoint_repository'])
            revision = spec.get('revision', info['checkpoint_revision'])
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{spec['public_path']}"
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + '.part')
            hasher = hashlib.sha256()
            total = 0
            print(f'{model}/{name}: downloading {spec["size"]} bytes', flush=True)
            try:
                with urlopen(Request(url, headers={'User-Agent': 'SidewalkBench-models/1.0'}), timeout=120) as response, partial.open('wb') as stream:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                        stream.write(block)
                        hasher.update(block)
                        total += len(block)
                if total != spec['size'] or hasher.hexdigest() != spec['sha256']:
                    raise ValueError(f'Checkpoint verification failed: {model}/{name}')
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            print(f'{model}/{name}: verified', flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=DEFAULT_MODELS, default=DEFAULT_MODELS,
                        help='Models to download (default: all nine; excludes OpenPilot).')
    args = parser.parse_args()
    download(args.models)


if __name__ == '__main__':
    main()
