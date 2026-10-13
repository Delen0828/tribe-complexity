"""Extract the two supplements and build their stimulus/provenance manifests."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studies import prepare_manifest, MANIFESTS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['kylie', 'hayeong', 'both'], default='both')
    args = parser.parse_args()
    for name in ('kylie', 'hayeong') if args.dataset == 'both' else (args.dataset,):
        manifest = prepare_manifest(name)
        print(f'{name}: {len(manifest["rows"])} stimuli, {len(manifest["comparisons"])} comparisons; {MANIFESTS / (name + ".json")}')
