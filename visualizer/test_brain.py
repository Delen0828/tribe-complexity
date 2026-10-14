"""Run the shared visualizer browser regressions with the installed Node Playwright."""
import argparse
import subprocess
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/')
    args = parser.parse_args()
    subprocess.run(['node', str(Path(__file__).with_name('test_studies.cjs')), args.url], check=True)
