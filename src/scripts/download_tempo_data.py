#!/usr/bin/env python3
"""Download TEMPO data files from URL list."""

import argparse
import yaml
import sys
import subprocess
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import init_directory


def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required")
    if 'file_list' not in config:
        raise ValueError("FATAL: 'file_list' required")

    file_list = Path(config['file_list'])
    if not file_list.exists():
        raise ValueError(f"FATAL: {file_list} doesn't exist")

    # Setup output
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)
    raw_dir = output_dir / 'raw'
    raw_dir.mkdir(exist_ok=True)

    # Save config
    shutil.copy2(config_path, output_dir / 'config.yaml')

    # Get URLs
    with open(file_list, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    # Limit files
    max_files = config.get('max_files')
    if debug:
        max_files = 3
    if max_files:
        urls = urls[:max_files]

    print(f"Downloading {len(urls)} files")

    # Check auth
    if not (Path.home() / '.netrc').exists():
        print("ERROR: ~/.netrc required for NASA Earthdata")
        sys.exit(1)

    cookies = Path.home() / '.urs_cookies'
    cookies.touch(mode=0o600, exist_ok=True)

    # Download
    failed = []
    for i, url in enumerate(urls, 1):
        filename = Path(url).name
        output_path = raw_dir / filename

        if output_path.exists():
            print(f"[{i}/{len(urls)}] EXISTS: {filename}")
            continue

        print(f"[{i}/{len(urls)}] Downloading: {filename}")

        cmd = [
            'wget', '-q',
            '--load-cookies', str(cookies),
            '--save-cookies', str(cookies),
            '--keep-session-cookies',
            '--no-check-certificate',
            '-O', str(output_path),
            url
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED")
            failed.append(url)

    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)