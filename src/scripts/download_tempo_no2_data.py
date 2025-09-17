#!/usr/bin/env python3
"""Download TEMPO NO2 L2 data files from URL list."""

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
    if 'rad_file_list' not in config:
        raise ValueError("FATAL: 'rad_file_list' required")

    rad_file_list = Path(config['rad_file_list'])
    if not rad_file_list.exists():
        raise ValueError(f"FATAL: {rad_file_list} doesn't exist")

    # Setup output
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)
    raw_dir = output_dir / 'raw'
    raw_dir.mkdir(exist_ok=True)

    # Save config
    shutil.copy2(config_path, output_dir / 'config.yaml')

    # Convert RAD URLs to NO2 URLs
    with open(rad_file_list, 'r') as f:
        rad_urls = [line.strip() for line in f if line.strip()]

    no2_urls = []
    for url in rad_urls:
        # Replace RAD_L1 with NO2_L2 in both path and filename
        no2_url = url.replace('RAD_L1', 'NO2_L2')
        no2_urls.append(no2_url)

    # Limit files
    max_files = config.get('max_files')
    if debug:
        max_files = 3
    if max_files:
        no2_urls = no2_urls[:max_files]

    print(f"Downloading {len(no2_urls)} NO2 files")

    # Check auth
    if not (Path.home() / '.netrc').exists():
        print("ERROR: ~/.netrc required for NASA Earthdata")
        sys.exit(1)

    cookies = Path.home() / '.urs_cookies'
    cookies.touch(mode=0o600, exist_ok=True)

    # Save URL mapping
    url_mapping = {}
    for rad_url, no2_url in zip(rad_urls[:len(no2_urls)], no2_urls):
        rad_name = Path(rad_url).name
        no2_name = Path(no2_url).name
        url_mapping[rad_name] = no2_name

    mapping_path = output_dir / 'rad_to_no2_mapping.yaml'
    with open(mapping_path, 'w') as f:
        yaml.dump(url_mapping, f)

    # Download
    failed = []
    for i, url in enumerate(no2_urls, 1):
        filename = Path(url).name
        output_path = raw_dir / filename

        if output_path.exists():
            print(f"[{i}/{len(no2_urls)}] EXISTS: {filename}")
            continue

        print(f"[{i}/{len(no2_urls)}] Downloading: {filename}")

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