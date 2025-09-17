#!/usr/bin/env python3
"""Download TEMPO O3TOT L2 data files from URL list."""

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

    # Convert RAD URLs to O3TOT URLs
    with open(rad_file_list, 'r') as f:
        rad_urls = [line.strip() for line in f if line.strip()]

    o3tot_urls = []
    file_mapping = {}  # Map O3TOT files to RAD files

    for url in rad_urls:
        # Replace RAD_L1 with O3TOT_L2 in both path and filename
        o3tot_url = url.replace('RAD_L1', 'O3TOT_L2')
        o3tot_urls.append(o3tot_url)

        # Track mapping
        rad_name = Path(url).name
        o3tot_name = Path(o3tot_url).name
        file_mapping[o3tot_name] = rad_name

    # Limit files
    max_files = config.get('max_files')
    if debug:
        max_files = 3
    if max_files:
        o3tot_urls = o3tot_urls[:max_files]
        # Update mapping to only include limited files
        file_mapping = {k: v for k, v in list(file_mapping.items())[:max_files]}

    print(f"Downloading {len(o3tot_urls)} O3TOT files")

    # Check auth
    if not (Path.home() / '.netrc').exists():
        print("ERROR: ~/.netrc required for NASA Earthdata")
        sys.exit(1)

    cookies = Path.home() / '.urs_cookies'
    cookies.touch(mode=0o600, exist_ok=True)

    # Save URL mapping
    mapping_path = output_dir / 'rad_to_o3tot_mapping.yaml'
    with open(mapping_path, 'w') as f:
        yaml.dump({'file_mapping': file_mapping}, f)

    # Download
    failed = []
    for i, url in enumerate(o3tot_urls, 1):
        filename = Path(url).name
        output_path = raw_dir / filename

        if output_path.exists():
            print(f"[{i}/{len(o3tot_urls)}] EXISTS: {filename}")
            continue

        print(f"[{i}/{len(o3tot_urls)}] Downloading: {filename}")

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
            if output_path.exists():
                output_path.unlink()  # Remove incomplete file

    print(f"\nDone. Downloaded: {len(o3tot_urls) - len(failed)}, Failed: {len(failed)}")
    if failed:
        print("Failed URLs:")
        for url in failed:
            print(f"  {url}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)