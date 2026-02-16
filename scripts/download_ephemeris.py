#!/usr/bin/env python3
"""
Download JPL ephemeris data for Loka.

This script downloads the necessary SPK (Spacecraft and Planet Kernel) files
from JPL for celestial body position calculations.
"""

import argparse
import os
from pathlib import Path
import urllib.request

# Available ephemeris files
EPHEMERIS_FILES = {
    "de440s": {
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
        "description": "Small DE440 ephemeris (1849-2150)",
    },
    "de440": {
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
        "description": "Full DE440 ephemeris (1550-2650)",
    },
    "de441": {
        "url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441.bsp",
        "description": "Extended DE441 ephemeris (-13200 to 17191)",
    },
}


def download_file(url: str, destination: Path, show_progress: bool = True):
    """Download a file with progress indicator."""
    print(f"Downloading: {url}")
    print(f"Destination: {destination}")
    
    def reporthook(block_num, block_size, total_size):
        if show_progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
    
    urllib.request.urlretrieve(url, destination, reporthook if show_progress else None)
    print("\nDownload complete!")


def main():
    parser = argparse.ArgumentParser(description="Download JPL ephemeris files")
    parser.add_argument(
        "--target",
        type=str,
        choices=list(EPHEMERIS_FILES.keys()),
        default="de440s",
        help="Ephemeris file to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: $LOKA_DATA_DIR/ephemeris)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available ephemeris files",
    )
    args = parser.parse_args()
    
    if args.list:
        print("Available ephemeris files:")
        for name, info in EPHEMERIS_FILES.items():
            print(f"  {name}: {info['description']}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        data_dir = os.environ.get("LOKA_DATA_DIR", "./data")
        output_dir = Path(data_dir) / "ephemeris"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the file
    ephemeris_info = EPHEMERIS_FILES[args.target]
    filename = ephemeris_info["url"].split("/")[-1]
    destination = output_dir / filename
    
    if destination.exists():
        print(f"File already exists: {destination}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Skipping download.")
            return
    
    download_file(ephemeris_info["url"], destination)
    print(f"\nEphemeris file saved to: {destination}")
    print(f"Set JPL_EPHEMERIS_PATH={output_dir} to use this file.")


if __name__ == "__main__":
    main()
