"""
Data loading utilities for TEMPO tiles.
"""

import torch
import numpy as np
from pathlib import Path
import glob
from typing import Optional, List
from tqdm import tqdm


class RandomBuffer:
    """Random buffer for efficient sampling from tiles."""

    def __init__(self):
        self.buffer = []

    def put(self, item):
        """Add item to buffer."""
        self.buffer.append(item)

    def get(self):
        """Remove and return random item from buffer."""
        if not self.buffer:
            raise IndexError("Buffer is empty")
        index = np.random.randint(0, len(self.buffer))
        return self.buffer.pop(index)

    def __len__(self):
        return len(self.buffer)


class TEMPODataset(torch.utils.data.IterableDataset):
    """Dataset for TEMPO tiles with random buffering."""

    def __init__(
        self,
        data_dir: str,
        min_buffer_size: int = 200,
        verbose: bool = True
    ):
        """
        Initialize TEMPO dataset.

        Args:
            data_dir: Directory containing .pt tile files
            min_buffer_size: Minimum buffer size before yielding samples
            verbose: Show loading progress
        """
        self.data_dir = Path(data_dir)
        self.min_buffer_size = min_buffer_size
        self.verbose = verbose

        # Find all .pt files
        self.files = sorted(glob.glob(str(self.data_dir / "*.pt")))
        if not self.files:
            raise ValueError(f"No .pt files found in {data_dir}")

        self.buffer = RandomBuffer()

        # Initial buffer fill
        if verbose:
            pbar = tqdm(total=min_buffer_size, desc="Loading initial buffer")

        while len(self.buffer) < self.min_buffer_size:
            self.load_file(np.random.randint(0, len(self.files)))
            if verbose:
                pbar.n = len(self.buffer)
                pbar.refresh()

        if verbose:
            pbar.close()
            print(f"Loaded dataset from {data_dir} with {len(self.files)} files")

    def load_file(self, file_idx: int):
        """Load tiles from a file into buffer."""
        tiles = torch.load(self.files[file_idx], weights_only=False)

        # Handle both single tile and batch of tiles
        if tiles.dim() == 3:
            # Single tile: [H, W, C]
            self.buffer.put(tiles.cpu())
        else:
            # Batch of tiles: [N, H, W, C]
            for tile in tiles:
                self.buffer.put(tile.cpu())

    def get_data(self):
        """Get single data sample."""
        tile = self.buffer.get()

        # Refill buffer if needed
        while len(self.buffer) < self.min_buffer_size:
            self.load_file(np.random.randint(0, len(self.files)))

        # Convert to channel-first format: [C, H, W]
        if tile.dim() == 3:
            tile = tile.permute(2, 0, 1)

        return tile

    def __iter__(self):
        """Iterate infinitely over dataset."""
        while True:
            yield self.get_data()


class TEMPODataLoader:
    """Simplified data loader for TEMPO tiles."""

    @staticmethod
    def get_dataloader(
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        min_buffer_size: int = 200,
        verbose: bool = True
    ) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for TEMPO tiles.

        Args:
            data_dir: Directory containing .pt files
            batch_size: Batch size
            num_workers: Number of worker processes
            min_buffer_size: Minimum buffer size for dataset
            verbose: Show loading progress

        Returns:
            DataLoader instance
        """
        dataset = TEMPODataset(
            data_dir=data_dir,
            min_buffer_size=min_buffer_size,
            verbose=verbose
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )

        return dataloader


def load_normalization_stats(stats_dir: str) -> tuple:
    """
    Load normalization statistics.

    Args:
        stats_dir: Directory containing mean_spectrum.pt and std_spectrum.pt

    Returns:
        Tuple of (mean_spectrum, std_spectrum) tensors
    """
    stats_dir = Path(stats_dir)
    mean_path = stats_dir / "mean_spectrum.pt"
    std_path = stats_dir / "std_spectrum.pt"

    if not mean_path.exists():
        raise FileNotFoundError(f"Mean file not found: {mean_path}")
    if not std_path.exists():
        raise FileNotFoundError(f"Std file not found: {std_path}")

    mean_spectrum = torch.load(mean_path, weights_only=False)
    std_spectrum = torch.load(std_path, weights_only=False)

    return mean_spectrum, std_spectrum