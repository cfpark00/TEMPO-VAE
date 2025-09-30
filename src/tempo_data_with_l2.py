"""
PyTorch dataset for TEMPO tiles with aligned L2 products.
Uses IterableDataset with RandomBuffer like the original.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict
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


class TEMPODatasetWithL2(IterableDataset):
    """Iterable dataset for TEMPO tiles with L2 products using random buffering."""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        min_buffer_size: int = 200,
        verbose: bool = True
    ):
        """
        Args:
            data_dir: Base directory containing tiles with L2 subfolders
            split: 'train' or 'val'
            min_buffer_size: Minimum buffer size before yielding samples
            verbose: Show loading progress
        """
        self.data_dir = Path(data_dir) / split
        self.min_buffer_size = min_buffer_size
        self.verbose = verbose

        if not self.data_dir.exists():
            raise FileNotFoundError(f"FATAL: Data directory not found: {self.data_dir}")

        # Get list of spectral tile files
        self.tile_files = sorted(self.data_dir.glob("*.pt"))
        if not self.tile_files:
            raise ValueError(f"FATAL: No .pt files found in {self.data_dir}")

        # Check for L2 subdirectories
        self.l2_products = ['NO2', 'O3TOT', 'HCHO', 'CLDO4']
        self.l2_dirs = {}
        for product in self.l2_products:
            l2_dir = self.data_dir / f'l2_{product}'
            if not l2_dir.exists():
                raise FileNotFoundError(f"FATAL: L2 directory not found: {l2_dir}")
            self.l2_dirs[product] = l2_dir

        self.buffer = RandomBuffer()
        self.tiles_per_file = 64

        # Initial buffer fill
        if verbose:
            pbar = tqdm(total=min_buffer_size, desc=f"Loading initial buffer ({split})")

        while len(self.buffer) < self.min_buffer_size:
            self.load_file(np.random.randint(0, len(self.tile_files)))
            if verbose:
                pbar.n = len(self.buffer)
                pbar.refresh()

        if verbose:
            pbar.close()
            print(f"Loaded {split} dataset with {len(self.tile_files)} files")

    def load_file(self, file_idx: int):
        """Load tiles from a file into buffer."""
        spectral_path = self.tile_files[file_idx]

        # Load spectral batch
        spectral_batch = torch.load(spectral_path, weights_only=True)

        # Load L2 batches
        l2_batches = {}
        for product in self.l2_products:
            l2_path = self.l2_dirs[product] / spectral_path.name
            if not l2_path.exists():
                raise FileNotFoundError(f"FATAL: L2 file not found: {l2_path}")
            l2_batches[product] = torch.load(l2_path, weights_only=True)

        # Add each tile to buffer as a dictionary
        for tile_idx in range(self.tiles_per_file):
            # Get spectral tile and convert to channel-first [C, H, W]
            spectral_tile = spectral_batch[tile_idx].cpu()  # Currently [H, W, C] = [64, 64, 1028]
            if spectral_tile.dim() == 3 and spectral_tile.shape[-1] == 1028:
                spectral_tile = spectral_tile.permute(2, 0, 1)  # Convert to [1028, 64, 64]

            tile_dict = {
                'spectral': spectral_tile  # Now [1028, 64, 64]
            }
            for product in self.l2_products:
                tile_dict[product] = l2_batches[product][tile_idx].cpu()  # [64, 64]

            self.buffer.put(tile_dict)

    def get_data(self):
        """Get single data sample."""
        tile_dict = self.buffer.get()

        # Refill buffer if needed
        while len(self.buffer) < self.min_buffer_size:
            self.load_file(np.random.randint(0, len(self.tile_files)))

        return tile_dict

    def __iter__(self):
        """Iterate over samples."""
        while True:
            yield self.get_data()


class TEMPODataLoaderWithL2:
    """DataLoader wrapper for TEMPO tiles with L2 products."""

    @staticmethod
    def get_dataloader(
        data_dir: str,
        split: str = 'train',
        batch_size: int = 32,
        num_workers: int = 4,
        min_buffer_size: int = 200,
        verbose: bool = True
    ) -> DataLoader:
        """
        Create DataLoader for TEMPO tiles with L2 products.
        Matches the interface of the original TEMPODataLoader.

        Args:
            data_dir: Base directory containing tiles
            split: 'train' or 'val'
            batch_size: Batch size
            num_workers: Number of data loading workers
            min_buffer_size: Minimum buffer size before yielding samples
            verbose: Show loading progress

        Returns:
            DataLoader instance
        """
        dataset = TEMPODatasetWithL2(
            data_dir=data_dir,
            split=split,
            min_buffer_size=min_buffer_size,
            verbose=verbose
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )