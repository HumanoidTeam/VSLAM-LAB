# Auto-generated stub for integrating a custom dataset into VSLAM-LAB
# Copy this file into VSLAM-LAB/Datasets/ and adapt as needed.

from pathlib import Path

class HMND_dataset:
    def __init__(self, benchmark_path: str):
        self.benchmark_path = Path(benchmark_path)
        self.dataset_root = self.benchmark_path / 'HMND'

    def get_sequences(self):
        # Return list of available sequences (directories)
        return sorted([p.name for p in self.dataset_root.iterdir() if p.is_dir()])

    # Implement other methods expected by Dataset_vslamlab if needed.
