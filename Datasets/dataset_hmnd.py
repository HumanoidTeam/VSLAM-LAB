import os
import yaml

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class HMND_dataset(DatasetVSLAMLab):

    def __init__(self, benchmark_path):
        # Initialize the dataset (loads yaml and base fields)
        super().__init__('hmnd', benchmark_path)

        # Override sequence_names by enumerating folders present in the benchmark path
        if os.path.isdir(self.dataset_path):
            discovered = [p for p in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, p))]
            discovered.sort()
            if discovered:
                self.sequence_names = discovered
        self.sequence_nicknames = self.sequence_names

    # No download for locally generated datasets
    def download_process(self, sequence_name):
        print(SCRIPT_LABEL + f"Local dataset detected. Skipping download and generation steps for {sequence_name}.")
        return


