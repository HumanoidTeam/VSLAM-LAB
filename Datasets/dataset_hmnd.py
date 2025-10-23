import os
import yaml
from typing import List

# Reuse VSLAM-LAB base class; when imported via get_dataset.py this module path is added to sys.path
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab


class HMND_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path: str):
        # Initialize using VSLAM-LAB base to set dataset_label, colors, paths, etc.
        super().__init__('hmnd', benchmark_path)

        # Prefer local YAML (next to this file) to override rgb_hz and sequences if provided
        scripts_yaml = os.path.join(os.path.dirname(__file__), 'dataset_hmnd.yaml')
        if os.path.exists(scripts_yaml):
            with open(scripts_yaml, 'r') as f:
                data = yaml.safe_load(f) or {}
            self.rgb_hz = float(data.get('rgb_hz', self.rgb_hz))
            self.modes = data.get('modes', getattr(self, 'modes', 'mono-vi'))
            seqs: List[str] = list(data.get('sequence_names', []) or [])
            if not seqs and os.path.isdir(self.dataset_path):
                seqs = [d for d in sorted(os.listdir(self.dataset_path)) if os.path.isdir(os.path.join(self.dataset_path, d))]
            if seqs:
                self.sequence_names = seqs
        else:
            # As a fallback, auto-discover if base YAML had empty list
            if (not getattr(self, 'sequence_names', [])) and os.path.isdir(self.dataset_path):
                self.sequence_names = [d for d in sorted(os.listdir(self.dataset_path)) if os.path.isdir(os.path.join(self.dataset_path, d))]

        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    # No download: data already present under benchmark path
    def download_sequence_data(self, sequence_name):
        return

    def create_rgb_folder(self, sequence_name):
        # Images expected at <dataset_path>/<sequence>/rgb written by exporter
        return

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        
        # Check if this is the new structure (camera folders under sequence)
        camera_folders = [d for d in os.listdir(sequence_path) 
                         if os.path.isdir(os.path.join(sequence_path, d)) 
                         and os.path.isdir(os.path.join(sequence_path, d, 'rgb'))]
        
        if camera_folders:
            # New structure: create rgb.txt for each camera folder
            for camera_name in camera_folders:
                camera_path = os.path.join(sequence_path, camera_name)
                rgb_path = os.path.join(camera_path, 'rgb')
                rgb_txt = os.path.join(camera_path, 'rgb.txt')
                if os.path.exists(rgb_txt):
                    continue
                if not os.path.isdir(rgb_path):
                    continue

                files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
                files.sort()
                frame_dt = 1.0 / max(1e-6, self.rgb_hz)
                with open(rgb_txt, 'w') as f:
                    for i, name in enumerate(files):
                        ts = i * frame_dt
                        f.write(f"{ts:.9f} rgb/{name}\n")
        else:
            # Old structure: single rgb folder at sequence level
            rgb_path = os.path.join(sequence_path, 'rgb')
            rgb_txt = os.path.join(sequence_path, 'rgb.txt')
            if os.path.exists(rgb_txt):
                return
            if not os.path.isdir(rgb_path):
                return

            files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
            files.sort()
            frame_dt = 1.0 / max(1e-6, self.rgb_hz)
            with open(rgb_txt, 'w') as f:
                for i, name in enumerate(files):
                    ts = i * frame_dt
                    f.write(f"{ts:.9f} rgb/{name}\n")

    def create_imu_csv(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        imu_csv = os.path.join(sequence_path, 'imu.csv')
        if os.path.exists(imu_csv):
            return
        os.makedirs(sequence_path, exist_ok=True)
        with open(imu_csv, 'w') as f:
            f.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],')
            f.write('a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calib = os.path.join(sequence_path, 'calibration.yaml')
        if os.path.exists(calib):
            return
        camera0 = {
            'model': 'UNKNOWN',
            'fx': 0.0, 'fy': 0.0, 'cx': 0.0, 'cy': 0.0,
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0,
        }
        imu = {
            'transform': [1.0, 0.0, 0.0, 0.0,
                          0.0, 1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0],
            'gyro_noise': 1.6e-4,
            'accel_noise': 2.8e-3,
            'gyro_bias': 2.2e-5,
            'accel_bias': 8.6e-4,
            'frequency': float(self.rgb_hz),
        }
        self.write_calibration_yaml(sequence_name, camera0=camera0, imu=imu)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        gt = os.path.join(sequence_path, 'groundtruth.txt')
        if os.path.exists(gt):
            return
        with open(gt, 'w') as f:
            f.write('# Ground truth not available; placeholder file.\n')

    def create_rgb_csv(self, sequence_name):
        """Create rgb.csv for VSLAM-LAB multicam format from active rig configuration."""
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rig_path = os.path.join(sequence_path, 'rig.yaml')
        
        if not os.path.exists(rig_path):
            print(f"Warning: No active rig found at {rig_path}")
            return
        
        # Load the active rig configuration
        with open(rig_path, 'r') as f:
            rig = yaml.safe_load(f) or {}
        
        if 'cameras' not in rig:
            print(f"Warning: No cameras found in rig configuration")
            return
        
        # Get selected cameras from rig defaults
        selected_cameras = rig.get('defaults', {}).get('run_cameras', [])
        if not selected_cameras:
            # Fallback: use all cameras
            selected_cameras = [cam['id'] for cam in rig['cameras']]
        
        # Use VSLAM-LAB utility to build multicam RGB CSV
        from Datasets.dataset_utilities import build_multicam_rgb_csv_rows
        from pathlib import Path
        
        try:
            print(f"Debug: sequence_path = {sequence_path}")
            print(f"Debug: selected_cameras = {selected_cameras}")
            print(f"Debug: rig cameras = {[cam['id'] for cam in rig['cameras']]}")
            rows = build_multicam_rgb_csv_rows(Path(sequence_path), rig, selected_cameras)
            
            # Write rgb.csv
            rgb_csv_path = os.path.join(sequence_path, 'rgb.csv')
            with open(rgb_csv_path, 'w') as f:
                # Write header
                header_parts = ['ts_rgbi (s)']
                for cam_id in selected_cameras:
                    header_parts.append(f'path_rgbi_{cam_id}')
                f.write(','.join(header_parts) + '\n')
                
                # Write data rows
                for row in rows:
                    row_parts = [str(row['ts_rgbi (s)'])]
                    for cam_id in selected_cameras:
                        row_parts.append(str(row.get(f'path_rgbi_{cam_id}', '')))
                    f.write(','.join(row_parts) + '\n')
            
            print(f"Created rgb.csv with {len(rows)} synchronized frames for {len(selected_cameras)} cameras")
            
        except Exception as e:
            print(f"Error creating rgb.csv: {e}")
            # Create a minimal rgb.csv as fallback
            rgb_csv_path = os.path.join(sequence_path, 'rgb.csv')
            with open(rgb_csv_path, 'w') as f:
                f.write('ts_rgbi (s),path_rgbi_0\n')

    def remove_unused_files(self, sequence_name):
        return


