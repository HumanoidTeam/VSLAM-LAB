import os
import yaml
import pandas as pd
from tqdm import tqdm

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from Datasets.dataset_utilities import load_rig_yaml, build_multicam_rgb_csv_rows

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class HMND_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path, dataset_name='hmnd'):
        super().__init__(dataset_name, benchmark_path)

        with open(self.yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def resolve_sequence_path(self, sequence_name: str) -> str:
        """Allow HMND to work with flattened layout without nested sequence folder.
        Prefer dataset root if nested folder doesn't exist.
        """
        nested = os.path.join(self.dataset_path, sequence_name)
        if os.path.exists(nested):
            return nested
        return os.path.join(self.dataset_path, sequence_name) if os.path.basename(self.dataset_path) != sequence_name else str(self.dataset_path)

    def _find_rig_yaml(self):
        """Return path to rig.yaml searching recursively under dataset root."""
        candidates = ['rig.yaml', 'rig.yml', 'RIG.yaml', 'RIG.yml']
        for cand in candidates:
            default_path = os.path.join(self.dataset_path, cand)
            if os.path.exists(default_path):
                return default_path
        for root, _dirs, files in os.walk(self.dataset_path):
            for cand in candidates:
                if cand in files:
                    return os.path.join(root, cand)
        return None

    def _find_file_recursive(self, filenames):
        """Search for the first match of any filename in list under dataset root."""
        for name in filenames:
            p = os.path.join(self.dataset_path, name)
            if os.path.exists(p):
                return p
        for root, _dirs, files in os.walk(self.dataset_path):
            for name in filenames:
                if name in files:
                    return os.path.join(root, name)
        return None

    def _resolve_relpath(self, primary_base, secondary_base, relpath):
        """Resolve relpath against primary base, then secondary base as fallback."""
        if os.path.isabs(relpath):
            return relpath
        cand1 = os.path.join(primary_base, relpath)
        if os.path.exists(cand1):
            return cand1
        cand2 = os.path.join(secondary_base, relpath)
        return cand2

    def download_sequence_data(self, sequence_name: str) -> None:
        return

    def create_rgb_folder(self, sequence_name: str) -> None:
        # Create convenience symlinks rgb_0, rgb_1 to selected rig cameras (if available)
        sequence_path = self.resolve_sequence_path(sequence_name)
        rig_yaml = self._find_rig_yaml()
        os.makedirs(sequence_path, exist_ok=True)
        if rig_yaml:
            rig = load_rig_yaml(rig_yaml)
            selected = rig.get('defaults', {}).get('run_cameras', [rig['cameras'][0]['id']])
            cid2cam = {c['id']: c for c in rig['cameras']}
            rig_dir = os.path.dirname(rig_yaml)

            # Create symlinks for the first two cameras if present
            for idx, cid in enumerate(sorted(selected)[:2]):
                cam = cid2cam[cid]
                target_dir = self._resolve_relpath(self.dataset_path, rig_dir, cam['data_dir'])
                link_dir = os.path.join(sequence_path, f'rgb_{idx}')
                try:
                    if os.path.islink(link_dir) or os.path.exists(link_dir):
                        continue
                    os.symlink(target_dir, link_dir)
                except FileExistsError:
                    pass

            # Also mirror the rig data_dir structure inside the sequence so paths in csv resolve
            for cid in sorted(selected):
                cam = cid2cam[cid]
                target_dir = self._resolve_relpath(self.dataset_path, rig_dir, cam['data_dir'])
                link_dir = os.path.join(sequence_path, cam['data_dir'])
                parent = os.path.dirname(link_dir)
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                try:
                    if not (os.path.islink(link_dir) or os.path.exists(link_dir)):
                        os.symlink(target_dir, link_dir)
                except FileExistsError:
                    pass
        else:
            # Fallback: if there are existing rgb_0,rgb_1 directories anywhere, link them
            rgb0 = None
            rgb1 = None
            for root, dirs, _files in os.walk(self.dataset_path):
                if 'rgb_0' in dirs and not rgb0:
                    rgb0 = os.path.join(root, 'rgb_0')
                if 'rgb_1' in dirs and not rgb1:
                    rgb1 = os.path.join(root, 'rgb_1')
                if rgb0 and rgb1:
                    break
            for idx, src in enumerate([rgb0, rgb1]):
                if src:
                    link_dir = os.path.join(sequence_path, f'rgb_{idx}')
                    try:
                        if not (os.path.islink(link_dir) or os.path.exists(link_dir)):
                            os.symlink(src, link_dir)
                    except FileExistsError:
                        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.resolve_sequence_path(sequence_name)
        rig_yaml = self._find_rig_yaml()
        if rig_yaml:
            rig = load_rig_yaml(rig_yaml)
            selected = rig.get('defaults', {}).get('run_cameras', [rig['cameras'][0]['id']])
            rig_dir = os.path.dirname(rig_yaml)

            # Prefer dataset root; if camera dirs not found there, fall back to rig dir
            rows = build_multicam_rgb_csv_rows(self.dataset_path, rig, selected)
            if len(rows) == 0:
                rows = build_multicam_rgb_csv_rows(rig_dir, rig, selected)
            if len(rows) == 0:
                print(f"{SCRIPT_LABEL}No synchronized pairs found for selected cameras {selected}")
                return

            # Build columns deterministically sorted by camera id
            cols = []
            for cid in sorted(selected):
                cols.extend([f"ts_rgb{cid} (s)", f"path_rgb{cid}"])

            df = pd.DataFrame(rows)
            df = df[cols]
            rgb_csv = os.path.join(sequence_path, 'rgb.csv')
            os.makedirs(sequence_path, exist_ok=True)
            df.to_csv(rgb_csv, index=False)
        else:
            # Fallback: copy existing rgb.csv if found anywhere under dataset
            src_rgb = self._find_file_recursive(['rgb.csv'])
            if src_rgb:
                dst_rgb = os.path.join(sequence_path, 'rgb.csv')
                try:
                    if os.path.exists(dst_rgb):
                        os.remove(dst_rgb)
                except FileNotFoundError:
                    pass
                import shutil
                shutil.copy(src_rgb, dst_rgb)
            else:
                print(f"{SCRIPT_LABEL}No rig.yaml or rgb.csv found under {self.dataset_path}. Skipping rgb.csv generation.")
                return

    def create_imu_csv(self, sequence_name: str) -> None:
        sequence_path = self.resolve_sequence_path(sequence_name)
        rig_yaml = self._find_rig_yaml()
        if rig_yaml:
            rig = load_rig_yaml(rig_yaml)
            rig_dir = os.path.dirname(rig_yaml)
            imu_src = self._resolve_relpath(self.dataset_path, rig_dir, rig['imu']['csv'])
        else:
            imu_src = self._find_file_recursive(['imu.csv'])
            if not imu_src:
                print(f"{SCRIPT_LABEL}No rig.yaml or imu.csv found under {self.dataset_path}. Skipping imu.csv link.")
                return
        imu_dst = os.path.join(sequence_path, 'imu.csv')
        os.makedirs(sequence_path, exist_ok=True)
        # Create a real imu.csv aligned to camera time if large offset detected
        import pandas as pd
        import numpy as np
        rgb_csv = os.path.join(sequence_path, 'rgb.csv')
        try:
            df_rgb = pd.read_csv(rgb_csv)
            # pick first available camera ts column
            ts_cols = [c for c in df_rgb.columns if c.startswith('ts_rgb')]
            r0 = float(df_rgb[ts_cols[0]].iloc[0]) if len(ts_cols) else None
        except Exception:
            r0 = None

        try:
            df_imu = pd.read_csv(imu_src)
            t_col = df_imu.columns[0]
            # Parse timestamps as numeric without prematurely casting to float
            t_series = pd.to_numeric(df_imu.iloc[:, 0], errors='coerce')
            # Detect nanosecond scale by median magnitude
            if t_series.notna().any() and float(t_series.median()) > 1e12:
                # Convert ns -> s with proper decimal precision to match RGB timestamps
                t_vals = t_series.astype(np.float64) / 1e9
                # Round to 6 decimal places to match RGB precision
                t_vals = np.round(t_vals, 6)
            else:
                # Already seconds (float or int)
                t_vals = t_series.astype(np.float64)
            i0 = float(t_vals.iloc[0])
        except Exception:
            i0 = None

        # Always convert timestamps to seconds and create a new CSV file
        df_imu_new = df_imu.copy()
        df_imu_new.iloc[:,0] = t_vals
        
        # Ensure header reflects seconds if original claimed ns
        if '[ns]' in str(t_col).lower() or 'ns' in str(t_col).lower():
            new_col = str(t_col).replace('[ns]', '[s]').replace('ns', 's')
            df_imu_new.rename(columns={t_col: new_col}, inplace=True)
        
        # If both available and offset is too large, shift IMU timestamps.
        # Make IMU start slightly BEFORE RGB (by 50 ms) to satisfy OKVIS startup.
        if r0 is not None and i0 is not None and abs(i0 - r0) > 1.0:
            delta = i0 - r0  # positive if IMU is newer (lags behind images)
            # We want imu_ts_shifted_seconds = r0 - 0.05 at the first row
            shift = delta + 0.05
            t_shifted = t_vals - shift
            df_imu_new.iloc[:,0] = t_shifted
        
        # Always write the converted CSV file
        df_imu_new.to_csv(imu_dst, index=False)
        
        # Validate timestamp format consistency between RGB and IMU
        try:
            df_rgb_check = pd.read_csv(rgb_csv)
            df_imu_check = pd.read_csv(imu_dst)
            
            # Get first timestamps from both files
            rgb_ts_cols = [c for c in df_rgb_check.columns if c.startswith('ts_rgb')]
            if rgb_ts_cols:
                rgb_first_ts = float(df_rgb_check[rgb_ts_cols[0]].iloc[0])
                imu_first_ts = float(df_imu_check.iloc[0, 0])
                
                # Check if timestamps are in similar magnitude (both seconds)
                if abs(rgb_first_ts - imu_first_ts) > 1e6:  # More than 1M difference suggests unit mismatch
                    print(f"{SCRIPT_LABEL}WARNING: RGB and IMU timestamp units may be inconsistent!")
                    print(f"  RGB first timestamp: {rgb_first_ts}")
                    print(f"  IMU first timestamp: {imu_first_ts}")
                    print(f"  Difference: {abs(rgb_first_ts - imu_first_ts)}")
                else:
                    print(f"{SCRIPT_LABEL}✓ RGB and IMU timestamps are consistent (both in seconds)")
        except Exception as e:
            print(f"{SCRIPT_LABEL}Could not validate timestamp consistency: {e}")

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.resolve_sequence_path(sequence_name)
        rig_yaml = self._find_rig_yaml()
        if not rig_yaml:
            # Fallback: copy existing calibration.yaml if present
            src_cal = self._find_file_recursive(['calibration.yaml'])
            if src_cal:
                dst_cal = os.path.join(sequence_path, 'calibration.yaml')
                import shutil
                shutil.copy(src_cal, dst_cal)
            else:
                print(f"{SCRIPT_LABEL}No rig.yaml or calibration.yaml found under {self.dataset_path}. Skipping calibration.yaml generation.")
            return

        rig = load_rig_yaml(rig_yaml)
        selected = rig.get('defaults', {}).get('run_cameras', [rig['cameras'][0]['id']])
        cid2cam = {c['id']: c for c in rig['cameras']}
        rig_dir = os.path.dirname(rig_yaml)

        # Build camera parameter list in selected id order
        cameras = []
        for cid in sorted(selected):
            cam = cid2cam[cid]
            cam_params = {
                'model': cam['model'],
                'fx': cam['intrinsics'][0], 'fy': cam['intrinsics'][1],
                'cx': cam['intrinsics'][2], 'cy': cam['intrinsics'][3],
                'k1': cam['distortion_coeffs'][0], 'k2': cam['distortion_coeffs'][1],
                'p1': cam['distortion_coeffs'][2], 'p2': cam['distortion_coeffs'][3],
                'k3': cam['distortion_coeffs'][4] if len(cam['distortion_coeffs']) > 4 else 0.0,
                'w': cam['resolution'][0], 'h': cam['resolution'][1],
                'distortion_model': 'radtan'
            }
            cameras.append(cam_params)

        # Build IMU transforms mapping to Camera<i> index order
        imu_transforms = {}
        for idx, cid in enumerate(sorted(selected)):
            cam = cid2cam[cid]
            # 0-indexed key (e.g., T_b_c0, T_b_c1)
            imu_transforms[f"T_b_c{idx}"] = cam['T_b_c']
            # 1-indexed duplicate (e.g., T_b_c1, T_b_c2) for OKVIS variants
            imu_transforms[f"T_b_c{idx+1}"] = cam['T_b_c']

        # IMU noise
        imu = {
            'gyro_noise': rig['imu']['noise'].get('gyroscope_noise_density', 0.0),
            'accel_noise': rig['imu']['noise'].get('accelerometer_noise_density', 0.0),
            'gyro_bias': rig['imu']['noise'].get('gyroscope_random_walk', 0.0),
            'accel_bias': rig['imu']['noise'].get('accelerometer_random_walk', 0.0),
            'frequency': rig['imu'].get('frequency', self.rgb_hz)
        }

        # Write standard VSLAM-LAB style
        self.write_calibration_yaml(sequence_name=sequence_name, cameras=cameras, imu=imu, imu_transforms=imu_transforms)

        # Also append camchain-style blocks expected by some OKVIS2 builds
        calib_path = os.path.join(sequence_path, 'calibration.yaml')
        with open(calib_path, 'a') as f:
            f.write("\n")
            for idx, cid in enumerate(sorted(selected)):
                cam = cid2cam[cid]
                fx, fy, cx, cy = cam['intrinsics']
                k1, k2, p1, p2 = cam['distortion_coeffs'][:4]
                w, h = cam['resolution']
                # Build T_BS array
                T = cam['T_b_c']
                if len(T) == 16:
                    flat = T
                else:
                    flat = [item for row in T for item in row]
                f.write(f"cam{idx}:\n")
                f.write(f"  T_BS: [{', '.join(map(str, flat))}]\n")
                f.write(f"  image_dimension: [{w}, {h}]\n")
                f.write(f"  distortion_coefficients: [{k1}, {k2}, {p1}, {p2}]\n")
                f.write(f"  focal_length: [{fx}, {fy}]\n")
                f.write(f"  principal_point: [{cx}, {cy}]\n")
                f.write(f"  distortion_type: radialtangential\n\n")


    def create_groundtruth_csv(self, sequence_name: str) -> None:
        return

    def remove_unused_files(self, sequence_name: str) -> None:
        return



