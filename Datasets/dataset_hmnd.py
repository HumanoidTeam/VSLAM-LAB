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
            # Fallback: check for head_rear_left/rgb and head_rear_right/rgb in sequence path
            head_left_rgb = os.path.join(sequence_path, 'head_rear_left', 'rgb')
            head_right_rgb = os.path.join(sequence_path, 'head_rear_right', 'rgb')
            
            # Create symlinks for head_rear_left/rgb -> rgb_0 and head_rear_right/rgb -> rgb_1
            if os.path.exists(head_left_rgb):
                link_dir = os.path.join(sequence_path, 'rgb_0')
                try:
                    if os.path.islink(link_dir):
                        # Check if it points to the right target
                        current_target = os.readlink(link_dir)
                        if current_target != 'head_rear_left/rgb':
                            os.unlink(link_dir)
                            os.symlink('head_rear_left/rgb', link_dir)
                    elif os.path.exists(link_dir):
                        # Don't overwrite existing directories
                        pass
                    else:
                        # Use relative path for portability
                        os.symlink('head_rear_left/rgb', link_dir)
                except (FileExistsError, OSError) as e:
                    # If symlink creation failed, try to recreate it
                    try:
                        if os.path.islink(link_dir):
                            os.unlink(link_dir)
                        os.symlink('head_rear_left/rgb', link_dir)
                    except:
                        pass
            
            if os.path.exists(head_right_rgb):
                link_dir = os.path.join(sequence_path, 'rgb_1')
                try:
                    if os.path.islink(link_dir):
                        # Check if it points to the right target
                        current_target = os.readlink(link_dir)
                        if current_target != 'head_rear_right/rgb':
                            os.unlink(link_dir)
                            os.symlink('head_rear_right/rgb', link_dir)
                    elif os.path.exists(link_dir):
                        # Don't overwrite existing directories
                        pass
                    else:
                        # Use relative path for portability
                        os.symlink('head_rear_right/rgb', link_dir)
                except (FileExistsError, OSError) as e:
                    # If symlink creation failed, try to recreate it
                    try:
                        if os.path.islink(link_dir):
                            os.unlink(link_dir)
                        os.symlink('head_rear_right/rgb', link_dir)
                    except:
                        pass
            
            # Additional fallback: if there are existing rgb_0,rgb_1 directories anywhere, link them
            if not os.path.exists(os.path.join(sequence_path, 'rgb_0')) or not os.path.exists(os.path.join(sequence_path, 'rgb_1')):
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
                
                # Read the source CSV
                df = pd.read_csv(src_rgb)
                
                # Update paths if they use head_rear_left/rgb or head_rear_right/rgb
                path_columns = [col for col in df.columns if col.startswith('path_')]
                for col in path_columns:
                    if df[col].dtype == 'object':  # String column
                        df[col] = df[col].str.replace('head_rear_left/rgb/', 'rgb_0/', regex=False)
                        df[col] = df[col].str.replace('head_rear_right/rgb/', 'rgb_1/', regex=False)
                
                # Write the updated CSV
                os.makedirs(sequence_path, exist_ok=True)
                df.to_csv(dst_rgb, index=False)
            else:
                print(f"{SCRIPT_LABEL}No rig.yaml or rgb.csv found under {self.dataset_path}. Skipping rgb.csv generation.")
                return

    def estimate_rig_imu_biases(self, calibration_sequence_name: str = None) -> tuple:
        """Estimate stationary IMU biases from a calibration sequence (e.g., 15 min stationary rig).
        
        Returns:
            tuple: (gyro_bias, accel_bias) or (None, None) if estimation fails
        """
        # Check for calibration sequence in dataset yaml
        if calibration_sequence_name is None:
            with open(self.yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                calibration_sequence_name = data.get('calibration_sequence', None)
        
        if calibration_sequence_name is None:
            return None, None
        
        calib_sequence_path = self.resolve_sequence_path(calibration_sequence_name)
        calib_imu_csv = os.path.join(calib_sequence_path, 'imu.csv')
        
        if not os.path.exists(calib_imu_csv):
            print(f"{SCRIPT_LABEL}WARNING: Calibration sequence '{calibration_sequence_name}' not found or missing imu.csv")
            return None, None
        
        try:
            import pandas as pd
            import numpy as np
            
            df_imu = pd.read_csv(calib_imu_csv)
            accel_cols = [c for c in df_imu.columns if 'a_' in c.lower() or 'accel' in c.lower()]
            gyro_cols = [c for c in df_imu.columns if 'w_' in c.lower() or 'gyro' in c.lower() or 'omega' in c.lower()]
            
            if len(accel_cols) < 3 or len(gyro_cols) < 3:
                print(f"{SCRIPT_LABEL}WARNING: Calibration sequence missing required IMU columns")
                return None, None
            
            # Use entire stationary sequence for accurate bias estimation
            gyro_mean = [float(df_imu[gyro_cols[i]].mean()) for i in range(3)]
            gyro_std = [float(df_imu[gyro_cols[i]].std()) for i in range(3)]
            accel_mean = [float(df_imu[accel_cols[i]].mean()) for i in range(3)]
            accel_std = [float(df_imu[accel_cols[i]].std()) for i in range(3)]
            
            # Check if data looks stationary (low gyro std, reasonable accel std)
            gyro_std_max = max(gyro_std)
            accel_std_max = max(accel_std)
            gravity_mag = np.linalg.norm(accel_mean)
            
            if gyro_std_max > 0.1 or accel_std_max > 2.0:
                print(f"{SCRIPT_LABEL}WARNING: Calibration sequence may not be stationary")
                print(f"  Gyro std: {gyro_std_max:.4f} rad/s (expected < 0.1)")
                print(f"  Accel std: {accel_std_max:.3f} m/s² (expected < 2.0)")
                return None, None
            
            if abs(gravity_mag - 9.81) > 3.0:
                print(f"{SCRIPT_LABEL}WARNING: Gravity magnitude ({gravity_mag:.2f} m/s²) differs from expected 9.81 m/s²")
                return None, None
            
            # Estimate biases: when stationary, gyro should be near zero (bias), accel = gravity + bias
            gyro_bias = gyro_mean
            
            # Find which axis has gravity (the one with largest magnitude)
            gravity_axis = np.argmax(np.abs(accel_mean))
            gravity_sign = np.sign(accel_mean[gravity_axis])
            expected_gravity = 9.81 * gravity_sign
            
            # Compute bias: accel = gravity + bias, so bias = accel - gravity
            # Gravity is on the axis with largest magnitude
            accel_bias = [accel_mean[0], accel_mean[1], accel_mean[2]]
            accel_bias[gravity_axis] = accel_mean[gravity_axis] - expected_gravity
            
            print(f"{SCRIPT_LABEL}✓ Calibrated IMU biases from stationary sequence '{calibration_sequence_name}':")
            print(f"  Gyro bias (g0): [{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, {gyro_bias[2]:.6f}] rad/s")
            print(f"  Accel bias (a0): [{accel_bias[0]:.3f}, {accel_bias[1]:.3f}, {accel_bias[2]:.3f}] m/s²")
            print(f"  (Using {len(df_imu)} IMU measurements from {calibration_sequence_name})")
            
            return gyro_bias, accel_bias
        except Exception as e:
            print(f"{SCRIPT_LABEL}WARNING: Failed to estimate rig IMU biases: {e}")
            return None, None

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
        
        # Downsample IMU data to match EUROC frequency (200 Hz) to avoid numerical issues
        # HMND has 500 Hz IMU which causes too many measurements per frame (34 vs 6 for EUROC)
        target_imu_hz = 200.0  # Match EUROC frequency
        if self.imu_hz > target_imu_hz:
            downsample_factor = int(self.imu_hz / target_imu_hz)
            if downsample_factor > 1:
                df_imu_new = df_imu_new.iloc[::downsample_factor].reset_index(drop=True)
                # Update timestamps after downsampling
                t_vals = df_imu_new.iloc[:, 0].values
                i0 = float(t_vals[0]) if len(t_vals) > 0 else i0
                print(f"{SCRIPT_LABEL}Downsampled IMU data from {self.imu_hz} Hz to ~{self.imu_hz/downsample_factor:.1f} Hz (factor: {downsample_factor})")
        
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
        
        # Estimate stationary IMU biases - prefer rig-level calibration from dedicated calibration sequence
        # If not available, use first N seconds of current sequence
        stationary_bias_gyro, stationary_bias_accel = self.estimate_rig_imu_biases()
        
        # Fallback: estimate from first N seconds if rig-level calibration not available
        if stationary_bias_gyro is None or stationary_bias_accel is None:
            try:
                accel_cols = [c for c in df_imu_new.columns if 'a_' in c.lower() or 'accel' in c.lower()]
                gyro_cols = [c for c in df_imu_new.columns if 'w_' in c.lower() or 'gyro' in c.lower() or 'omega' in c.lower()]
                t_col = df_imu_new.columns[0]
                
                if len(accel_cols) >= 3 and len(gyro_cols) >= 3:
                    # Use first N seconds for stationary bias estimation (typically 2-5 seconds)
                    # When device is stationary, gyro should be near zero, accel should be gravity + bias
                    stationary_period_seconds = 3.0  # Use first 3 seconds for stationary estimation
                    t_start = float(df_imu_new.iloc[0, 0])
                    t_end_stationary = t_start + stationary_period_seconds
                    
                    # Filter IMU data from stationary period
                    df_stationary = df_imu_new[df_imu_new[t_col] <= t_end_stationary].copy()
                    
                    if len(df_stationary) > 10:  # Need enough samples
                        # Compute statistics for stationary period
                        gyro_mean = [float(df_stationary[gyro_cols[i]].mean()) for i in range(3)]
                        gyro_std = [float(df_stationary[gyro_cols[i]].std()) for i in range(3)]
                        accel_mean = [float(df_stationary[accel_cols[i]].mean()) for i in range(3)]
                        accel_std = [float(df_stationary[accel_cols[i]].std()) for i in range(3)]
                        
                        # Check if data looks stationary (low gyro std, reasonable accel std)
                        gyro_std_max = max(gyro_std)
                        accel_std_max = max(accel_std)
                        
                        if gyro_std_max < 0.1 and accel_std_max < 2.0:  # Reasonable thresholds for stationary
                            stationary_bias_gyro = gyro_mean
                            # For accelerometer: when stationary, accel = gravity + bias
                            # Gravity is on Z-axis (up) ~9.81 m/s², so bias_z = accel_z - g
                            gravity_mag = np.linalg.norm(accel_mean)
                            gravity_axis = np.argmax(np.abs(accel_mean))
                            
                            if abs(gravity_mag - 9.81) < 3.0:  # Gravity magnitude is reasonable
                                # Find which axis has gravity (the one with largest magnitude)
                                gravity_axis = np.argmax(np.abs(accel_mean))
                                gravity_sign = np.sign(accel_mean[gravity_axis])
                                expected_gravity = 9.81 * gravity_sign
                                
                                # Compute bias: accel = gravity + bias, so bias = accel - gravity
                                stationary_bias_accel = [accel_mean[0], accel_mean[1], accel_mean[2]]
                                stationary_bias_accel[gravity_axis] = accel_mean[gravity_axis] - expected_gravity
                                
                                print(f"{SCRIPT_LABEL}✓ Estimated stationary IMU biases from first {stationary_period_seconds}s:")
                                print(f"  Gyro bias (g0): [{stationary_bias_gyro[0]:.6f}, {stationary_bias_gyro[1]:.6f}, {stationary_bias_gyro[2]:.6f}] rad/s")
                                print(f"  Accel bias (a0): [{stationary_bias_accel[0]:.3f}, {stationary_bias_accel[1]:.3f}, {stationary_bias_accel[2]:.3f}] m/s²")
                                print(f"  (These should be consistent across sequences from the same rig)")
                            else:
                                print(f"{SCRIPT_LABEL}WARNING: Gravity magnitude ({gravity_mag:.2f} m/s²) differs from expected 9.81 m/s²")
                                print(f"  Cannot estimate stationary biases reliably")
                        else:
                            print(f"{SCRIPT_LABEL}WARNING: First {stationary_period_seconds}s may not be stationary")
                            print(f"  Gyro std: {gyro_std_max:.4f} rad/s (expected < 0.1), Accel std: {accel_std_max:.3f} m/s² (expected < 2.0)")
                    else:
                        print(f"{SCRIPT_LABEL}WARNING: Not enough IMU samples in first {stationary_period_seconds}s for stationary bias estimation")
            except Exception as e:
                print(f"{SCRIPT_LABEL}WARNING: Failed to estimate stationary biases: {e}")
        
        # Store stationary biases in calibration.yaml if available
        if stationary_bias_gyro is not None and stationary_bias_accel is not None:
            calib_path = os.path.join(sequence_path, 'calibration.yaml')
            if os.path.exists(calib_path):
                try:
                    import yaml
                    with open(calib_path, 'r') as f:
                        calib_data = yaml.safe_load(f)
                    
                    # Add initial biases to IMU section
                    if 'IMU' not in calib_data:
                        calib_data['IMU'] = {}
                    
                    calib_data['IMU']['InitialGyroBias'] = stationary_bias_gyro
                    calib_data['IMU']['InitialAccelBias'] = stationary_bias_accel
                    
                    # Write back to file
                    with open(calib_path, 'w') as f:
                        yaml.dump(calib_data, f, default_flow_style=False, sort_keys=False)
                    
                    print(f"{SCRIPT_LABEL}✓ Stored stationary IMU biases in calibration.yaml")
                except Exception as e:
                    print(f"{SCRIPT_LABEL}WARNING: Failed to store biases in calibration.yaml: {e}")
        
        # Diagnostic: Check IMU data statistics for OKVIS2 compatibility
        try:
            # Compute mean accelerometer values to check gravity direction
            accel_cols = [c for c in df_imu_new.columns if 'a_' in c.lower() or 'accel' in c.lower()]
            gyro_cols = [c for c in df_imu_new.columns if 'w_' in c.lower() or 'gyro' in c.lower() or 'omega' in c.lower()]
            
            if len(accel_cols) >= 3:
                accel_mean = [float(df_imu_new[accel_cols[i]].mean()) for i in range(3)]
                accel_std = [float(df_imu_new[accel_cols[i]].std()) for i in range(3)]
                # Check if gravity magnitude is reasonable (should be ~9.8 m/s²)
                gravity_mag = np.linalg.norm(accel_mean)
                
                # Check for suspiciously large accelerometer values (possible unit issues)
                accel_max = [float(df_imu_new[accel_cols[i]].abs().max()) for i in range(3)]
                if any(a > 100.0 for a in accel_max):
                    print(f"{SCRIPT_LABEL}WARNING: Large accelerometer values detected (max: {accel_max}) - check for unit issues (should be m/s²)")
            
            if len(gyro_cols) >= 3:
                gyro_mean = [float(df_imu_new[gyro_cols[i]].mean()) for i in range(3)]
                gyro_std = [float(df_imu_new[gyro_cols[i]].std()) for i in range(3)]
                gyro_max = [float(df_imu_new[gyro_cols[i]].abs().max()) for i in range(3)]
                # Check for suspiciously large gyroscope values (possible unit issues - should be rad/s, not deg/s)
                if any(g > 10.0 for g in gyro_max):
                    print(f"{SCRIPT_LABEL}WARNING: Large gyroscope values detected (max: {gyro_max}) - check for unit issues (should be rad/s, not deg/s)")
        except Exception as e:
            pass  # Skip diagnostics if parsing fails
        
        # Validate timestamp format consistency between RGB and IMU
        try:
            df_rgb_check = pd.read_csv(rgb_csv)
            df_imu_check = pd.read_csv(imu_dst)
            
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
            print(f"{SCRIPT_LABEL}Could not validate timestamp consistency: {e}")

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.resolve_sequence_path(sequence_name)
        rig_yaml = self._find_rig_yaml()
        if not rig_yaml:
            # Fallback: copy existing calibration.yaml if present
            src_cal = self._find_file_recursive(['calibration.yaml'])
            if src_cal:
                dst_cal = os.path.join(sequence_path, 'calibration.yaml')
                # Check if source and destination are the same file
                import shutil
                if os.path.abspath(src_cal) != os.path.abspath(dst_cal):
                    shutil.copy(src_cal, dst_cal)
                # If they're the same, the file already exists where it should be
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
            # 1-indexed for OKVIS2 (camId=0 expects T_b_c1, camId=1 expects T_b_c2)
            imu_transforms[f"T_b_c{idx+1}"] = cam['T_b_c']

        # IMU noise - use EUROC defaults if missing (critical for OKVIS2 IMU fusion)
        # These are typical values for IMU sensors; 0.0 would cause OKVIS2 to over-trust IMU
        default_gyro_noise = 1.696800e-04  # rad/s/sqrt(Hz) - EUROC default
        default_accel_noise = 2.000000e-03  # m/s²/sqrt(Hz) - EUROC default
        default_gyro_bias = 1.939300e-05   # rad/s/sqrt(Hz) - EUROC default
        default_accel_bias = 3.000000e-03  # m/s²/sqrt(Hz) - EUROC default
        
        imu = {
            'gyro_noise': rig['imu']['noise'].get('gyroscope_noise_density', default_gyro_noise),
            'accel_noise': rig['imu']['noise'].get('accelerometer_noise_density', default_accel_noise),
            'gyro_bias': rig['imu']['noise'].get('gyroscope_random_walk', default_gyro_bias),
            'accel_bias': rig['imu']['noise'].get('accelerometer_random_walk', default_accel_bias),
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


