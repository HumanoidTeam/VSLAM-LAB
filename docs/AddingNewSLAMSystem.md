# Adding New SLAM Systems to VSLAM-LAB

This comprehensive guide covers the complete workflow for integrating new SLAM systems into VSLAM-LAB, from processing MCAP files to creating multi-camera rigs and implementing SLAM system wrappers.

## Table of Contents

1. [MCAP → Dataset → Rig → SLAM Workflow](#mcap--dataset--rig--slam-workflow)
2. [Adding a New SLAM System](#adding-a-new-slam-system)
3. [File Structure Reference](#file-structure-reference)
4. [Example Integrations](#example-integrations)
5. [Testing and Validation](#testing-and-validation)

## MCAP → Dataset → Rig → SLAM Workflow

### Overview

The VSLAM-LAB workflow processes raw sensor data through several stages:

```
MCAP File → Dataset Generation → Rig Creation → SLAM Execution
```

### Step 1: MCAP File Processing

MCAP (MessagePack) files contain raw sensor data from robotic systems. To process them:

1. **Extract sensor data** from MCAP files into organized folder structures
2. **Generate calibration files** for cameras and IMU
3. **Create synchronized data streams** (RGB images, IMU measurements)

### Step 2: Dataset Generation

VSLAM-LAB expects datasets in a specific structure:

```
VSLAM-LAB-Benchmark/
└── YOUR_DATASET/
    └── sequence_01/
        ├── rgb/                    # Image folders (or camera-specific folders)
        │   ├── img_000001.png
        │   ├── img_000002.png
        │   └── ...
        ├── calibration.yaml        # Camera and IMU calibration
        ├── rgb.csv                 # Synchronized image timestamps and paths
        ├── imu.csv                 # IMU measurements
        └── groundtruth.txt         # Ground truth trajectory (optional)
```

#### Key Files:

- **`calibration.yaml`**: Contains camera intrinsics, extrinsics, and IMU parameters
- **`rgb.csv`**: Synchronized image timestamps and file paths
- **`imu.csv`**: IMU measurements with timestamps

### Step 3: Rig Creation

For multi-camera systems, use the rig builder tool:

```bash
# Discover available cameras in a dataset
pixi run rig-discover DATASET SEQUENCE

# Create a stereo rig
pixi run rig-create DATASET SEQUENCE --name rear_stereo --cams rear_left rear_right --ref rear_left --activate

# Activate an existing rig
pixi run rig-activate DATASET SEQUENCE rear_stereo
```

The rig builder:
- Discovers camera folders and metadata
- Creates `rig.yaml` configuration files
- Generates synchronized multi-camera CSV files
- Handles camera-IMU calibration

### Step 4: SLAM Execution

Once datasets and rigs are prepared, run SLAM systems:

```bash
# Run a specific baseline on a sequence
pixi run demo BASELINE DATASET SEQUENCE

# Example: Run ORB-SLAM3-DEV on EuRoC dataset
pixi run demo orbslam3-dev euroc MH_01_easy
```

## Adding a New SLAM System

### Step 1: Create Python Baseline Class

Create a new file `Baselines/baseline_{name}.py`:

```python
import os.path
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

class YOURSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='yourslam', baseline_folder='YOUR_SLAM'):
        default_parameters = {
            'verbose': 1, 
            'mode': 'mono',  # or 'mono-vi', 'stereo-vi', 'multi-vi'
            'param1': 'value1'
        }
        
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'  # Color for console output
        self.modes = ['mono']  # Supported modes

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        # For C++ executables
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)
        # For Python scripts
        # return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    def is_installed(self): 
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed')

# Development version (optional)
class YOURSLAM_baseline_dev(YOURSLAM_baseline):
    def __init__(self):
        super().__init__(baseline_name='yourslam-dev', baseline_folder='YOUR_SLAM-DEV')

    def is_installed(self):
        # Check for specific executables
        executable = os.path.isfile(os.path.join(self.baseline_path, 'bin', 'vslamlab_yourslam'))
        return (True, 'is installed') if executable else (False, 'not installed (auto install available)')
```

### Step 2: Add Pixi Environment Configuration

Add to `pixi.toml`:

```toml
# Environment definition
yourslam-dev = {features = ["yourslam-dev", "cuda126", "py11"], no-default-feature = true}

# Feature configuration
[feature.yourslam-dev]
channels = ["https://fast.prefix.dev/conda-forge"]
platforms = ["linux-64"]

[feature.yourslam-dev.system-requirements]
cuda = "12.0"  # If CUDA is required

[feature.yourslam-dev.tasks]
git-clone = {cmd = "git clone https://github.com/user/yourslam.git Baselines/YOUR_SLAM-DEV"}
install = {cmd = "./build.sh", cwd = "Baselines/YOUR_SLAM-DEV"}
execute-mono = {cmd = "./bin/vslamlab_yourslam", cwd = "Baselines/YOUR_SLAM-DEV"}

[feature.yourslam-dev.dependencies]
# Add required dependencies
compilers = "*"
cmake = "*"
opencv = "*"
```

### Step 3: Create C++ Wrapper (for C++ SLAM systems)

Create wrapper executables that interface with VSLAM-LAB. Example: `Baselines/YOUR_SLAM-DEV/src/vslamlab_yourslam.cpp`

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char **argv) {
    // Parse VSLAM-LAB arguments
    std::string sequence_path, calibration_yaml, rgb_csv, exp_folder, exp_id, settings_yaml;
    bool verbose = true;
    
    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("sequence_path:") != std::string::npos) {
            sequence_path = arg.substr(14);
        }
        if (arg.find("calibration_yaml:") != std::string::npos) {
            calibration_yaml = arg.substr(17);
        }
        if (arg.find("rgb_csv:") != std::string::npos) {
            rgb_csv = arg.substr(8);
        }
        if (arg.find("exp_folder:") != std::string::npos) {
            exp_folder = arg.substr(11);
        }
        if (arg.find("exp_id:") != std::string::npos) {
            exp_id = arg.substr(7);
        }
        if (arg.find("settings_yaml:") != std::string::npos) {
            settings_yaml = arg.substr(14);
        }
        if (arg.find("verbose:") != std::string::npos) {
            verbose = std::stoi(arg.substr(8));
        }
    }
    
    // Load calibration
    // Load dataset
    // Run SLAM algorithm
    // Output trajectory in VSLAM-LAB format
    
    return 0;
}
```

**Key Requirements for C++ Wrappers:**
- Parse VSLAM-LAB standard arguments
- Read `calibration.yaml` for camera/IMU parameters
- Process synchronized data from `rgb.csv` and `imu.csv`
- Output trajectory as `{exp_id}_CameraTrajectory.txt` and `{exp_id}_KeyFrameTrajectory.txt`

### Step 4: Create Settings YAML

Create `Baselines/YOUR_SLAM-DEV/vslamlab_yourslam_settings.yaml`:

```yaml
%YAML:1.0

# Algorithm-specific parameters
Algorithm.param1: value1
Algorithm.param2: value2

# Camera parameters (will be overridden by calibration.yaml)
Camera.fx: 500.0
Camera.fy: 500.0
Camera.cx: 320.0
Camera.cy: 240.0

# IMU parameters (will be overridden by calibration.yaml)
IMU.NoiseGyro: 1.6968e-04
IMU.NoiseAcc: 2.0000e-3
IMU.Gravity: [0.0, 0.0, 9.81]
```

### Step 5: Register in get_baseline.py

Add to `Baselines/get_baseline.py`:

```python
from Baselines.baseline_yourslam import YOURSLAM_baseline
from Baselines.baseline_yourslam import YOURSLAM_baseline_dev

def get_baseline_switcher():
    return {
        # ... existing baselines ...
        "yourslam": lambda: YOURSLAM_baseline(),
        "yourslam-dev": lambda: YOURSLAM_baseline_dev(),
    }
```

## File Structure Reference

```
VSLAM-LAB/
├── Baselines/
│   ├── baseline_{name}.py          # Python interface
│   ├── {NAME}/                     # Conda package version
│   │   └── vslamlab_{name}_settings.yaml
│   └── {NAME}-DEV/                 # Development version
│       ├── bin/                    # Executables output
│       ├── src/                    # Source code
│       │   └── vslamlab_{name}.cpp # C++ wrapper
│       ├── CMakeLists.txt          # Build configuration
│       └── vslamlab_{name}_settings.yaml
├── pixi.toml                       # Environment configuration
└── docs/
    └── AddingNewSLAMSystem.md      # This documentation
```

## Example Integrations

### OKVIS2-DEV (Complete Multi-Camera VI-SLAM)

**Reference**: `Baselines/OKVIS2-DEV/`

- **C++ Wrapper**: `okvis_apps/src/vslamlab_okvis2_mono_vi.cpp`
- **Python Class**: `Baselines/baseline_okvis2.py`
- **Modes**: `['mono-vi', 'stereo-vi', 'multi-vi']`
- **Pixi Tasks**: `execute-mono_vi`, `execute-stereo_vi`, `execute-multi_vi`

### ORB-SLAM2-DEV (Visual SLAM)

**Reference**: `Baselines/ORB_SLAM2-DEV/`

- **C++ Wrappers**: `bin/vslamlab_orbslam2_mono`, `vslamlab_orbslam2_stereo`
- **Python Class**: `Baselines/baseline_orbslam2.py`
- **Modes**: `['mono', 'rgbd', 'stereo']`
- **Pixi Tasks**: `execute-mono`, `execute-rgbd`, `execute-stereo`

### DROID-SLAM-DEV (Python-based)

**Reference**: `Baselines/DROID-SLAM-DEV/`

- **Python Scripts**: `vslamlab_droidslam_mono.py`, `vslamlab_droidslam_stereo.py`
- **Python Class**: `Baselines/baseline_droidslam.py`
- **Modes**: `['mono', 'rgbd', 'stereo']`
- **Uses**: `build_execute_command_python()`

## Testing and Validation

### 1. Test Installation

```bash
# Check if baseline is properly registered
pixi run print-baselines

# Check installation status
pixi run baseline-info yourslam-dev
```

### 2. Test Execution

```bash
# Test with a simple sequence
pixi run demo yourslam-dev euroc MH_01_easy

# Test with multi-camera rig
pixi run demo yourslam-dev hmnd HMND
```

### 3. Validate Output

Check that the following files are generated:
- `{exp_id}_CameraTrajectory.txt` - Camera trajectory
- `{exp_id}_KeyFrameTrajectory.txt` - Keyframe trajectory
- `system_output_{exp_id}.txt` - System logs

### 4. Integration Testing

```bash
# Run full experiment
pixi run vslamlab --exp_yaml exp_yourslam.yaml

# Evaluate results
pixi run evaluate --exp_yaml exp_yourslam.yaml
```

## Common Issues and Solutions

### 1. Build Failures

- **Missing Dependencies**: Add required packages to `[feature.yourslam-dev.dependencies]`
- **CUDA Issues**: Ensure CUDA version compatibility in `system-requirements`
- **CMake Errors**: Check `CMakeLists.txt` configuration

### 2. Runtime Errors

- **File Not Found**: Verify file paths in C++ wrapper
- **Calibration Issues**: Check `calibration.yaml` format
- **Synchronization**: Ensure `rgb.csv` and `imu.csv` are properly synchronized

### 3. Performance Issues

- **Memory Usage**: Monitor with `pixi run kill-all` if needed
- **GPU Memory**: Adjust batch sizes or model parameters
- **CPU Usage**: Optimize threading in settings YAML

## Best Practices

1. **Follow Naming Conventions**: Use consistent naming for files and classes
2. **Error Handling**: Implement robust error handling in wrappers
3. **Logging**: Use VSLAM-LAB's logging system for debugging
4. **Documentation**: Document all parameters and modes
5. **Testing**: Test with multiple datasets and configurations
6. **Performance**: Optimize for both accuracy and speed

## Support and Resources

- **VSLAM-LAB Repository**: [GitHub](https://github.com/alejandrofontan/VSLAM-LAB)
- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Check existing baseline implementations for reference
- **Community**: Join discussions in the VSLAM-LAB community

---

*This documentation is maintained as part of VSLAM-LAB. For updates and corrections, please refer to the latest version in the repository.*
