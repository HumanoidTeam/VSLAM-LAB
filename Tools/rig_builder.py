import os
import sys
import argparse
import yaml
import glob
import cv2
from pathlib import Path

# Minimal printer
SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


def find_dataset_root(benchmark_root: Path, dataset_name: str) -> Path:
    return Path(benchmark_root) / dataset_name.upper()


def discover_cameras(dataset_root: Path):
    """Discover candidate camera folders and metadata under dataset_root.
    Returns a dict keyed by camera name/id with fields: data_dir, intrinsics, distortion_coeffs,
    resolution, T_b_c, model.
    """
    candidates = {}
    # Heuristics: look for common camera folders and metadata files
    for root, dirs, files in os.walk(dataset_root):
        # camera folder heuristic: contains images and optional camera.yaml
        if any(fname.lower().endswith(('.png', '.jpg', '.jpeg')) for fname in files):
            # If this folder is a generic image subfolder (e.g., 'rgb'),
            # name the camera by its parent folder, and set data_dir accordingly.
            base = os.path.basename(root)
            parent = os.path.basename(os.path.dirname(root))
            if base.lower() in ('rgb', 'images', 'img', 'left', 'right'):
                cam_name = parent
            else:
                cam_name = base
            rel_dir = os.path.relpath(root, dataset_root)
            meta_paths = [
                os.path.join(root, 'camera.yaml'),
                os.path.join(root, 'cam.yaml'),
                os.path.join(root, 'calib.yaml'),
            ]
            meta = None
            for mp in meta_paths:
                if os.path.exists(mp):
                    with open(mp, 'r') as f:
                        try:
                            meta = yaml.safe_load(f) or {}
                        except Exception:
                            meta = None
                    break
            # Also look in the parent for a calibration.yaml commonly stored per camera
            parent_calib = os.path.join(os.path.dirname(root), 'calibration.yaml')
            if os.path.exists(parent_calib):
                try:
                    with open(parent_calib, 'r') as f:
                        parent_meta = yaml.safe_load(f) or {}
                    if meta is None:
                        meta = parent_meta
                    elif isinstance(parent_meta, dict):
                        meta = {**parent_meta, **meta}
                except Exception:
                    pass
            # Expect keys; be permissive, fill later in create
            candidates[cam_name] = {
                'data_dir': rel_dir,
                'meta': meta or {},
            }
    return candidates


def find_imu(dataset_root: Path):
    """Locate an IMU CSV and optional noise/frequency metadata."""
    imu_csv = None
    noise = {}
    for root, _dirs, files in os.walk(dataset_root):
        for fname in files:
            low = fname.lower()
            if low == 'imu.csv' or low.endswith('_imu.csv'):
                imu_csv = os.path.join(root, fname)
                imu_csv = os.path.relpath(imu_csv, dataset_root)
                break
        if imu_csv:
            break
    # Optional: look for imu.yaml
    imu_yaml = None
    for root, _dirs, files in os.walk(dataset_root):
        for fname in files:
            if fname.lower() in ('imu.yaml', 'imu.yml'):
                imu_yaml = os.path.join(root, fname)
                break
        if imu_yaml:
            break
    if imu_yaml and os.path.exists(os.path.join(dataset_root, imu_yaml)):
        with open(os.path.join(dataset_root, imu_yaml), 'r') as f:
            try:
                data = yaml.safe_load(f) or {}
                noise = data.get('noise', {})
                if 'frequency' in data:
                    noise['frequency'] = data['frequency']
            except Exception:
                pass
    return imu_csv, noise


def read_yaml_file(path: str):
    try:
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and len(data) > 0:
            return data
    except Exception:
        pass
    # Fallback: parse common OpenCV-style YAML with flat keys and matrix blocks
    try:
        with open(path, 'r') as f:
            text = f.read()
        parsed = {}
        # Simple line-based extraction for Camera.fx/fy/cx/cy and Camera.w/h
        import re
        def grab_num(key):
            m = re.search(rf"^{re.escape(key)}\s*:\s*([\-0-9eE\.]+)\s*$", text, re.M)
            return float(m.group(1)) if m else None
        for k in ['Camera.fx','Camera.fy','Camera.cx','Camera.cy','Camera.w','Camera.h']:
            v = grab_num(k)
            if v is not None:
                parsed[k] = v
        # Try IMU.T_b_c0/1/2 blocks with !!opencv-matrix data: [..]
        mat_keys = re.findall(r"^IMU\.(T_b_c\d+):\s*!!opencv-matrix[\s\S]*?data:\s*\[(.*?)\]", text, re.M)
        for mk, arr in mat_keys:
            try:
                nums = [float(x.strip()) for x in arr.replace('\n', ' ').split(',') if x.strip()]
                parsed[f'IMU.{mk}'] = {'data': nums}
            except Exception:
                continue
        return parsed
    except Exception:
        return {}


def find_camera_calibration(dataset_root: Path, camera_name: str):
    # First try dataset_root/<camera_name>/calibration.yaml
    direct = os.path.join(dataset_root, camera_name, 'calibration.yaml')
    if os.path.exists(direct):
        return direct
    # Otherwise search recursively for calibration.yaml with parent == camera_name
    for root, _dirs, files in os.walk(dataset_root):
        if 'calibration.yaml' in files and os.path.basename(root) == camera_name:
            return os.path.join(root, 'calibration.yaml')
    return None


def normalize_camera_from_meta(cam_name: str, meta: dict):
    """Extract intrinsics/distortion/resolution/T_b_c/model from a generic meta dict.
    Errors are deferred; return a partial spec that the caller validates.
    """
    def get_nested(path, default=None):
        cur = meta
        for p in path.split('.'):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def get_any(key, default=None):
        # Try nested lookup (split by '.') then flat key as-is
        val = get_nested(key, None)
        if val is not None:
            return val
        if isinstance(meta, dict) and key in meta:
            return meta[key]
        return default

    # Common schemas for intrinsics
    intr = get_any('intrinsics') or get_any('camera_matrix') or get_any('K')
    if isinstance(intr, dict):
        fx = intr.get('fx'); fy = intr.get('fy'); cx = intr.get('cx'); cy = intr.get('cy')
    elif isinstance(intr, (list, tuple)) and len(intr) >= 4 and not any(isinstance(x, (list, tuple)) for x in intr):
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    elif isinstance(intr, (list, tuple)) and len(intr) == 3 and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in intr):
        fx, fy, cx, cy = intr[0][0], intr[1][1], intr[0][2], intr[1][2]
    else:
        # Try Camera.fx schema (flat keys like 'Camera.fx')
        fx = get_any('Camera.fx'); fy = get_any('Camera.fy'); cx = get_any('Camera.cx'); cy = get_any('Camera.cy')

    dist = get_any('distortion_coeffs') or get_any('distortion') or get_any('D')
    if isinstance(dist, dict):
        k1 = dist.get('k1'); k2 = dist.get('k2'); p1 = dist.get('p1'); p2 = dist.get('p2'); k3 = dist.get('k3', 0.0)
        distortion_coeffs = [k1, k2, p1, p2, k3]
    elif isinstance(dist, (list, tuple)):
        vals = list(dist) + [0.0] * (5 - len(dist))
        distortion_coeffs = vals[:5]
    else:
        # Try Camera.k* schema
        k1 = get_any('Camera.k1', 0.0); k2 = get_any('Camera.k2', 0.0); p1 = get_any('Camera.p1', 0.0); p2 = get_any('Camera.p2', 0.0)
        k3 = get_any('Camera.k3', 0.0)
        distortion_coeffs = [k1, k2, p1, p2, k3]

    res = get_any('resolution')
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        w, h = int(res[0]), int(res[1])
    else:
        # Try generic width/height, else Camera.w/h (flat keys)
        w = get_any('width') or get_any('Camera.w'); h = get_any('height') or get_any('Camera.h')
        w = int(w) if w else None
        h = int(h) if h else None

    # Extrinsics: try direct, then IMU.T_b_c*
    T = get_any('T_b_c') or get_any('T_BC') or get_any('T_BS') or get_any('T')
    if isinstance(T, (list, tuple)) and len(T) == 16:
        T_b_c = [float(x) for x in T]
    elif isinstance(T, (list, tuple)) and len(T) == 4 and all(isinstance(r, (list, tuple)) and len(r) == 4 for r in T):
        T_b_c = [float(x) for row in T for x in row]
    else:
        # Check OpenCV-matrix style under IMU.T_b_c* as flat keys
        T_b_c = None
        if isinstance(meta, dict):
            # First try preferred keys
            for k in ('IMU.T_b_c0', 'IMU.T_b_c1'):
                if k in meta and isinstance(meta[k], dict):
                    data = meta[k].get('data', [])
                    if isinstance(data, list) and len(data) >= 16:
                        T_b_c = [float(x) for x in data[:16]]
                        break
            # Else scan all IMU.T_b_c*
            if T_b_c is None:
                for k, v in meta.items():
                    if isinstance(k, str) and k.startswith('IMU.T_b_c'):
                        if isinstance(v, dict) and 'data' in v:
                            data = v.get('data', [])
                            if isinstance(data, list) and len(data) >= 16:
                                T_b_c = [float(x) for x in data[:16]]
                                break
                        elif isinstance(v, (list, tuple)):
                            flat = [item for row in v for item in (row if isinstance(row, (list, tuple)) else [row])]
                            if len(flat) >= 16:
                                T_b_c = [float(x) for x in flat[:16]]
                                break

    model = get_any('model') or get_any('camera_model') or get_any('Camera.model') or 'pinhole'
    if isinstance(model, str):
        model = model.capitalize()

    return {
        'name': cam_name,
        'intrinsics': [fx, fy, cx, cy] if None not in (fx, fy, cx, cy) else None,
        'distortion_coeffs': distortion_coeffs,
        'resolution': [w, h] if None not in (w, h) else None,
        'T_b_c': T_b_c,
        'model': model,
    }


def cmd_discover(args):
    dataset_root = find_dataset_root(args.benchmark, args.dataset)
    cams = discover_cameras(dataset_root)
    imu_csv, noise = find_imu(dataset_root)
    print(f"{SCRIPT_LABEL}Dataset: {dataset_root}")
    print(f"{SCRIPT_LABEL}Found cameras: {sorted(cams.keys())}")
    if imu_csv:
        print(f"{SCRIPT_LABEL}Found IMU csv: {imu_csv}")
    else:
        print(f"{SCRIPT_LABEL}IMU csv not found")


def cmd_create(args):
    dataset_root = find_dataset_root(args.benchmark, args.dataset)
    cams = discover_cameras(dataset_root)
    missing = [c for c in args.cams if c not in cams]
    if missing:
        print(f"{SCRIPT_LABEL}Unknown cameras: {missing}")
        sys.exit(1)

    imu_csv, noise = find_imu(dataset_root)
    if not imu_csv:
        print(f"{SCRIPT_LABEL}IMU csv not found under {dataset_root}")
        sys.exit(1)

    # Normalize camera metadata
    selected = []
    for name in args.cams:
        base_meta = cams[name]['meta']
        meta = normalize_camera_from_meta(name, base_meta)
        # Attach data_dir
        meta['data_dir'] = cams[name]['data_dir']
        # If critical fields missing, try loading parent calibration.yaml explicitly
        if (not meta['intrinsics'] or not meta['T_b_c']):
            parent_dir = os.path.dirname(os.path.join(dataset_root, meta['data_dir']))
            calib_path = os.path.join(parent_dir, 'calibration.yaml')
            if os.path.exists(calib_path):
                try:
                    with open(calib_path, 'r') as f:
                        calib_meta = yaml.safe_load(f) or {}
                    meta = normalize_camera_from_meta(name, {**calib_meta, **base_meta})
                    meta['data_dir'] = cams[name]['data_dir']
                except Exception:
                    pass
        # If still missing, search for a calib file elsewhere under dataset
        if (not meta['intrinsics'] or not meta['T_b_c']):
            alt_calib = find_camera_calibration(dataset_root, name)
            if alt_calib:
                alt_meta = read_yaml_file(alt_calib)
                meta2 = normalize_camera_from_meta(name, {**alt_meta, **base_meta})
                # Keep any fields we already have
                meta['intrinsics'] = meta['intrinsics'] or meta2['intrinsics']
                meta['T_b_c'] = meta['T_b_c'] or meta2['T_b_c']
                # Resolution fallback below
        # If resolution missing, try to infer from an image
        if not meta['resolution']:
            img_glob = os.path.join(dataset_root, meta['data_dir'], '*.png')
            imgs = glob.glob(img_glob) or glob.glob(os.path.join(dataset_root, meta['data_dir'], '*.jpg'))
            if imgs:
                img = cv2.imread(imgs[0])
                if img is not None:
                    h, w = img.shape[:2]
                    meta['resolution'] = [w, h]
        if not meta['intrinsics'] or not meta['resolution'] or not meta['T_b_c']:
            if not args.fallback_identity:
                print(f"{SCRIPT_LABEL}Camera '{name}' missing intrinsics/resolution/T_b_c. Provide metadata or use --fallback-identity")
                sys.exit(1)
            meta['T_b_c'] = meta['T_b_c'] or [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
            meta['intrinsics'] = meta['intrinsics'] or [500.0, 500.0, 320.0, 240.0]
            meta['resolution'] = meta['resolution'] or [640, 480]
        selected.append(meta)

    # Assign deterministic IDs by order
    id_map = {name: idx for idx, name in enumerate(args.cams)}
    cameras_yaml = []
    for meta in selected:
        cameras_yaml.append({
            'id': id_map[meta['name']],
            'name': meta['name'],
            'model': meta['model'],
            'intrinsics': meta['intrinsics'],
            'distortion_coeffs': meta['distortion_coeffs'],
            'resolution': meta['resolution'],
            'T_b_c': meta['T_b_c'],
            'data_dir': meta['data_dir'],
        })

    rig = {
        'imu': {
            'csv': imu_csv,
            'noise': noise,
        },
        'cameras': cameras_yaml,
        'defaults': {
            'run_cameras': [id_map[n] for n in args.cams],
            'reference_camera': id_map[args.ref] if args.ref else [id_map[n] for n in args.cams][0],
        },
        'pairing': {
            'reference_camera': id_map[args.ref] if args.ref else [id_map[n] for n in args.cams][0],
            'max_time_offset': float(args.max_dt),
            'require_all': True,
        }
    }

    rigs_dir = dataset_root / 'rigs'
    os.makedirs(rigs_dir, exist_ok=True)
    rig_path = rigs_dir / f"{args.name}.yaml"
    with open(rig_path, 'w') as f:
        yaml.safe_dump(rig, f, sort_keys=False)
    print(f"{SCRIPT_LABEL}Wrote rig: {rig_path}")

    if args.activate:
        activate_rig(dataset_root, args.name)


def activate_rig(dataset_root: Path, name: str):
    rigs_dir = dataset_root / 'rigs'
    src = rigs_dir / f"{name}.yaml"
    if not src.exists():
        print(f"{SCRIPT_LABEL}Rig not found: {src}")
        sys.exit(1)
    dst = dataset_root / 'rig.yaml'
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
    except FileNotFoundError:
        pass
    os.symlink(os.path.relpath(src, dataset_root), dst)
    print(f"{SCRIPT_LABEL}Activated rig: {dst} -> {src}")


def cmd_activate(args):
    dataset_root = find_dataset_root(args.benchmark, args.dataset)
    activate_rig(dataset_root, args.name)


def build_parser():
    p = argparse.ArgumentParser(description="VSLAM-LAB Rig Builder")
    sub = p.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('dataset', type=str, help='dataset key (e.g., hmnd)')
    common.add_argument('sequence', type=str, help='sequence name (unused, for symmetry)')
    common.add_argument('--benchmark', type=Path, default=Path(os.path.expandvars('/home/vmehta/datasets/vislam-lab/VSLAM-LAB-Benchmark')), help='Benchmark root')

    pd = sub.add_parser('discover', parents=[common], help='Discover cameras and IMU in dataset root')
    pd.set_defaults(func=cmd_discover)

    pc = sub.add_parser('create', parents=[common], help='Create a named rig from camera subset')
    pc.add_argument('--name', required=True, help='rig name (e.g., rear_stereo)')
    pc.add_argument('--cams', nargs='+', required=True, help='camera folder names to include, in order')
    pc.add_argument('--ref', default=None, help='reference camera name for pairing')
    pc.add_argument('--max-dt', default=0.02, help='max timestamp offset (s)')
    pc.add_argument('--activate', action='store_true', help='activate newly created rig')
    pc.add_argument('--fallback-identity', action='store_true', help='allow identity T_b_c and default intrinsics if missing')
    pc.set_defaults(func=cmd_create)

    pa = sub.add_parser('activate', parents=[common], help='Activate an existing rig by name')
    pa.add_argument('name', help='rig name to activate')
    pa.set_defaults(func=cmd_activate)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()


