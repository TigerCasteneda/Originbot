import argparse
import re
import shutil
from pathlib import Path


def replace_yaml_key(content: str, key: str, value: str):
    pattern = rf"(^\s*{re.escape(key)}\s*:\s*).*$"
    if re.search(pattern, content, flags=re.MULTILINE):
        updated = re.sub(pattern, rf"\1{value}", content, flags=re.MULTILINE)
        return updated, True
    return content, False


def patch_yaml_file(yaml_path: Path, onnx_path: Path, calibration_dir: Path):
    original = yaml_path.read_text(encoding="utf-8", errors="ignore")
    updated = original
    changed = False

    replacements = {
        "onnx_model": str(onnx_path),
        "cal_data_dir": str(calibration_dir),
        "input_type_rt": "nv12",
        "input_type_train": "rgb",
        "input_shape": "1x3x224x224",
        "mean_value": "123.675,116.28,103.53",
        "scale_value": "0.0171248,0.017507,0.0174292",
    }

    for key, value in replacements.items():
        updated, key_changed = replace_yaml_key(updated, key, value)
        changed = changed or key_changed

    if changed and updated != original:
        backup_path = yaml_path.with_suffix(yaml_path.suffix + ".bak")
        backup_path.write_text(original, encoding="utf-8")
        yaml_path.write_text(updated, encoding="utf-8")
    return changed


def find_target_yaml_files(mapper_dir: Path):
    yaml_files = sorted(mapper_dir.glob("*.yaml")) + sorted(mapper_dir.glob("*.yml"))
    target_yaml_files = []
    for yaml_file in yaml_files:
        text = yaml_file.read_text(encoding="utf-8", errors="ignore")
        if "onnx_model" in text or "cal_data_dir" in text:
            target_yaml_files.append(yaml_file)
    return target_yaml_files


def is_same_path(path_a: Path, path_b: Path):
    return path_a.resolve() == path_b.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Horizon OpenExplorer mapper folder for OriginBot model conversion."
    )
    parser.add_argument(
        "--mapper-dir",
        required=True,
        help="Path to .../10_model_convert/mapper directory",
    )
    parser.add_argument(
        "--onnx",
        default="./best_line_follower_model_xy.onnx",
        help="Path to source ONNX model",
    )
    parser.add_argument(
        "--dataset-dir",
        default="./image_dataset",
        help="Path to dataset directory",
    )
    args = parser.parse_args()

    mapper_dir = Path(args.mapper_dir).resolve()
    onnx_path = Path(args.onnx).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    if not mapper_dir.is_dir():
        raise FileNotFoundError(f"Mapper directory not found: {mapper_dir}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    mapper_onnx_path = (mapper_dir / "best_line_follower_model_xy.onnx").resolve()
    if is_same_path(onnx_path, mapper_onnx_path):
        print(f"[SKIP] ONNX already in mapper: {mapper_onnx_path}")
    else:
        shutil.copy2(onnx_path, mapper_onnx_path)
        print(f"[OK] Copied ONNX -> {mapper_onnx_path}")

    mapper_dataset_path = (mapper_dir / "image_dataset").resolve()
    if is_same_path(dataset_dir, mapper_dataset_path):
        print(f"[SKIP] Dataset already in mapper: {mapper_dataset_path}")
    else:
        if mapper_dataset_path.exists():
            shutil.rmtree(mapper_dataset_path)
        shutil.copytree(dataset_dir, mapper_dataset_path)
        print(f"[OK] Copied dataset -> {mapper_dataset_path}")

    calibration_dir = mapper_dir / "calibration_data_bgr_f32"
    calibration_dir.mkdir(exist_ok=True)

    yaml_files = find_target_yaml_files(mapper_dir)
    if not yaml_files:
        print("[WARN] No yaml/yml with onnx_model or cal_data_dir found in mapper.")
        print("[INFO] Continue with sample scripts if your yaml is configured elsewhere.")
        return

    patched_count = 0
    for yaml_file in yaml_files:
        changed = patch_yaml_file(yaml_file, mapper_onnx_path, calibration_dir)
        if changed:
            patched_count += 1
            print(f"[OK] Patched yaml -> {yaml_file}")
        else:
            print(f"[SKIP] No matching keys updated -> {yaml_file}")

    print(f"[DONE] YAML patched: {patched_count}/{len(yaml_files)}")


if __name__ == "__main__":
    main()
