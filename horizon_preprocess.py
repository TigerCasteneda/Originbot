import argparse
from pathlib import Path

import cv2
import numpy as np


def collect_images(src_dir: Path):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for pattern in patterns:
        paths.extend(src_dir.glob(pattern))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess image dataset into BGR float32 raw files for Horizon calibration."
    )
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--dst_dir", required=True)
    parser.add_argument("--pic_ext", default=".rgb")
    parser.add_argument("--read_mode", default="opencv")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    image_paths = collect_images(src_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {src_dir}")

    written = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(
            image, (args.width, args.height), interpolation=cv2.INTER_LINEAR
        )
        image = image.astype(np.float32)
        output_path = dst_dir / f"{image_path.stem}{args.pic_ext}"
        image.tofile(str(output_path))
        written += 1

    if written == 0:
        raise RuntimeError("No valid images were written during preprocessing.")

    print(f"[DONE] wrote {written} files to {dst_dir}")


if __name__ == "__main__":
    main()
