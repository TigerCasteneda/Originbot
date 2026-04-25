import argparse
import sys
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np

h1 = 70
h2 = 518


def get_available_cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def validate_numpy_compatibility():
    version_text = str(np.__version__)
    major_text = version_text.split(".")[0]
    try:
        major = int(major_text)
    except ValueError:
        major = 1
    if major >= 2:
        raise RuntimeError(
            f"Incompatible numpy version detected: {version_text}. "
            "Torch 1.12.1 / torchvision 0.13.1 require numpy<2. "
            "Please run: pip install --force-reinstall 'numpy==1.26.4' "
            "and restart Python/Jupyter kernel."
        )


def get_x(path):
    """Gets the x value from the image filename"""
    return (float(int(path.split("_")[1])) / 960.0 - 1.0)


def get_y(path):
    """Gets the y value from the image filename"""
    return (float(int(path.split("_")[2])) / (0.5 * (h2 - h1)) - 1.0)


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directories, random_hflips=False):
        if isinstance(directories, (str, os.PathLike)):
            directories = [directories]
        self.directories = [str(Path(directory)) for directory in directories]
        self.random_hflips = random_hflips
        self.image_paths = []
        self.files_found_per_dir = {}
        self.files_used_per_dir = {}
        self.invalid_filenames = []

        for directory in self.directories:
            image_paths = sorted(glob.glob(os.path.join(directory, "*.jpg")))
            self.files_found_per_dir[directory] = len(image_paths)
            used_count = 0
            for image_path in image_paths:
                filename = os.path.basename(image_path)
                try:
                    get_x(filename)
                    get_y(filename)
                except (ValueError, IndexError):
                    self.invalid_filenames.append(image_path)
                    continue
                self.image_paths.append(image_path)
                used_count += 1
            self.files_used_per_dir[directory] = used_count

        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
        self.color_jitter_no_hue = transforms.ColorJitter(0.3, 0.3, 0.3, 0.0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path)))
        y = float(get_y(os.path.basename(image_path)))

        if self.random_hflips:
            if float(np.random.rand(1)) > 0.5:
                image = transforms.functional.hflip(image)
                x = -x

        try:
            image = self.color_jitter(image)
        except OverflowError:
            image = self.color_jitter_no_hue(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(
            image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        return image, torch.tensor([x, y], dtype=torch.float32)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train ResNet18 for XY coordinate regression."
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=None,
        help="Legacy single dataset directory path",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        dest="dataset_dirs",
        default=[],
        help="Dataset directory path. Repeat this option to use multiple directories.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and validation (default: 128)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, get_available_cpu_count()),
        help="DataLoader worker processes (default: min(8, CPU quota))",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when num_workers > 0 (default: 2)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=3,
        help="Run validation every N epochs, plus final epoch (default: 3)",
    )
    parser.add_argument(
        "--disable-channels-last",
        action="store_true",
        help="Disable channels-last memory format optimization on GPU",
    )
    parser.add_argument(
        "--expected-gpu-name",
        type=str,
        default="",
        help="Optional GPU name keyword check, e.g. 'NVIDIA A16'",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training for better GPU throughput",
    )
    opts = parser.parse_args(args or sys.argv[1:])
    validate_numpy_compatibility()

    max_workers = get_available_cpu_count()
    if opts.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if opts.num_workers > max_workers:
        print(
            f"[WARN] Reducing num_workers from {opts.num_workers} to {max_workers} "
            f"based on current CPU quota."
        )
        opts.num_workers = max_workers
    if opts.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if opts.val_every < 1:
        raise ValueError("--val-every must be >= 1")

    if opts.dataset_dirs:
        dataset_dirs = [str(Path(directory)) for directory in opts.dataset_dirs]
    elif opts.dataset_dir:
        dataset_dirs = [str(Path(opts.dataset_dir))]
    else:
        dataset_dirs = [str(Path.cwd() / "image_dataset")]

    for directory in dataset_dirs:
        if not Path(directory).is_dir():
            raise ValueError(f"Dataset directory does not exist: {directory}")

    dataset = XYDataset(dataset_dirs, random_hflips=False)

    print("Using datasets:")
    for directory in dataset_dirs:
        print(f" - {directory}")
    for directory in dataset_dirs:
        found = dataset.files_found_per_dir[directory]
        used = dataset.files_used_per_dir[directory]
        print(f"   {directory}: found={found}, used={used}")
    print(f"Merged usable samples: {len(dataset)}")
    print(f"Skipped invalid filenames: {len(dataset.invalid_filenames)}")
    if dataset.invalid_filenames:
        print(f"First invalid filename: {dataset.invalid_filenames[0]}")

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 images for train/test split.")

    test_percent = 0.1
    num_test = max(1, int(test_percent * len(dataset)))
    num_test = min(num_test, len(dataset) - 1)
    num_train = len(dataset) - num_test
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_test]
    )

    train_loader_kwargs = dict(
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader_kwargs = dict(
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if opts.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        test_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = opts.prefetch_factor
        test_loader_kwargs["prefetch_factor"] = opts.prefetch_factor

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_loader_kwargs)
    if len(train_loader) == 0:
        raise ValueError(
            "Train loader has zero batches. Reduce --batch-size or collect more data."
        )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, 2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU training.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    if opts.expected_gpu_name and opts.expected_gpu_name not in gpu_name:
        raise RuntimeError(
            f"GPU mismatch: expected keyword '{opts.expected_gpu_name}', got '{gpu_name}'"
        )
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    use_channels_last = bool(torch.cuda.is_available() and not opts.disable_channels_last)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    print(f"device={device}, cuda={torch.cuda.is_available()}, gpu={gpu_name}")
    print(
        f"batch_size={opts.batch_size}, num_workers={opts.num_workers}, "
        f"prefetch_factor={opts.prefetch_factor if opts.num_workers > 0 else 'n/a'}, "
        f"amp={opts.amp}, val_every={opts.val_every}, channels_last={use_channels_last}"
    )

    num_epochs = opts.epochs
    if num_epochs < 1:
        raise ValueError("--epochs must be >= 1")
    best_model_path = str(Path.cwd() / "best_line_follower_model_xy.pth")
    best_loss = 1e9

    optimizer = optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=opts.amp):
                outputs = model(images)
                loss = F.mse_loss(outputs, labels)
            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        train_loss /= len(train_loader)

        should_validate = ((epoch + 1) % opts.val_every == 0) or (epoch == num_epochs - 1)
        if should_validate:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device, non_blocking=True)
                    if use_channels_last:
                        images = images.contiguous(memory_format=torch.channels_last)
                    labels = labels.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=opts.amp):
                        outputs = model(images)
                        loss = F.mse_loss(outputs, labels)
                    test_loss += loss.item()
            test_loss /= len(test_loader)
            print("Epoch:%d,Train loss:%f,Val loss:%f" % (epoch, train_loss, test_loss))
            if test_loss < best_loss:
                print("save")
                torch.save(model.state_dict(), best_model_path)
                best_loss = test_loss
        else:
            print(
                "Epoch:%d,Train loss:%f,Val skipped (every %d epochs)"
                % (epoch, train_loss, opts.val_every)
            )


if __name__ == "__main__":
    main()
