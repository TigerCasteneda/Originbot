#!/usr/bin/env python3
import argparse
import copy
import glob
import os
import random
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

models = torchvision.models

h1 = 70
h2 = 518
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
FLOAT32_BYTES = 4

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError, ValueError):
    pass


def get_available_cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def get_shm_stats():
    shm_path = "/dev/shm"
    if not hasattr(os, "statvfs"):
        return None
    try:
        stats = os.statvfs(shm_path)
        total = stats.f_frsize * stats.f_blocks
        available = stats.f_frsize * stats.f_bavail
        return {
            "path": shm_path,
            "total_bytes": int(total),
            "available_bytes": int(available),
        }
    except OSError:
        return None


def format_bytes(num_bytes):
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def estimate_batch_tensor_bytes(batch_size):
    image_bytes = batch_size * IMAGE_CHANNELS * IMAGE_WIDTH * IMAGE_HEIGHT * FLOAT32_BYTES
    label_bytes = batch_size * 2 * FLOAT32_BYTES
    return image_bytes + label_bytes


def choose_safe_num_workers(requested_workers, batch_size, prefetch_factor):
    shm_stats = get_shm_stats()
    batch_tensor_bytes = estimate_batch_tensor_bytes(batch_size)
    if shm_stats is None:
        return requested_workers, prefetch_factor, None

    available = shm_stats["available_bytes"]
    safe_workers = requested_workers
    safe_prefetch = prefetch_factor

    # Container /dev/shm is often small. Reduce worker pressure before the
    # DataLoader starts forking or spawning subprocesses.
    if available < 512 * 1024 * 1024 and safe_workers > 0:
        safe_workers = 0
    elif available < 1024 * 1024 * 1024 and safe_workers > 1:
        safe_workers = 1
    elif available < 1536 * 1024 * 1024 and safe_workers > 2:
        safe_workers = 2

    if safe_workers > 0:
        estimated_prefetch_bytes = batch_tensor_bytes * safe_workers * max(1, safe_prefetch)
        if estimated_prefetch_bytes > available * 0.6:
            safe_prefetch = 1
        if batch_tensor_bytes * safe_workers > available * 0.8:
            safe_workers = max(0, min(1, safe_workers))
            safe_prefetch = 1

    info = {
        "shm_path": shm_stats["path"],
        "shm_total_bytes": shm_stats["total_bytes"],
        "shm_available_bytes": shm_stats["available_bytes"],
        "estimated_batch_bytes": batch_tensor_bytes,
        "requested_workers": requested_workers,
        "safe_workers": safe_workers,
        "requested_prefetch_factor": prefetch_factor,
        "safe_prefetch_factor": safe_prefetch,
    }
    return safe_workers, safe_prefetch, info


def build_dataloaders(train_dataset, test_dataset, opts, device):
    pin_memory = device.type == "cuda"
    train_loader_kwargs = dict(
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader_kwargs = dict(
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if opts.num_workers > 0:
        persistent_workers = True if opts.persistent_workers is None else bool(opts.persistent_workers)
        train_loader_kwargs["persistent_workers"] = persistent_workers
        test_loader_kwargs["persistent_workers"] = persistent_workers
        train_loader_kwargs["prefetch_factor"] = opts.prefetch_factor
        test_loader_kwargs["prefetch_factor"] = opts.prefetch_factor
        try:
            mp_context = torch.multiprocessing.get_context("spawn")
            train_loader_kwargs["multiprocessing_context"] = mp_context
            test_loader_kwargs["multiprocessing_context"] = mp_context
        except Exception:
            pass
    else:
        train_loader_kwargs["pin_memory"] = False
        test_loader_kwargs["pin_memory"] = False

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_loader_kwargs)
    return train_loader, test_loader


def is_shared_memory_error(exc):
    text = str(exc).lower()
    return (
        "bus error" in text
        or "shared memory" in text
        or "no space left on device" in text
        or "unexpected bus error" in text
    )


def validate_numpy_compatibility():
    version_text = str(np.__version__)
    major_text = version_text.split(".")[0]
    try:
        major = int(major_text)
    except ValueError:
        major = 1

    torch_major_text = str(torch.__version__).split(".", 1)[0]
    try:
        torch_major = int(torch_major_text)
    except ValueError:
        torch_major = 1

    if major >= 2 and torch_major < 2:
        raise RuntimeError(
            f"Incompatible numpy version detected: {version_text}. "
            f"Current torch/torchvision: {torch.__version__} / {torchvision.__version__}. "
            "Older torch builds require numpy<2. "
            "Please run: pip install --force-reinstall 'numpy==1.26.4' "
            "and restart Python/Jupyter kernel."
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_grad_scaler(enabled):
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        try:
            return amp_module.GradScaler("cuda", enabled=enabled)
        except TypeError:
            try:
                return amp_module.GradScaler(device="cuda", enabled=enabled)
            except TypeError:
                return amp_module.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled):
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        try:
            return amp_module.autocast("cuda", enabled=enabled)
        except TypeError:
            try:
                return amp_module.autocast(device_type="cuda", enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.autocast(enabled=enabled)


def is_probably_nvidia_gpu(gpu_name):
    name = str(gpu_name).strip().lower()
    return "nvidia" in name or "geforce" in name or "rtx" in name or "gtx" in name


def resolve_device(requested_device):
    requested = (requested_device or "cuda").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0"), None
        return torch.device("cpu"), "[WARN] CUDA is not available, falling back to CPU."

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested), None
        return torch.device("cpu"), "[WARN] CUDA is not available, falling back to CPU."

    return torch.device("cpu" if requested == "cpu" else requested), None


def initialize_cuda_context(device):
    # Force lazy CUDA context creation before the first backward pass so
    # runtimes with CUDA-compat layers do not emit late initialization warnings.
    try:
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_device(device_index)
    except Exception:
        pass
    try:
        _ = torch.empty(1, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        pass


def warm_up_cuda_libraries(device):
    # Some CUDA-compat runtimes delay cuBLAS handle creation until the first
    # matrix multiply used by backward(). Run a tiny GEMM upfront.
    try:
        with torch.no_grad():
            a = torch.randn((8, 8), device=device)
            b = torch.randn((8, 8), device=device)
            _ = torch.mm(a, b)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        pass


def get_x(path):
    """Gets the x value from the image filename."""
    return float(int(path.split("_")[1])) / 960.0 - 1.0


def get_y(path):
    """Gets the y value from the image filename."""
    return float(int(path.split("_")[2])) / (0.5 * (h2 - h1)) - 1.0


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

        image = PIL.Image.open(image_path).convert("RGB")
        x = float(get_x(os.path.basename(image_path)))
        y = float(get_y(os.path.basename(image_path)))

        if self.random_hflips and float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x

        try:
            image = self.color_jitter(image)
        except OverflowError:
            image = self.color_jitter_no_hue(image)
        image = transforms.functional.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(
            image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        return image, torch.tensor([x, y], dtype=torch.float32)


def make_model(use_pretrained):
    if hasattr(models, "ResNet18_Weights"):
        weights = None
        if use_pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            model = models.resnet18(weights=weights)
        except TypeError:
            model = models.resnet18(pretrained=use_pretrained)
        except Exception as exc:
            if use_pretrained:
                print(
                    f"[WARN] Falling back to random init because pretrained weights "
                    f"are unavailable: {exc}"
                )
            model = models.resnet18(weights=None)
    else:
        try:
            model = models.resnet18(pretrained=use_pretrained)
        except Exception as exc:
            if use_pretrained:
                print(
                    f"[WARN] Falling back to random init because pretrained weights "
                    f"are unavailable: {exc}"
                )
            model = models.resnet18(pretrained=False)

    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def parse_args(args=None):
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
        "--output-pth",
        default=None,
        help="Path to save the best checkpoint (.pth). Default: ./best_line_follower_model_xy.pth",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and training initialization (default: 42)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam (default: 0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device: cuda, cuda:0, cpu, or auto (default: cuda)",
    )
    parser.add_argument(
        "--disable-channels-last",
        action="store_true",
        help="Disable channels-last memory format optimization on GPU",
    )
    parser.add_argument(
        "--force-channels-last",
        action="store_true",
        help="Force channels-last even on non-NVIDIA CUDA-compatible runtimes",
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
    parser.add_argument(
        "--force-amp",
        action="store_true",
        help="Force AMP even on non-NVIDIA CUDA-compatible runtimes",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="Use torchvision pretrained weights when available (default: on)",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable pretrained weights",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers",
    )
    parser.set_defaults(persistent_workers=None)
    parser.add_argument(
        "--no-shm-guard",
        action="store_true",
        help="Do not auto-reduce DataLoader workers based on /dev/shm size",
    )
    return parser.parse_args(args)


def main(args=None):
    opts = parse_args(args)
    validate_numpy_compatibility()
    set_seed(opts.seed)

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
    if opts.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if opts.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

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

    device, device_warn = resolve_device(opts.device)
    if device_warn:
        print(device_warn)

    if device.type == "cuda":
        device_index = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(device_index)
    else:
        gpu_name = "cpu"

    if opts.expected_gpu_name and opts.expected_gpu_name not in gpu_name:
        raise RuntimeError(
            f"GPU mismatch: expected keyword '{opts.expected_gpu_name}', got '{gpu_name}'"
        )

    is_nvidia_gpu = device.type == "cuda" and is_probably_nvidia_gpu(gpu_name)
    if opts.amp and not (is_nvidia_gpu or opts.force_amp):
        print(
            f"[WARN] Disabling AMP on non-NVIDIA or CPU runtime: {gpu_name}. "
            "This path is often unstable on compatibility-layer CUDA stacks."
        )
        opts.amp = False
    elif opts.force_amp and device.type == "cuda" and not is_nvidia_gpu:
        print(
            f"[WARN] Forcing AMP on non-NVIDIA CUDA-compatible runtime: {gpu_name}."
        )

    use_channels_last = bool(
        device.type == "cuda"
        and not opts.disable_channels_last
        and (is_nvidia_gpu or opts.force_channels_last)
    )
    if device.type == "cuda" and not is_nvidia_gpu and not opts.disable_channels_last and not opts.force_channels_last:
        print(
            f"[WARN] Disabling channels_last on non-NVIDIA GPU runtime: {gpu_name}. "
            "Use --disable-channels-last explicitly if you want this behavior documented."
        )
    elif opts.force_channels_last and device.type == "cuda" and not is_nvidia_gpu:
        print(
            f"[WARN] Forcing channels_last on non-NVIDIA CUDA-compatible runtime: {gpu_name}."
        )

    if not opts.no_shm_guard:
        safe_workers, safe_prefetch_factor, shm_info = choose_safe_num_workers(
            opts.num_workers, opts.batch_size, opts.prefetch_factor
        )
        if shm_info is not None:
            print(
                "[INFO] shared_memory:"
                f" path={shm_info['shm_path']},"
                f" total={format_bytes(shm_info['shm_total_bytes'])},"
                f" available={format_bytes(shm_info['shm_available_bytes'])},"
                f" estimated_batch={format_bytes(shm_info['estimated_batch_bytes'])}"
            )
        if safe_workers != opts.num_workers:
            print(
                f"[WARN] Reducing num_workers from {opts.num_workers} to {safe_workers} "
                "because current shared memory is too small for multi-process DataLoader."
            )
            opts.num_workers = safe_workers
        if safe_prefetch_factor != opts.prefetch_factor:
            print(
                f"[WARN] Reducing prefetch_factor from {opts.prefetch_factor} to "
                f"{safe_prefetch_factor} based on shared memory limits."
            )
            opts.prefetch_factor = safe_prefetch_factor
    else:
        print("[INFO] /dev/shm guard disabled; using requested DataLoader workers unchanged.")

    test_percent = 0.1
    num_test = max(1, int(test_percent * len(dataset)))
    num_test = min(num_test, len(dataset) - 1)
    num_train = len(dataset) - num_test
    split_generator = torch.Generator().manual_seed(opts.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_test], generator=split_generator
    )
    train_loader, test_loader = build_dataloaders(train_dataset, test_dataset, opts, device)
    if len(train_loader) == 0:
        raise ValueError(
            "Train loader has zero batches. Reduce --batch-size or collect more data."
        )

    model = make_model(opts.pretrained)
    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device.type == "cuda":
        initialize_cuda_context(device)
        warm_up_cuda_libraries(device)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    print(f"torch={torch.__version__}, torchvision={torchvision.__version__}")
    print(f"device={device}, cuda={torch.cuda.is_available()}, gpu={gpu_name}")
    print(
        f"batch_size={opts.batch_size}, num_workers={opts.num_workers}, "
        f"prefetch_factor={opts.prefetch_factor if opts.num_workers > 0 else 'n/a'}, "
        f"amp={opts.amp}, val_every={opts.val_every}, channels_last={use_channels_last}"
    )

    best_model_path = str(Path(opts.output_pth).resolve()) if opts.output_pth else str(
        Path.cwd() / "best_line_follower_model_xy.pth"
    )
    best_loss = float("inf")

    optimizer = optim.Adam(
        model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay
    )
    scaler = create_grad_scaler(enabled=opts.amp)

    for epoch in range(opts.epochs):
        while True:
            epoch_model_state = copy.deepcopy(model.state_dict())
            epoch_optimizer_state = copy.deepcopy(optimizer.state_dict())
            epoch_scaler_state = copy.deepcopy(scaler.state_dict())
            try:
                model.train()
                train_loss = 0.0
                train_samples = 0
                for images, labels in train_loader:
                    images = images.to(device, non_blocking=True)
                    if use_channels_last:
                        images = images.contiguous(memory_format=torch.channels_last)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(enabled=opts.amp):
                        outputs = model(images)
                        loss = F.mse_loss(outputs, labels)
                    batch_size = images.size(0)
                    train_loss += loss.item() * batch_size
                    train_samples += batch_size
                    if opts.amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                train_loss /= max(1, train_samples)

                should_validate = ((epoch + 1) % opts.val_every == 0) or (epoch == opts.epochs - 1)
                if should_validate:
                    model.eval()
                    test_loss = 0.0
                    test_samples = 0
                    with torch.no_grad():
                        for images, labels in test_loader:
                            images = images.to(device, non_blocking=True)
                            if use_channels_last:
                                images = images.contiguous(memory_format=torch.channels_last)
                            labels = labels.to(device, non_blocking=True)
                            with autocast_context(enabled=opts.amp):
                                outputs = model(images)
                                loss = F.mse_loss(outputs, labels)
                            batch_size = images.size(0)
                            test_loss += loss.item() * batch_size
                            test_samples += batch_size
                    test_loss /= max(1, test_samples)
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
                break
            except RuntimeError as exc:
                if not is_shared_memory_error(exc):
                    raise
                if opts.num_workers == 0:
                    raise
                print(
                    "[WARN] DataLoader hit a shared-memory failure. "
                    "Falling back to num_workers=0 and retrying this epoch."
                )
                model.load_state_dict(epoch_model_state)
                optimizer.load_state_dict(epoch_optimizer_state)
                scaler.load_state_dict(epoch_scaler_state)
                opts.num_workers = 0
                opts.prefetch_factor = 1
                opts.persistent_workers = False
                train_loader, test_loader = build_dataloaders(train_dataset, test_dataset, opts, device)


if __name__ == "__main__":
    main()
