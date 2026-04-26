#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

PATTERN = re.compile(r'^xy_(\d+)_(\d+)_')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class XYImageDataset(Dataset):
    def __init__(self, samples, transform, label_mean, label_std):
        self.samples = samples
        self.transform = transform
        self.label_mean = torch.tensor(label_mean, dtype=torch.float32)
        self.label_std = torch.tensor(label_std, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample['path']).convert('RGB')
        image = self.transform(image)
        target_raw = torch.tensor(sample['target'], dtype=torch.float32)
        target_norm = (target_raw - self.label_mean) / self.label_std
        return image, target_norm, target_raw


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 for XY regression from filename labels.')
    parser.add_argument('--dataset-dir', default='/home/ubuntu/数据')
    parser.add_argument('--output-pth', default='/home/ubuntu/best_line_follower_model_xy.pth')
    parser.add_argument('--stats-file', default='/home/ubuntu/10_model_convert/label_stats_xy.json')
    parser.add_argument('--summary-file', default='/home/ubuntu/10_model_convert/training_summary_xy.json')
    parser.add_argument('--split-file', default='/home/ubuntu/10_model_convert/train_split_xy.json')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pretrained', action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(dataset_dir):
    dataset_dir = Path(dataset_dir)
    samples = []
    for path in sorted(dataset_dir.glob('*.jpg')):
        match = PATTERN.match(path.name)
        if not match:
            continue
        x_value = float(match.group(1))
        y_value = float(match.group(2))
        samples.append({'path': str(path), 'target': [x_value, y_value], 'name': path.name})
    if not samples:
        raise RuntimeError(f'No labelled jpg images found in {dataset_dir}')
    return samples


def split_samples(samples, val_ratio, seed):
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def compute_label_stats(samples):
    xs = [sample['target'][0] for sample in samples]
    ys = [sample['target'][1] for sample in samples]
    mean = [sum(xs) / len(xs), sum(ys) / len(ys)]
    std = []
    for values, mean_value in [(xs, mean[0]), (ys, mean[1])]:
        variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
        std_value = math.sqrt(variance)
        std.append(std_value if std_value > 1e-6 else 1.0)
    return mean, std


def make_model(use_pretrained):
    weights = None
    if use_pretrained:
        try:
            weights = ResNet18_Weights.DEFAULT
        except Exception:
            weights = None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        print(f'Falling back to random init because pretrained weights are unavailable: {exc}')
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets_norm, _ in loader:
        images = images.to(device, non_blocking=True)
        targets_norm = targets_norm.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, targets_norm)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_mean, label_std):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_abs_error = torch.zeros(2, dtype=torch.float64)
    label_mean = label_mean.to(device)
    label_std = label_std.to(device)
    for images, targets_norm, targets_raw in loader:
        images = images.to(device, non_blocking=True)
        targets_norm = targets_norm.to(device, non_blocking=True)
        targets_raw = targets_raw.to(device, non_blocking=True)
        preds_norm = model(images)
        loss = criterion(preds_norm, targets_norm)
        preds_raw = preds_norm * label_std + label_mean
        total_abs_error += (preds_raw - targets_raw).abs().sum(dim=0).cpu().double()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    avg_loss = total_loss / max(1, total_samples)
    avg_mae = (total_abs_error / max(1, total_samples)).tolist()
    return avg_loss, avg_mae


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        device = 'cpu'

    dataset_dir = Path(args.dataset_dir)
    output_pth = Path(args.output_pth)
    stats_file = Path(args.stats_file)
    summary_file = Path(args.summary_file)
    split_file = Path(args.split_file)

    output_pth.parent.mkdir(parents=True, exist_ok=True)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples(dataset_dir)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    label_mean, label_std = compute_label_stats(train_samples)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = XYImageDataset(train_samples, train_transform, label_mean, label_std)
    val_dataset = XYImageDataset(val_samples, val_transform, label_mean, label_std)

    pin_memory = device.startswith('cuda')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    model = make_model(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.SmoothL1Loss(beta=1.0)

    label_mean_tensor = torch.tensor(label_mean, dtype=torch.float32)
    label_std_tensor = torch.tensor(label_std, dtype=torch.float32)

    best_val_loss = float('inf')
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, label_mean_tensor, label_std_tensor)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae_x': val_mae[0],
            'val_mae_y': val_mae[1],
            'lr': current_lr,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_mae=({val_mae[0]:.2f}, {val_mae[1]:.2f}) | lr={current_lr:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_pth)

    stats_payload = {
        'label_mean': label_mean,
        'label_std': label_std,
        'image_size': args.image_size,
        'imagenet_mean': IMAGENET_MEAN,
        'imagenet_std': IMAGENET_STD,
        'train_count': len(train_samples),
        'val_count': len(val_samples),
        'seed': args.seed,
        'dataset_dir': str(dataset_dir),
        'output_pth': str(output_pth),
    }
    stats_file.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    split_payload = {
        'train': [sample['name'] for sample in train_samples],
        'val': [sample['name'] for sample in val_samples],
    }
    split_file.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    summary_payload = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'history': history,
        'device': device,
        'pretrained': bool(args.pretrained),
        'output_pth': str(output_pth),
        'stats_file': str(stats_file),
        'split_file': str(split_file),
    }
    summary_file.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print()
    print('Training complete.')
    print('best checkpoint =', output_pth)
    print('label stats     =', stats_file)
    print('summary         =', summary_file)
    print('split           =', split_file)


if __name__ == '__main__':
    main()
