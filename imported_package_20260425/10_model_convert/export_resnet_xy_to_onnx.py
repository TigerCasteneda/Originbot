#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import nn
from torchvision.models import resnet18


class DenormalizedRegressor(nn.Module):
    def __init__(self, base_model, label_mean, label_std):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('label_mean', torch.tensor(label_mean, dtype=torch.float32))
        self.register_buffer('label_std', torch.tensor(label_std, dtype=torch.float32))

    def forward(self, x):
        preds_norm = self.base_model(x)
        return preds_norm * self.label_std + self.label_mean


def parse_args():
    parser = argparse.ArgumentParser(description='Export trained ResNet18 XY regressor to ONNX with denormalized outputs.')
    parser.add_argument('--pth', default='/home/ubuntu/best_line_follower_model_xy.pth')
    parser.add_argument('--stats-file', default='/home/ubuntu/10_model_convert/label_stats_xy.json')
    parser.add_argument('--onnx-out', default='/home/ubuntu/10_model_convert/mapper/best_line_follower_model_xy.onnx')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--opset', type=int, default=11)
    return parser.parse_args()


def main():
    args = parse_args()
    pth_path = Path(args.pth)
    stats_path = Path(args.stats_file)
    onnx_path = Path(args.onnx_out)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    label_mean = stats['label_mean']
    label_std = stats['label_std']

    base_model = resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 2)
    state = torch.load(pth_path, map_location='cpu')
    base_model.load_state_dict(state)
    base_model.eval()

    export_model = DenormalizedRegressor(base_model, label_mean, label_std)
    export_model.eval()

    dummy = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    with torch.no_grad():
        preview = export_model(dummy)
        print('preview output =', preview.squeeze(0).tolist())

    torch.onnx.export(
        export_model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output_xy'],
        dynamic_axes=None,
    )
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print('onnx exported to', onnx_path)


if __name__ == '__main__':
    main()
