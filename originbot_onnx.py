import argparse
import inspect
from pathlib import Path

import torch
import torchvision

try:
    import onnx
except ImportError:
    onnx = None


def build_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 2)
    return model


def load_state_dict(pth_path: Path):
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(str(pth_path), **load_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Export OriginBot ResNet18 checkpoint to ONNX.")
    parser.add_argument(
        "--pth",
        default="./best_line_follower_model_xy.pth",
        help="Path to trained checkpoint (.pth)",
    )
    parser.add_argument(
        "--onnx-out",
        default="./best_line_follower_model_xy.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    opts = parser.parse_args()

    pth_path = Path(opts.pth).resolve()
    onnx_path = Path(opts.onnx_out).resolve()

    if not pth_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    model = build_model()
    state_dict = load_state_dict(pth_path)
    model.load_state_dict(state_dict)
    model.eval()

    sample = torch.randn(1, 3, 224, 224, requires_grad=False)
    export_kwargs = dict(
        export_params=True,
        opset_version=opts.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, sample, str(onnx_path), **export_kwargs)
    print(f"[DONE] ONNX exported: {onnx_path}")

    if onnx is not None:
        net = onnx.load(str(onnx_path))
        onnx.checker.check_model(net)
        print("[DONE] ONNX check passed.")
    else:
        print("[INFO] Install `onnx` to run post-export model checks.")


if __name__ == "__main__":
    main()
