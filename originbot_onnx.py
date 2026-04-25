import torchvision
import torch
import inspect

try:
    import onnx
except ImportError:
    onnx = None

model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('./best_line_follower_model_xy.pth', map_location=device))
model = model.to(device)
model.eval()
x = torch.randn(1, 3, 224, 224, requires_grad=False).to(device)
export_kwargs = dict(
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
if "dynamo" in inspect.signature(torch.onnx.export).parameters:
    export_kwargs["dynamo"] = False

torch.onnx.export(
    model,
    x,
    "./best_line_follower_model_xy.onnx",
    **export_kwargs
)
if onnx is not None:
    net = onnx.load("./best_line_follower_model_xy.onnx")
    onnx.checker.check_model(net)
    onnx.helper.printable_graph(net.graph)
else:
    print("ONNX export complete. Install `onnx` to run post-export model checks.")
