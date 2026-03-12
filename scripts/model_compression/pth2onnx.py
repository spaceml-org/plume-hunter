import numpy
import torch
from unet import UNet
import argparse
from pathlib import Path
import os

import onnx
from onnxsim import simplify
from onnxoptimizer import optimize
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

torch.manual_seed(42)

def GetParser():
    """Argument parser for reading Ntuples script."""
    parser = argparse.ArgumentParser(
        description="Reading Ntuples command line options."
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Specify the type of model (ortho or unortho)",
    )
    args = parser.parse_args()
    return args


args = GetParser()

if args.model == "ortho":
    pth_model_path = "Ortho_UNet_SEM_SEGMENTATION_20250821_213112.pth"
    save_path = "Ortho/"
elif args.model == "unortho":
    pth_model_path = "Unortho_UNet_SEM_SEGMENTATION_20250826_154450_copy_around_54.pth"
    save_path = "Unortho/"
else:
    print("Model needs to be either ortho or unortho")
os.makedirs(save_path, exist_ok=True)

pth_model = UNet(input_bands=86, output_classes=1, hidden_channels=16)
pth_model.load_state_dict(torch.load(pth_model_path, map_location="cpu"))
pth_model.eval()

num_params = sum(p.numel() for p in pth_model.parameters())
print(f"Number of parameters: {num_params:,}")
print(f"Approx model size (FP32): {num_params * 4 / (1024**2):.2f} MB")


dummy_input = torch.randn(1, 86, 128, 128)
torch.onnx.export(
    pth_model,
    dummy_input,
    os.path.join(save_path, "unet_model.onnx"),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18
)
print("Exported to unet_model.onnx")

onnx_model = onnx.load(os.path.join(save_path,"unet_model.onnx"))
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

passes = [
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_nop_dropout",
    "eliminate_unused_initializer",
    "fuse_add_bias_into_conv"
]
model_path = os.path.join(save_path, "unet_model.onnx") # Load and simplify
simplified_path = os.path.join(save_path, "unet_model_simp.onnx")
model = onnx.load(model_path)
model_simp, check = simplify(model)
model_opt = optimize(model_simp, passes)
onnx.save(model_opt, simplified_path)

model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, os.path.join(save_path, "unet_model_fp16.onnx"))
print("Converted to float16")

quantize_dynamic(
    model_input=simplified_path,
    model_output=os.path.join(save_path, "unet_model_quant.onnx"),
    weight_type=QuantType.QInt8
)
print("Quantised to INT8")

# num_params = sum(p.numel() for p in model["state_dict"].parameters())
# print(f"Number of parameters: {num_params:,}")

size_bytes = os.path.getsize(pth_model_path)
size_mb = size_bytes / (1024 ** 2)
print(f"Pth model file size: {size_mb:.2f} MB")

models = ["unet_model.onnx", "unet_model_simp.onnx", "unet_model_fp16.onnx", "unet_model_quant.onnx"]
for model_file in models:
    try:
        size = Path(os.path.join(save_path, model_file)).stat().st_size / 1024 / 1024
        print(f"{model_file:25s} -> {size:.2f} MB")
    except FileNotFoundError:
        print(f"{model_file:25s} -> Not found")

    
# Run inference
print("Running inference ...")

def load_model(path):
    return ort.InferenceSession(path)

def run_inference(session, x):

    input_tensor = session.get_inputs()[0]
    input_name = input_tensor.name
    input_type = input_tensor.type

    # Convert PyTorch tensor → numpy array
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    # Type casting
    if "float16" in input_type:
        x = x.astype(numpy.float16)
    else:
        x = x.astype(numpy.float32)

    output_name = session.get_outputs()[0].name

    return session.run(
        [output_name],
        {input_name: x}
    )[0]


def eval_pth_model(model, x_tensor):
    model.eval()

    with torch.no_grad():
        out = model(x_tensor)

    return out.detach().cpu().numpy()


batch_size = 1

def mae(a, b):
    return numpy.mean(numpy.abs(a - b))


def rel_error(a, b):
    return numpy.max(numpy.abs(a - b)) / (numpy.max(numpy.abs(a)) + 1e-8)

fp32_session = ort.InferenceSession(os.path.join(save_path, "unet_model.onnx"))
fp16_session = ort.InferenceSession(os.path.join(save_path, "unet_model_fp16.onnx"))
int8_session = ort.InferenceSession(os.path.join(save_path, "unet_model_quant.onnx"))

fp32_out = run_inference(fp32_session, dummy_input)
fp16_out = run_inference(fp16_session, dummy_input)
int8_out = run_inference(int8_session, dummy_input)

pth_out = eval_pth_model(pth_model, dummy_input)

print("PTH vs FP32 ONNX MAE:", mae(pth_out, fp32_out))
print("PTH vs FP16 ONNX MAE:", mae(pth_out, fp16_out))
print("PTH vs INT8 ONNX MAE:", mae(pth_out, int8_out))

print("PTH vs FP32 ONNX Relative Err:", rel_error(pth_out, fp32_out))
print("PTH vs FP16 ONNX Relative Err:", rel_error(pth_out, fp16_out))
print("PTH vs INT8 ONNX Relative Err:", rel_error(pth_out, int8_out))