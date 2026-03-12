from unet import UNet
import torch
import argparse
import os

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
    pth_model_path = "/data/atlas/atlasdata3/maggiechen/ESL_pth_to_onnx/Ortho_UNet_SEM_SEGMENTATION_20250821_213112.pth"
    save_path = "/data/atlas/atlasdata3/maggiechen/ESL_pth_to_onnx/Ortho/"
elif args.model == "unortho":
    pth_model_path = "/data/atlas/atlasdata3/maggiechen/ESL_pth_to_onnx/Unortho_UNet_SEM_SEGMENTATION_20250826_154450_copy_around_54.pth"
    save_path = "/data/atlas/atlasdata3/maggiechen/ESL_pth_to_onnx/Unortho/"
else:
    print("Model needs to be either ortho or unortho")

# Load your PyTorch model
pth_model = UNet(input_bands=86, output_classes=1, hidden_channels=16)
pth_model.load_state_dict(torch.load(pth_model_path, map_location="cpu"))
pth_model.eval()

# Create dummy input
dummy_input = torch.randn(1, 86, 128, 128)

# Define ONNX models to benchmark
onnx_models = {
    'ONNX (FP32)': os.path.join(save_path, "unet_model_simp.onnx"),
    'ONNX (FP16)': os.path.join(save_path, "unet_model_fp16.onnx"),
    'ONNX (Quantized Int8)': os.path.join(save_path, "unet_model_quant.onnx"),
}

# Run benchmark
from benchmark_inference import main
results = main(pth_model, pth_model_path, onnx_models, dummy_input, num_runs=100, use_gpu=True)
