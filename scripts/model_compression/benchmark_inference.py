import torch
import onnxruntime as ort
import time
import numpy as np
from pathlib import Path
import pandas as pd
import tracemalloc
import psutil
import os
import platform


def print_hardware_specs():
    """Print hardware specifications"""
    print("\n" + "="*70)
    print("HARDWARE SPECIFICATIONS")
    print("="*70)
    
    # CPU info
    print("\nCPU:")
    print(f"  Processor: {platform.processor()}")
    print(f"  CPU Count: {os.cpu_count()} cores")
    
    # Memory info
    mem = psutil.virtual_memory()
    print(f"\nMemory:")
    print(f"  Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"  Available RAM: {mem.available / (1024**3):.2f} GB")
    print(f"  Used RAM: {mem.used / (1024**3):.2f} GB")
    
    # GPU info
    print(f"\nGPU:")
    if torch.cuda.is_available():
        print(f"  CUDA Available: Yes")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print(f"  CUDA Available: No")
    
    # OS info
    print(f"\nOperating System:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    
    # Framework versions
    print(f"\nFramework Versions:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  ONNX Runtime: {ort.__version__}")
    print(f"  NumPy: {np.__version__}")
    print("="*70)


def benchmark_pytorch_model(model, input_tensor, num_runs=100, warmup_runs=10):
    """Benchmark PyTorch model inference time"""
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return times


def benchmark_onnx_model(onnx_path, input_tensor, num_runs=100, warmup_runs=10, use_gpu=False):
    """Benchmark ONNX model inference time"""
    # Set up providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"Error loading ONNX model {onnx_path}: {e}")
        return None
    
    # Get input name and expected dtype
    input_name = session.get_inputs()[0].name
    expected_dtype = session.get_inputs()[0].type
    
    # Convert tensor to numpy
    input_numpy = input_tensor.numpy()
    
    # Convert to expected dtype if needed
    if 'float16' in str(expected_dtype):
        input_numpy = input_numpy.astype(np.float16)
    elif 'int8' in str(expected_dtype):
        input_numpy = input_numpy.astype(np.int8)
    elif 'float32' in str(expected_dtype):
        input_numpy = input_numpy.astype(np.float32)
    # Add more dtype conversions as needed
    
    # Warmup
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: input_numpy})
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: input_numpy})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return times


def get_model_size(model_path):
    """Get model file size in MB"""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def measure_pytorch_memory(model, input_tensor, num_runs=10):
    """Measure peak memory usage of PyTorch model during inference"""
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    tracemalloc.start()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return peak / (1024)  # Convert to KB


def measure_onnx_memory(onnx_path, input_tensor, num_runs=10):
    """Measure peak memory usage of ONNX model during inference"""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    expected_dtype = session.get_inputs()[0].type
    
    input_numpy = input_tensor.numpy()
    
    if 'float16' in str(expected_dtype):
        input_numpy = input_numpy.astype(np.float16)
    elif 'int8' in str(expected_dtype):
        input_numpy = input_numpy.astype(np.int8)
    elif 'float32' in str(expected_dtype):
        input_numpy = input_numpy.astype(np.float32)
    
    tracemalloc.start()
    
    for _ in range(num_runs):
        _ = session.run(None, {input_name: input_numpy})
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return peak / (1024 )  # Convert to KB


def print_memory_results(memory_results, model_sizes):
    """Print formatted memory usage results"""
    print("\n" + "="*70)
    print("MEMORY FOOTPRINT ANALYSIS")
    print("="*70)
    
    df_data = []
    
    for model_name, peak_memory in memory_results.items():
        if peak_memory is None:
            print(f"\n{model_name}: FAILED TO MEASURE")
            continue
        
        model_size = model_sizes.get(model_name, 0)
        
        print(f"\n{model_name}")
        print("-" * 70)
        print(f"  Model file size:     {model_size:.2f} MB")
        print(f"  Peak memory usage:   {peak_memory:.2f} KB")
        
        df_data.append({
            'Model': model_name,
            'File Size (MB)': model_size,
            'Peak Memory (KB)': peak_memory,
        })
    
    # Create comparison table
    if df_data:
        print("\n" + "="*70)
        print("MEMORY COMPARISON TABLE")
        print("="*70)
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Calculate compression ratios relative to PyTorch
        if len(df_data) > 1:
            pth_size = df_data[0]['File Size (MB)']
            pth_memory = df_data[0]['Peak Memory (KB)']
            print("\n" + "="*70)
            print("COMPRESSION & MEMORY SAVINGS RELATIVE TO PyTorch")
            print("="*70)
            for i, row in enumerate(df_data[1:], 1):
                size_reduction = (1 - row['File Size (MB)'] / pth_size) * 100
                memory_reduction = (1 - row['Peak Memory (KB)'] / pth_memory) * 100
                print(f"{row['Model']:30s}")
                print(f"  File size reduction: {size_reduction:6.1f}%")
                print(f"  Memory savings:      {memory_reduction:6.1f}%")


def print_benchmark_results(results_dict):
    """Print formatted benchmark results"""
    print("\n" + "="*70)
    print("INFERENCE TIME BENCHMARK RESULTS")
    print("="*70)
    
    df_data = []
    
    for model_name, times in results_dict.items():
        if times is None:
            print(f"\n{model_name}: FAILED TO LOAD")
            continue
        
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        
        print(f"\n{model_name}")
        print("-" * 70)
        print(f"  Mean:   {mean_time:.4f} ms")
        print(f"  Median: {median_time:.4f} ms")
        print(f"  Std:    {std_time:.4f} ms")
        print(f"  Min:    {min_time:.4f} ms")
        print(f"  Max:    {max_time:.4f} ms")
        
        df_data.append({
            'Model': model_name,
            'Mean (ms)': mean_time,
            'Median (ms)': median_time,
            'Std (ms)': std_time,
            'Min (ms)': min_time,
            'Max (ms)': max_time,
        })
    
    # Create comparison table
    if df_data:
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # Calculate speedups relative to PyTorch
        if len(df_data) > 1:
            pth_mean = df_data[0]['Mean (ms)']
            print("\n" + "="*70)
            print("SPEEDUP RELATIVE TO PyTorch")
            print("="*70)
            for i, row in enumerate(df_data[1:], 1):
                speedup = pth_mean / row['Mean (ms)']
                print(f"{row['Model']:30s}: {speedup:.2f}x")


def main(pth_model, pth_model_path, onnx_models, dummy_input, num_runs=100, use_gpu=False):
    """
    Benchmark all models
    
    Args:
        pth_model: PyTorch model instance
        pth_model_path: Path to PyTorch model weights
        onnx_models: Dict of {model_name: onnx_path}
        dummy_input: Dummy input tensor
        num_runs: Number of inference runs for benchmarking
        use_gpu: Whether to use GPU if available
    """
    torch.manual_seed(42)
    np.random.seed(42)
    print_hardware_specs()
    
    results = {}
    memory_results = {}
    model_sizes = {}
    
    # Benchmark PyTorch model
    print("\nBenchmarking PyTorch model...")
    if use_gpu and torch.cuda.is_available():
        pth_model = pth_model.cuda()
        dummy_input = dummy_input.cuda()
        print("Using gpu!")
    
    pth_times = benchmark_pytorch_model(pth_model, dummy_input, num_runs=num_runs)
    results['PyTorch (FP32)'] = pth_times
    
    # Measure PyTorch memory
    print("Measuring PyTorch memory usage...")
    pth_memory = measure_pytorch_memory(pth_model, dummy_input.cpu() if use_gpu else dummy_input)
    memory_results['PyTorch (FP32)'] = pth_memory
    model_sizes['PyTorch (FP32)'] = os.path.getsize(pth_model_path) / (1024 * 1024)
    
    # Benchmark ONNX models
    for model_name, onnx_path in onnx_models.items():
        if not Path(onnx_path).exists():
            print(f"\nONNX model not found: {onnx_path}")
            results[model_name] = None
            memory_results[model_name] = None
            continue
        
        print(f"Benchmarking {model_name}...")
        onnx_times = benchmark_onnx_model(
            onnx_path, 
            dummy_input.cpu(),  # ONNX typically uses CPU
            num_runs=num_runs,
            use_gpu=use_gpu
        )
        results[model_name] = onnx_times
        
        # Measure ONNX memory
        print(f"Measuring {model_name} memory usage...")
        onnx_memory = measure_onnx_memory(onnx_path, dummy_input.cpu())
        memory_results[model_name] = onnx_memory
        model_sizes[model_name] = get_model_size(onnx_path)
    
    # Print results
    print_benchmark_results(results)
    print_memory_results(memory_results, model_sizes)
    
    return results, memory_results, model_sizes


if __name__ == "__main__":
    # Example usage:
    # from your_model import UNet
    # 
    # pth_model = UNet(input_bands=86, output_classes=1, hidden_channels=16)
    # pth_model.load_state_dict(torch.load(pth_model_path, map_location="cpu"))
    # pth_model.eval()
    #
    # dummy_input = torch.randn(1, 86, 128, 128)
    #
    # onnx_models = {
    #     'ONNX (FP32)': "unet_model_simp.onnx",
    #     'ONNX (FP16)': "unet_model_fp16.onnx",
    #     'ONNX (Quantized Int8)': "unet_model_quant.onnx",
    # }
    #
    # inference_times, memory_usage, model_sizes = main(pth_model, pth_model_path, onnx_models, dummy_input, num_runs=100, use_gpu=False)
    
    print("Benchmark script loaded. Use the example code above to run benchmarks.")