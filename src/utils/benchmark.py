from __future__ import annotations
import time
import torch
from pathlib import Path

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters.
    
    Args:
        model: PyTorch model.
        
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in MB.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Size in MB.
    """
    total, _ = count_parameters(model)
    return total * 4 / (1024 * 1024)  # 4 bytes per float32

def benchmark_inference(model: torch.nn.Module, input_shape: tuple, iterations: int = 10, device: str = "cuda") -> dict:
    """Benchmark inference latency.
    
    Args:
        model: PyTorch model.
        input_shape: Input tensor shape (e.g., (1, 3, 224, 224)).
        iterations: Number of iterations for timing.
        device: Device to run on.
        
    Returns:
        Dictionary with timing results.
    """
    model.eval()
    model = model.to(device)
    dummy = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(dummy)
            times.append(time.time() - start)
    
    times = times[1:]  # Skip first
    return {
        "mean_latency_ms": float(np.mean(times) * 1000),
        "std_latency_ms": float(np.std(times) * 1000),
        "min_latency_ms": float(np.min(times) * 1000),
        "max_latency_ms": float(np.max(times) * 1000),
        "throughput_images_per_sec": float(input_shape[0] / np.mean(times)),
    }

import numpy as np
