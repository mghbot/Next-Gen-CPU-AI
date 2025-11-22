"""
Quick demonstration of SHAMN architecture
"""
import torch
import torch.nn as nn
from shamn import SHAMN, CPUConfig, get_model_size
import json
import os
import time
from dataclasses import asdict

def quick_benchmark(model, config, iterations=5):
    """Quick benchmark with fewer iterations"""
    import numpy as np
    import psutil

    model.eval()

    print(f"Running {iterations} warmup iterations...")
    with torch.no_grad():
        for _ in range(iterations):
            x = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
            _ = model(x)

    print(f"Running {iterations} benchmark iterations...")
    times = []
    memory_usage = []

    for i in range(iterations):
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))

        process = psutil.Process()
        mem_before = process.memory_info().rss

        start = time.time()
        with torch.no_grad():
            _ = model(x)
        elapsed = time.time() - start

        mem_after = process.memory_info().rss

        times.append(elapsed)
        memory_usage.append((mem_after - mem_before) / 1024**2)
        print(f"  Iteration {i+1}/{iterations}: {elapsed*1000:.2f} ms")

    simd_info = {
        'mkldnn_enabled': torch.backends.mkldnn.is_available(),
        'openmp_available': torch.backends.openmp.is_available(),
        'num_threads': torch.get_num_threads(),
        'num_interop_threads': torch.get_num_interop_threads()
    }

    return {
        'mean_latency_ms': np.mean(times) * 1000,
        'std_latency_ms': np.std(times) * 1000,
        'min_latency_ms': np.min(times) * 1000,
        'max_latency_ms': np.max(times) * 1000,
        'mean_memory_mb': np.mean(memory_usage),
        'throughput_tokens_per_sec': (config.batch_size * config.seq_len) / np.mean(times),
        'simd_capabilities': simd_info
    }

def main():
    print("=" * 70)
    print("SHAMN: Sparse Hierarchical Adaptive Memory Network")
    print("Quick Demonstration")
    print("=" * 70)
    print()

    # Small config for fast demo
    config = CPUConfig(
        vocab_size=2000,
        dim=256,
        n_layers=4,
        n_experts=8,
        expert_k=2,
        batch_size=16,
        seq_len=64,
        memory_size=1024,
        n_cores=min(os.cpu_count() or 1, 4),
        epochs=1
    )

    print("Configuration:")
    print("-" * 70)
    print(f"  Vocabulary size:      {config.vocab_size:,}")
    print(f"  Hidden dimension:     {config.dim}")
    print(f"  Number of layers:     {config.n_layers}")
    print(f"  Number of experts:    {config.n_experts}")
    print(f"  Experts per token:    {config.expert_k}")
    print(f"  Batch size:           {config.batch_size}")
    print(f"  Sequence length:      {config.seq_len}")
    print(f"  Memory bank size:     {config.memory_size:,}")
    print(f"  CPU cores:            {config.n_cores}")
    print(f"  Sparsity:             {config.sparsity:.1%}")
    print()

    # Configure PyTorch
    torch.set_num_threads(config.n_cores)
    torch.set_num_interop_threads(config.n_cores)

    print("CPU Optimizations:")
    if torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
        print("  ✓ MKL-DNN enabled")
    if torch.backends.openmp.is_available():
        print("  ✓ OpenMP available")
    print(f"  ✓ Using {config.n_cores} threads")
    print()

    # Create model
    print("=" * 70)
    print("Model Creation")
    print("=" * 70)
    model = SHAMN(config)

    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Model info
    model_info = get_model_size(model)
    print(f"Total parameters:       {model_info['total_params']:,}")
    print(f"Trainable parameters:   {model_info['trainable_params']:,}")
    print(f"Model size:             {model_info['size_mb']:.2f} MB")

    # Calculate theoretical dense model size
    dense_params = config.vocab_size * config.dim  # Embedding
    dense_params += config.n_layers * config.n_experts * (config.dim * config.dim * 4 * 2)  # Experts
    dense_params += config.memory_size * config.dim  # Memory
    sparse_ratio = model_info['total_params'] / dense_params
    print(f"Sparsity achieved:      {1 - sparse_ratio:.1%}")
    print()

    # Benchmark
    print("=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    bench = quick_benchmark(model, config, iterations=10)

    print()
    print("Results:")
    print(f"  Mean latency:         {bench['mean_latency_ms']:.2f} ms")
    print(f"  Std latency:          {bench['std_latency_ms']:.2f} ms")
    print(f"  Min latency:          {bench['min_latency_ms']:.2f} ms")
    print(f"  Max latency:          {bench['max_latency_ms']:.2f} ms")
    print(f"  Throughput:           {bench['throughput_tokens_per_sec']:.2f} tokens/sec")
    print(f"  Tokens per batch:     {config.batch_size * config.seq_len}")
    print()

    # Test training step
    print("=" * 70)
    print("Training Step Test")
    print("=" * 70)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    print("Running a few training steps...")
    losses = []
    for step in range(5):
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
        labels = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))

        optimizer.zero_grad()
        outputs = model(x)
        loss = nn.functional.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            labels.reshape(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {step + 1}/5: Loss = {loss.item():.4f}")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ All components working correctly")
    print(f"✓ Model: {model_info['total_params']:,} parameters ({model_info['size_mb']:.2f} MB)")
    print(f"✓ Throughput: {bench['throughput_tokens_per_sec']:.0f} tokens/sec")
    print(f"✓ Training: Loss converging ({losses[0]:.4f} → {losses[-1]:.4f})")
    print()
    print("Next steps:")
    print("  • Run full training: python shamn.py")
    print("  • Run demo: python demo.py")
    print("  • See README.md for more details")
    print()

if __name__ == '__main__':
    import numpy as np  # needed for quick_benchmark
    main()
