"""
Demonstration of SHAMN architecture with reasonable settings for testing
"""
import torch
import torch.nn as nn
from shamn import SHAMN, CPUConfig, CPUDistributedTrainer, generate_dummy_data, benchmark_model, get_model_size
import json
import os
from dataclasses import asdict

def main():
    """Run a demonstration with reasonable settings"""
    print("=" * 70)
    print("SHAMN: Sparse Hierarchical Adaptive Memory Network")
    print("CPU-Optimized AI Architecture Demonstration")
    print("=" * 70)
    print()

    # Reasonable config for demonstration
    config = CPUConfig(
        vocab_size=5000,
        dim=256,
        n_layers=4,
        n_experts=8,
        expert_k=2,
        block_size=64,
        sparsity=0.90,
        memory_size=2048,
        batch_size=32,
        seq_len=128,
        lr=3e-4,
        n_cores=min(os.cpu_count() or 1, 8),  # Use available cores, max 8
        epochs=2  # Just 2 epochs for demo
    )

    print("Configuration:")
    print("-" * 70)
    for key, value in asdict(config).items():
        print(f"  {key:20s}: {value}")
    print()

    # Configure PyTorch for CPU
    torch.set_num_threads(config.n_cores)
    torch.set_num_interop_threads(config.n_cores)

    if torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
        print("✓ MKL-DNN enabled")

    if torch.backends.openmp.is_available():
        print("✓ OpenMP available")

    print(f"✓ Using {config.n_cores} cores")
    print()

    # Create model
    print("=" * 70)
    print("Creating Model")
    print("=" * 70)
    model = SHAMN(config)

    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Print model info
    model_info = get_model_size(model)
    print(f"Total parameters:     {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")
    print(f"Model size:           {model_info['size_mb']:.2f} MB")
    print()

    # Initial benchmark
    print("=" * 70)
    print("Initial Benchmark")
    print("=" * 70)
    bench = benchmark_model(model, config)
    print(json.dumps(bench, indent=2))
    print()

    # Generate training data
    print("=" * 70)
    print("Generating Training Data")
    print("=" * 70)
    data_path = generate_dummy_data(config, num_tokens=500000)
    print()

    # Create trainer
    print("=" * 70)
    print("Training")
    print("=" * 70)
    trainer = CPUDistributedTrainer(model, config, data_path)

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 70)

        avg_loss = trainer.train_epoch()
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': avg_loss,
            'config': asdict(config)
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    # Final benchmark
    print()
    print("=" * 70)
    print("Final Benchmark")
    print("=" * 70)
    bench = benchmark_model(model, config)
    print(json.dumps(bench, indent=2))
    print()

    # Performance summary
    print("=" * 70)
    print("Performance Summary")
    print("=" * 70)
    if trainer.throughput:
        import numpy as np
        print(f"Average throughput:     {np.mean(trainer.throughput):.2f} tokens/sec")
        print(f"Peak throughput:        {np.max(trainer.throughput):.2f} tokens/sec")
        print(f"Min throughput:         {np.min(trainer.throughput):.2f} tokens/sec")

    if trainer.losses:
        print(f"Final training loss:    {trainer.losses[-1]:.4f}")
        print(f"Initial loss:           {trainer.losses[0]:.4f}")
        print(f"Loss improvement:       {trainer.losses[0] - trainer.losses[-1]:.4f}")

    print()

    # Cleanup
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"✓ Cleaned up {data_path}")

    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
