"""
Test training with a small dataset
"""
import torch
from shamn import SHAMN, CPUConfig, CPUDistributedTrainer, generate_dummy_data
import os

def test_small_training():
    """Test training with minimal config"""
    print("=== Testing Training Loop ===\n")

    # Small config for fast testing
    config = CPUConfig(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        n_experts=4,
        expert_k=2,
        batch_size=16,
        seq_len=64,
        lr=1e-3,
        n_cores=min(4, os.cpu_count() or 1),
        epochs=1,
        memory_size=512
    )

    print(f"Config:")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Dim: {config.dim}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Experts: {config.n_experts}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Cores: {config.n_cores}\n")

    # Set thread count
    torch.set_num_threads(config.n_cores)

    # Create model
    print("Creating model...")
    model = SHAMN(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")

    # Generate small dataset
    print("Generating test data...")
    num_tokens = 50000  # Small dataset for fast testing
    data_path = generate_dummy_data(config, num_tokens=num_tokens)
    print()

    # Create trainer
    print("Creating trainer...")
    trainer = CPUDistributedTrainer(model, config, data_path)
    print()

    # Train for 1 epoch
    print("Training for 1 epoch...")
    print("-" * 60)
    avg_loss = trainer.train_epoch()
    print("-" * 60)
    print(f"\nAverage loss: {avg_loss:.4f}")

    if len(trainer.losses) > 0:
        print(f"Final loss: {trainer.losses[-1]:.4f}")
        print(f"Number of steps: {len(trainer.losses)}")

    if len(trainer.throughput) > 0:
        import numpy as np
        print(f"Average throughput: {np.mean(trainer.throughput):.2f} tokens/sec")
        print(f"Peak throughput: {np.max(trainer.throughput):.2f} tokens/sec")

    # Cleanup
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"\nCleaned up {data_path}")

    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, config.vocab_size, (2, config.seq_len))
        test_output = model(test_input)
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Expected: (2, {config.seq_len}, {config.vocab_size})")

        assert test_output.shape == (2, config.seq_len, config.vocab_size), "Wrong output shape!"

    print("\n✓ Training test passed!")
    return True

if __name__ == '__main__':
    try:
        success = test_small_training()
        if success:
            print("\n🎉 All training tests passed!")
            exit(0)
        else:
            print("\n❌ Training test failed")
            exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
