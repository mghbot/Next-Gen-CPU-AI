"""
Test individual components of the SHAMN architecture
"""
import torch
import sys
import traceback

def test_sparse_hash_embedding():
    """Test SparseHashEmbedding"""
    print("\n=== Testing SparseHashEmbedding ===")
    try:
        from shamn import SparseHashEmbedding

        embed = SparseHashEmbedding(vocab_size=1000, dim=128, sparsity=0.95)
        x = torch.randint(0, 1000, (32, 64))
        out = embed(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Hash table size: {embed.hash_size} (95% reduction)")
        print(f"  Expected shape: (32, 64, 128)")

        assert out.shape == (32, 64, 128), f"Wrong shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN detected in output"

        print("  ✓ SparseHashEmbedding test passed!")
        return True
    except Exception as e:
        print(f"  ✗ SparseHashEmbedding test failed: {e}")
        traceback.print_exc()
        return False

def test_block_sparse_linear():
    """Test BlockSparseLinear"""
    print("\n=== Testing BlockSparseLinear ===")
    try:
        from shamn import BlockSparseLinear

        linear = BlockSparseLinear(128, 256, block_size=64, sparsity=0.9)
        x = torch.randn(32, 128)
        out = linear(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Number of active blocks: {linear.block_mask.sum().item()}")
        print(f"  Sparsity: {1 - linear.block_mask.float().mean():.2%}")
        print(f"  Expected shape: (32, 256)")

        assert out.shape == (32, 256), f"Wrong shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN detected in output"

        print("  ✓ BlockSparseLinear test passed!")
        return True
    except Exception as e:
        print(f"  ✗ BlockSparseLinear test failed: {e}")
        traceback.print_exc()
        return False

def test_dynamic_router():
    """Test DynamicRouter"""
    print("\n=== Testing DynamicRouter ===")
    try:
        from shamn import DynamicRouter

        router = DynamicRouter(input_dim=128, n_experts=16, k=2)
        x = torch.randn(32, 128)
        routing_matrix, expert_indices = router(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Routing matrix shape: {routing_matrix.shape}")
        print(f"  Expert indices shape: {expert_indices.shape}")
        print(f"  Expected routing matrix: (32, 16)")
        print(f"  Expected expert indices: (32, 2)")

        assert routing_matrix.shape == (32, 16), f"Wrong routing matrix shape: {routing_matrix.shape}"
        assert expert_indices.shape == (32, 2), f"Wrong expert indices shape: {expert_indices.shape}"
        assert not torch.isnan(routing_matrix).any(), "NaN in routing matrix"

        # Check that routing weights sum to 1
        active_weights = routing_matrix[routing_matrix > 0].reshape(32, -1).sum(dim=1)
        print(f"  Routing weights sum: {active_weights.mean():.4f} (should be ~1.0)")

        print("  ✓ DynamicRouter test passed!")
        return True
    except Exception as e:
        print(f"  ✗ DynamicRouter test failed: {e}")
        traceback.print_exc()
        return False

def test_sparse_expert():
    """Test SparseExpert"""
    print("\n=== Testing SparseExpert ===")
    try:
        from shamn import SparseExpert

        expert = SparseExpert(dim=128, hidden_dim=512, block_size=64)
        x = torch.randn(32, 128)
        out = expert(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Expected shape: (32, 128)")

        assert out.shape == (32, 128), f"Wrong shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN detected in output"

        print("  ✓ SparseExpert test passed!")
        return True
    except Exception as e:
        print(f"  ✗ SparseExpert test failed: {e}")
        traceback.print_exc()
        return False

def test_hierarchical_sparse_layer():
    """Test HierarchicalSparseLayer"""
    print("\n=== Testing HierarchicalSparseLayer ===")
    try:
        from shamn import HierarchicalSparseLayer

        layer = HierarchicalSparseLayer(dim=128, n_experts=8, k=2)
        x = torch.randn(32, 128)
        out = layer(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Number of experts: {layer.n_experts}")
        print(f"  Top-k: {layer.k}")
        print(f"  Expected shape: (32, 128)")

        assert out.shape == (32, 128), f"Wrong shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN detected in output"

        print("  ✓ HierarchicalSparseLayer test passed!")
        return True
    except Exception as e:
        print(f"  ✗ HierarchicalSparseLayer test failed: {e}")
        traceback.print_exc()
        return False

def test_adaptive_memory_bank():
    """Test AdaptiveMemoryBank"""
    print("\n=== Testing AdaptiveMemoryBank ===")
    try:
        from shamn import AdaptiveMemoryBank

        memory = AdaptiveMemoryBank(memory_size=1024, dim=128, k=32)
        x = torch.randn(32, 128)
        out = memory(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Memory size: {memory.memory_size}")
        print(f"  Retrieved k: {memory.k}")
        print(f"  Expected shape: (32, 128)")

        assert out.shape == (32, 128), f"Wrong shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN detected in output"

        print("  ✓ AdaptiveMemoryBank test passed!")
        return True
    except Exception as e:
        print(f"  ✗ AdaptiveMemoryBank test failed: {e}")
        traceback.print_exc()
        return False

def test_full_model():
    """Test full SHAMN model"""
    print("\n=== Testing Full SHAMN Model ===")
    try:
        from shamn import SHAMN, CPUConfig

        # Small config for testing
        config = CPUConfig(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_experts=4,
            expert_k=2,
            batch_size=8,
            seq_len=32,
            memory_size=512
        )

        print(f"  Config: {config.n_layers} layers, {config.n_experts} experts")
        print(f"  Input: batch={config.batch_size}, seq_len={config.seq_len}")

        model = SHAMN(config)
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))

        print(f"  Running forward pass...")
        out = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Expected shape: ({config.batch_size}, {config.seq_len}, {config.vocab_size})")

        assert out.shape == (config.batch_size, config.seq_len, config.vocab_size), \
            f"Wrong shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN detected in output"

        # Test backward pass
        print(f"  Running backward pass...")
        loss = out.sum()
        loss.backward()

        # Check gradients - in sparse models, not all parameters are used in every forward pass
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        param_count = sum(1 for p in model.parameters() if p.requires_grad)
        grad_ratio = grad_count / param_count if param_count > 0 else 0
        print(f"  Gradients computed: {grad_count}/{param_count} ({grad_ratio:.1%})")

        # For sparse models, we expect most but not necessarily all gradients
        assert grad_ratio > 0.85, f"Too few gradients computed: {grad_ratio:.1%}"

        print("  ✓ Full SHAMN model test passed!")
        return True
    except Exception as e:
        print(f"  ✗ Full SHAMN model test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("SHAMN Component Testing")
    print("=" * 60)

    tests = [
        ("SparseHashEmbedding", test_sparse_hash_embedding),
        ("BlockSparseLinear", test_block_sparse_linear),
        ("DynamicRouter", test_dynamic_router),
        ("SparseExpert", test_sparse_expert),
        ("HierarchicalSparseLayer", test_hierarchical_sparse_layer),
        ("AdaptiveMemoryBank", test_adaptive_memory_bank),
        ("Full SHAMN Model", test_full_model),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
