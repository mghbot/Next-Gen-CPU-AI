import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
import hashlib
import time
import os
from functools import partial
from typing import Tuple, List, Optional
import psutil
import threading
from queue import Queue
import struct
from pathlib import Path
import json
from dataclasses import dataclass, asdict

@dataclass
class CPUConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_experts: int = 16
    expert_k: int = 2
    block_size: int = 64
    sparsity: float = 0.90
    memory_size: int = 65536
    batch_size: int = 96
    seq_len: int = 512
    lr: float = 3e-4
    n_cores: int = 12
    epochs: int = 10

class SparseHashEmbedding(nn.Module):
    """Hash-based sparse embedding that reduces memory by using hash collisions"""
    def __init__(self, vocab_size: int, dim: int, sparsity: float = 0.95):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.hash_size = max(int(vocab_size * (1 - sparsity)), 1)
        self.weight = nn.Parameter(torch.randn(self.hash_size, dim) * 0.02)
        self.hash_keys = nn.Parameter(torch.randint(0, 2**31, (vocab_size,)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.remainder(self.hash_keys[x], self.hash_size)
        return F.embedding(indices, self.weight)

class BlockSparseLinear(nn.Module):
    """Block-sparse linear layer optimized for CPU cache"""
    def __init__(self, in_features: int, out_features: int, block_size: int = 64, sparsity: float = 0.9):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        n_blocks_in = (in_features + block_size - 1) // block_size
        n_blocks_out = (out_features + block_size - 1) // block_size

        # Create sparse block mask
        mask = torch.rand(n_blocks_out, n_blocks_in) > sparsity
        # Ensure at least some blocks are active
        if mask.sum() == 0:
            mask[0, 0] = True

        self.register_buffer('block_mask', mask)

        nnz_blocks = mask.sum().item()
        self.weight = nn.Parameter(torch.randn(nnz_blocks, block_size, block_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.out_block_idx, self.in_block_idx = mask.nonzero(as_tuple=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D (batch, in_features) and 3D (batch, seq, in_features) inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, x.shape[-1])

        batch_size = x.shape[0]
        out = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)

        for i, (out_b, in_b) in enumerate(zip(self.out_block_idx, self.in_block_idx)):
            in_start = in_b * self.block_size
            in_end = min(in_start + self.block_size, self.in_features)
            out_start = out_b * self.block_size
            out_end = min(out_start + self.block_size, self.out_features)

            # Skip if block dimensions are invalid
            if in_end <= in_start or out_end <= out_start:
                continue

            x_block = x[:, in_start:in_end]
            w_block = self.weight[i, :in_end-in_start, :out_end-out_start]

            # Skip if either dimension is empty
            if x_block.shape[1] == 0 or w_block.shape[1] == 0:
                continue

            out[:, out_start:out_end] += x_block @ w_block

        out = out + self.bias

        # Restore original shape if needed
        if len(original_shape) == 3:
            out = out.reshape(original_shape[0], original_shape[1], -1)

        return out

class DynamicRouter(nn.Module):
    """Top-K router for sparse mixture of experts"""
    def __init__(self, input_dim: int, n_experts: int, k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.gate = nn.Linear(input_dim, n_experts, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x) / (self.temperature.abs() + 1e-5)
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        routing_weights = torch.softmax(top_k_logits, dim=-1)

        # Handle both 2D (batch, dim) and 3D (batch, seq, dim) inputs
        routing_matrix = torch.zeros(*x.shape[:-1], self.n_experts, device=x.device)
        routing_matrix.scatter_(-1, top_k_indices, routing_weights)

        return routing_matrix, top_k_indices

class SparseExpert(nn.Module):
    """Single expert with block-sparse layers"""
    def __init__(self, dim: int, hidden_dim: int, block_size: int = 64):
        super().__init__()
        self.w1 = BlockSparseLinear(dim, hidden_dim, block_size, sparsity=0.85)
        self.w2 = BlockSparseLinear(hidden_dim, dim, block_size, sparsity=0.85)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x)))

class HierarchicalSparseLayer(nn.Module):
    """Hierarchical MoE layer with multiple routing depths"""
    def __init__(self, dim: int, n_experts: int = 16, k: int = 2, depth: int = 2):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.k = k
        self.depth = depth

        self.experts = nn.ModuleList([SparseExpert(dim, dim * 4) for _ in range(n_experts)])
        self.routers = nn.ModuleList([DynamicRouter(dim, n_experts, k) for _ in range(depth)])
        self.cross_layer = BlockSparseLinear(dim, dim, sparsity=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Handle both 2D (batch, dim) and 3D (batch, seq, dim) inputs
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, dim = x.shape
            x = x.reshape(-1, dim)
            residual_flat = residual.reshape(-1, dim)
        else:
            residual_flat = residual

        for depth_idx in range(self.depth):
            routing_matrix, expert_indices = self.routers[depth_idx](x)
            expert_outputs = torch.zeros_like(x)

            for i, expert in enumerate(self.experts):
                mask = routing_matrix[:, i] > 0
                if mask.any():
                    expert_input = x[mask]
                    weighted_output = expert(expert_input) * routing_matrix[mask, i].unsqueeze(-1)
                    expert_outputs[mask] += weighted_output

        output = residual_flat + self.cross_layer(expert_outputs)

        # Restore original shape if needed
        if len(original_shape) == 3:
            output = output.reshape(batch_size, seq_len, -1)

        return output

class AdaptiveMemoryBank(nn.Module):
    """External memory with sparse retrieval"""
    def __init__(self, memory_size: int = 65536, dim: int = 512, k: int = 32):
        super().__init__()
        self.memory_size = memory_size
        self.dim = dim
        self.k = min(k, memory_size)

        self.memory = nn.Parameter(torch.randn(memory_size, dim) * 0.02)

        self.query_proj = BlockSparseLinear(dim, dim, sparsity=0.9)
        self.key_proj = BlockSparseLinear(dim, dim, sparsity=0.9)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        # Handle both 2D (batch, dim) and 3D (batch, seq, dim) inputs
        original_shape = query.shape
        if query.dim() == 3:
            batch_size, seq_len, dim = query.shape
            query = query.reshape(-1, dim)
        else:
            batch_size, dim = query.shape
            seq_len = 1

        q = self.query_proj(query)
        k = self.key_proj(self.memory)

        # Compute pairwise distances
        scores = torch.cdist(q.unsqueeze(0), k.unsqueeze(0), p=2).squeeze(0)

        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1, largest=False)

        # Gather retrieved memories
        retrieved = self.memory[top_k_indices]
        weights = torch.softmax(-top_k_scores, dim=-1).unsqueeze(-1)
        output = (retrieved * weights).sum(dim=1)

        # Restore original shape if needed
        if len(original_shape) == 3:
            output = output.reshape(batch_size, seq_len, -1)

        return output

class SHAMN(nn.Module):
    """Sparse Hierarchical Adaptive Memory Network"""
    def __init__(self, config: CPUConfig):
        super().__init__()
        self.config = config

        self.embedding = SparseHashEmbedding(config.vocab_size, config.dim, sparsity=config.sparsity)
        self.pos_encoding = nn.Parameter(torch.randn(2048, config.dim) * 0.02)

        self.layers = nn.ModuleList([
            HierarchicalSparseLayer(config.dim, config.n_experts, config.expert_k)
            for _ in range(config.n_layers)
        ])

        self.memory = AdaptiveMemoryBank(config.memory_size, config.dim)
        self.output_proj = BlockSparseLinear(config.dim, config.vocab_size, sparsity=config.sparsity)
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.embedding(x) + self.pos_encoding[:seq_len]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x + self.memory(x))
        return self.output_proj(x)

class CPUOptimizedDataLoader:
    """Memory-mapped data loader optimized for CPU"""
    def __init__(self, data_path: str, config: CPUConfig):
        self.config = config
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = np.memmap(data_path, dtype=np.int32, mode='r')
        self.total_tokens = len(self.data)

        if self.total_tokens < config.seq_len + 1:
            raise ValueError(f"Not enough tokens in data file. Need at least {config.seq_len + 1}, got {self.total_tokens}")

        self.samples = (self.total_tokens - config.seq_len - 1) // config.batch_size

    def __iter__(self):
        buffer = torch.empty(self.config.batch_size, self.config.seq_len + 1, dtype=torch.long)

        for i in range(0, self.samples, self.config.batch_size):
            start_idx = i * self.config.batch_size
            end_idx = start_idx + self.config.batch_size * (self.config.seq_len + 1)

            if end_idx > self.total_tokens:
                break

            # Load data in chunks
            chunk = self.data[start_idx:end_idx]
            buffer_data = chunk.reshape(self.config.batch_size, -1)

            if buffer_data.shape[1] < self.config.seq_len + 1:
                break

            # Make a copy to avoid non-writable array warning
            buffer.copy_(torch.from_numpy(np.array(buffer_data)))
            yield buffer[:, :-1].contiguous(), buffer[:, 1:].contiguous()

    def __len__(self):
        return self.samples

class CPUDistributedTrainer:
    """CPU-optimized trainer with intra-op parallelism"""
    def __init__(self, model: nn.Module, config: CPUConfig, data_path: str):
        self.model = model
        self.config = config

        # Verify data exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")

        self.dataloader = CPUOptimizedDataLoader(data_path, config)
        self.optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
        self.accumulation_steps = 4

        # Set CPU affinity if available
        if hasattr(os, 'sched_setaffinity'):
            try:
                os.sched_setaffinity(0, range(config.n_cores))
            except Exception as e:
                print(f"Warning: Could not set CPU affinity: {e}")

        self.throughput = []
        self.losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        steps = 0

        start_time = time.time()

        try:
            for input_ids, labels in self.dataloader:
                # Forward pass
                self.optimizer.zero_grad()

                outputs = self.model(input_ids)
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    labels.reshape(-1),
                    reduction='mean'
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step
                self.optimizer.step()

                total_loss += loss.item()
                steps += 1

                if steps % 10 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = (steps * self.config.batch_size * self.config.seq_len) / elapsed
                    self.throughput.append(tokens_per_sec)
                    avg_loss = total_loss / steps
                    self.losses.append(avg_loss)
                    print(f"Step {steps}, Loss: {avg_loss:.4f}, Throughput: {tokens_per_sec:.2f} tokens/sec")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()

        if steps == 0:
            return 0.0

        return total_loss / steps

def benchmark_model(model: nn.Module, config: CPUConfig):
    """Benchmark model performance"""
    model.eval()

    print("Warming up model...")
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            x = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
            _ = model(x)

    print("Running benchmark...")
    times = []
    memory_usage = []

    for i in range(20):
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

        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/20 iterations")

    # Get SIMD info
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

def generate_dummy_data(config: CPUConfig, num_tokens: int = 1000000):
    """Generate dummy training data"""
    print(f"Generating {num_tokens} tokens of dummy data...")
    data = np.random.randint(0, config.vocab_size, size=(num_tokens,), dtype=np.int32)
    data_path = 'train.dat'
    data.tofile(data_path)
    print(f"Data saved to {data_path} ({os.path.getsize(data_path) / 1024 / 1024:.2f} MB)")
    return data_path

def get_model_size(model: nn.Module):
    """Calculate model size in parameters and MB"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel()

    for buffer in model.buffers():
        buffer_size += buffer.numel()

    total_size = param_size + buffer_size
    size_mb = total_size * 4 / 1024 / 1024  # Assuming float32

    return {
        'total_params': total_size,
        'trainable_params': param_size,
        'buffers': buffer_size,
        'size_mb': size_mb
    }

def main():
    """Main training loop"""
    print("=== SHAMN: Sparse Hierarchical Adaptive Memory Network ===\n")

    # Configuration
    config = CPUConfig()

    # Detect available cores
    available_cores = os.cpu_count() or 1
    config.n_cores = min(config.n_cores, available_cores)
    print(f"Using {config.n_cores} cores (out of {available_cores} available)")

    # Configure PyTorch for CPU
    torch.set_num_threads(config.n_cores)
    torch.set_num_interop_threads(config.n_cores)

    if torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
        print("MKL-DNN enabled")

    if torch.backends.openmp.is_available():
        print("OpenMP available")

    print(f"\nConfiguration:")
    print(json.dumps(asdict(config), indent=2))

    # Create model
    print("\n=== Creating Model ===")
    model = SHAMN(config)

    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Print model info
    model_info = get_model_size(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")
    print(f"  Model size: {model_info['size_mb']:.2f} MB")

    # Initial benchmark
    print("\n=== Initial Benchmark ===")
    bench = benchmark_model(model, config)
    print(json.dumps(bench, indent=2))

    # Generate training data
    print("\n=== Generating Training Data ===")
    data_path = generate_dummy_data(config, num_tokens=1000000)

    # Create trainer
    print("\n=== Creating Trainer ===")
    trainer = CPUDistributedTrainer(model, config, data_path)

    # Training
    print("\n=== Training ===")
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        print("-" * 50)

        avg_loss = trainer.train_epoch()
        print(f"\nEpoch {epoch + 1} completed - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': avg_loss,
            'config': asdict(config)
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Final benchmark
    print("\n=== Final Benchmark ===")
    bench = benchmark_model(model, config)
    print(json.dumps(bench, indent=2))

    # Cleanup
    if os.path.exists('train.dat'):
        os.remove('train.dat')
        print("\nCleaned up training data")

    print("\n=== Training Complete ===")
    print(f"Average throughput: {np.mean(trainer.throughput):.2f} tokens/sec")
    print(f"Final loss: {trainer.losses[-1] if trainer.losses else 'N/A':.4f}")

if __name__ == '__main__':
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()
