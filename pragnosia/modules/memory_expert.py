"""
Memory Expert for Pragnosia - Long Context Memory.

Implements a memory mechanism inspired by human hippocampal memory:
- Memory bank for storing compressed representations
- Memory attention for retrieval
- Memory gate for selective storage
- Memory consolidation for compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MemoryState:
    """State of the memory expert."""
    memory_bank: torch.Tensor  # [batch, memory_size, hidden_size]
    memory_mask: torch.Tensor  # [batch, memory_size] - which slots are used
    write_pointer: int  # Next position to write


class MemoryExpert(nn.Module):
    """
    Memory Expert for long context understanding.

    Features:
    - Compressed memory bank for storing past context
    - Memory attention mechanism for retrieval
    - Gated memory writing
    - Optional memory consolidation

    Brain Analog: Hippocampus - memory formation and retrieval
    """

    def __init__(
        self,
        hidden_size: int,
        memory_size: int = 256,
        num_heads: int = 8,
        compression_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.compression_ratio = compression_ratio

        # Memory compression: compress tokens before storing
        self.compressor = nn.Sequential(
            nn.Linear(hidden_size * compression_ratio, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Memory query/key/value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Write gate - determines what to store in memory
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Read gate - modulates retrieved memories
        self.read_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _create_memory_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MemoryState:
        """Create initial memory state."""
        return MemoryState(
            memory_bank=torch.zeros(
                batch_size, self.memory_size, self.hidden_size,
                device=device, dtype=dtype
            ),
            memory_mask=torch.zeros(
                batch_size, self.memory_size,
                device=device, dtype=torch.bool
            ),
            write_pointer=0,
        )

    def _compress_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress tokens for memory storage.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            compressed: [batch, seq_len // compression_ratio, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Pad sequence length to be divisible by compression ratio
        pad_len = (self.compression_ratio - seq_len % self.compression_ratio) % self.compression_ratio
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))

        # Reshape for compression
        new_seq_len = hidden_states.shape[1] // self.compression_ratio
        hidden_states = hidden_states.view(
            batch_size, new_seq_len, self.compression_ratio * hidden_size
        )

        # Apply compression
        compressed = self.compressor(hidden_states)

        return compressed

    def _memory_attention(
        self,
        query: torch.Tensor,
        memory_bank: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attend to memory bank.

        Args:
            query: [batch, seq_len, hidden_size]
            memory_bank: [batch, memory_size, hidden_size]
            memory_mask: [batch, memory_size]

        Returns:
            retrieved: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = query.shape

        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(memory_bank)
        v = self.v_proj(memory_bank)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.memory_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.memory_size, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply memory mask (mask unused memory slots)
        if memory_mask is not None:
            # Expand mask for heads and query positions
            mask = memory_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, memory_size]
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        # Softmax (handle all-masked case)
        if memory_mask.any():
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        else:
            attn_probs = torch.zeros_like(attn_scores)

        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        retrieved = torch.matmul(attn_probs, v)

        # Reshape back
        retrieved = retrieved.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        retrieved = self.o_proj(retrieved)

        return retrieved

    def _write_to_memory(
        self,
        hidden_states: torch.Tensor,
        memory_state: MemoryState,
    ) -> MemoryState:
        """
        Write to memory bank.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            memory_state: Current memory state

        Returns:
            Updated memory state
        """
        batch_size = hidden_states.shape[0]

        # Compute write gate
        write_gates = self.write_gate(hidden_states)  # [batch, seq_len, 1]

        # Compress tokens
        compressed = self._compress_tokens(hidden_states)  # [batch, compressed_seq, hidden_size]
        compressed_seq_len = compressed.shape[1]

        # Apply write gate (use mean gate for each compressed segment)
        # Reshape gates to match compressed tokens
        gate_seq_len = write_gates.shape[1]
        pad_len = (self.compression_ratio - gate_seq_len % self.compression_ratio) % self.compression_ratio
        if pad_len > 0:
            write_gates = F.pad(write_gates, (0, 0, 0, pad_len))

        write_gates = write_gates.view(
            batch_size, compressed_seq_len, self.compression_ratio, 1
        ).mean(dim=2)  # [batch, compressed_seq, 1]

        # Gate the compressed tokens
        gated_compressed = compressed * write_gates

        # Write to memory bank (circular buffer)
        memory_bank = memory_state.memory_bank.clone()
        memory_mask = memory_state.memory_mask.clone()

        write_start = memory_state.write_pointer
        for i in range(compressed_seq_len):
            write_idx = (write_start + i) % self.memory_size
            memory_bank[:, write_idx] = gated_compressed[:, i]
            memory_mask[:, write_idx] = True

        new_write_pointer = (write_start + compressed_seq_len) % self.memory_size

        return MemoryState(
            memory_bank=memory_bank,
            memory_mask=memory_mask,
            write_pointer=new_write_pointer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: Optional[MemoryState] = None,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, MemoryState]:
        """
        Forward pass of memory expert.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            memory_state: Optional previous memory state
            update_memory: Whether to update memory

        Returns:
            output: [batch, seq_len, hidden_size]
            memory_state: Updated memory state
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self._create_memory_state(batch_size, device, dtype)

        # Ensure memory state has correct batch size
        if memory_state.memory_bank.shape[0] != batch_size:
            memory_state = self._create_memory_state(batch_size, device, dtype)

        # Read from memory
        retrieved = self._memory_attention(
            hidden_states,
            memory_state.memory_bank,
            memory_state.memory_mask,
        )

        # Compute read gate
        gate_input = torch.cat([hidden_states, retrieved], dim=-1)
        read_gate = self.read_gate(gate_input)  # [batch, seq_len, 1]

        # Gated memory output
        output = hidden_states + read_gate * retrieved
        output = self.norm(output)

        # Update memory if enabled
        if update_memory:
            memory_state = self._write_to_memory(hidden_states, memory_state)

        return output, memory_state

    def reset_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
        """Reset memory state."""
        return self._create_memory_state(batch_size, device, dtype)


class MemoryAugmentedTransformerBlock(nn.Module):
    """
    Transformer block augmented with memory expert.

    Combines standard transformer attention with memory retrieval.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        memory_size: int = 256,
        memory_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Standard self-attention (using existing PragnosiaAttention would require import)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Memory expert
        self.memory_expert = MemoryExpert(
            hidden_size=hidden_size,
            memory_size=memory_size,
            num_heads=memory_heads,
            dropout=dropout,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Memory weight - learned interpolation between self-attn and memory
        self.memory_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: Optional[MemoryState] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, MemoryState]:
        """Forward pass."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        attn_output, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask,
        )

        # Memory retrieval
        memory_output, memory_state = self.memory_expert(
            hidden_states, memory_state
        )

        # Combine with learned weight
        weight = torch.sigmoid(self.memory_weight)
        combined = (1 - weight) * attn_output + weight * memory_output

        hidden_states = residual + combined

        return hidden_states, memory_state
