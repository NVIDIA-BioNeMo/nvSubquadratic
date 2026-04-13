"""Shared embedding module for ARC-AGI networks (ARCViT and ARCResNet)."""

import torch
from torch import nn


class ARCColorTaskEmbedding(nn.Module):
    """Embeds discrete colors and a task ID into continuous representations."""

    def __init__(self, num_colors: int, num_tasks: int, hidden_dim: int, num_task_tokens: int = 1):
        """Initialise embedding tables with truncated-normal weights."""
        super().__init__()
        self.num_colors = num_colors
        self.num_task_tokens = num_task_tokens
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        self.task_embed = nn.Embedding(num_tasks, hidden_dim * num_task_tokens)

        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.task_embed.weight, std=0.02)

    def forward(self, pixel_values: torch.Tensor, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed pixel colors and task IDs into continuous vectors.

        Args:
            pixel_values: [B, H, W] int tensor of color indices.
            task_ids: [B] int tensor of task IDs.

        Returns:
            x: [B, H, W, hidden_dim] float tensor of color embeddings.
            task_tok: [B, hidden_dim * num_task_tokens] float tensor of task embeddings.
        """
        x = self.color_embed(pixel_values.long().clamp(0, self.num_colors - 1))
        task_tok = self.task_embed(task_ids.long())
        return x, task_tok
