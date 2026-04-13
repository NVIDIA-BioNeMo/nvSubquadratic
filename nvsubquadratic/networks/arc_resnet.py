"""ARC-AGI ResNet wrapper: colour embedding + task token → ResidualNetwork."""

from typing import Any, Dict, Literal

import torch
from torch import nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ARCResNet(nn.Module):
    """Thin ARC-AGI wrapper around a ResidualNetwork (typically Hyena-based).

    Handles the ARC-specific input/output contract:
    - Discrete colour embedding: [B, H, W] int → [B, H, W, hidden_dim] float
    - Task-token injection: per-task embedding broadcast-added over H×W so
      every spatial position is conditioned on the task identity.
    - Output permutation: [B, H, W, num_colors] → [B, num_colors, H, W]

    The inner ResidualNetwork receives no explicit conditioning tensor
    (condition_mixer should be Identity in the block configs).
    """

    def __init__(
        self,
        num_tasks: int,
        num_colors: int,
        hidden_dim: int,
        resnet_cfg: LazyConfig,
        task_injection: Literal["broadcast", "film"] = "broadcast",
        cond_dropout_prob: float = 0.0,
    ) -> None:
        """Initialise colour/task embeddings and instantiate the ResidualNetwork."""
        super().__init__()
        self.num_colors = num_colors
        self.task_injection = task_injection
        self.cond_dropout_prob = cond_dropout_prob

        from nvsubquadratic.networks.arc_embedding import ARCColorTaskEmbedding

        self.embedding = ARCColorTaskEmbedding(num_colors=num_colors, num_tasks=num_tasks, hidden_dim=hidden_dim)
        self.resnet = instantiate(resnet_cfg)

    def forward(self, input_and_condition: Dict[str, Any]) -> Dict[str, Any]:
        """Embed colours + task token, run ResNet, return logits [B, C, H, W]."""
        pixel_values = input_and_condition["input"]  # [B, H, W]
        task_ids = input_and_condition["condition"]["task_id"]  # [B]

        x, task_tok = self.embedding(pixel_values, task_ids)

        if self.task_injection == "broadcast":
            x = x + task_tok[:, None, None, :]  # broadcast over H, W
            out = self.resnet({"input": x, "condition": None})
        elif self.task_injection == "film":
            if self.training and self.cond_dropout_prob > 0.0:
                mask = torch.rand(task_tok.shape[0], 1, device=task_tok.device) >= self.cond_dropout_prob
                task_tok = task_tok * mask
            out = self.resnet({"input": x, "condition": task_tok})
        else:
            raise ValueError(f"Unknown task_injection: {self.task_injection}")

        logits = out["logits"].permute(0, 3, 1, 2)  # [B, num_colors, H, W]
        return {"logits": logits}
