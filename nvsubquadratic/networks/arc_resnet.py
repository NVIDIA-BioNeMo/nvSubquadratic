"""ARC-AGI ResNet wrapper: colour embedding + task token → ResidualNetwork."""

from typing import Any, Dict

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
    ) -> None:
        """Initialise colour/task embeddings and instantiate the ResidualNetwork."""
        super().__init__()
        self.num_colors = num_colors
        self.color_embed = nn.Embedding(num_colors, hidden_dim)
        self.task_embed = nn.Embedding(num_tasks, hidden_dim)
        self.resnet = instantiate(resnet_cfg)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.task_embed.weight, std=0.02)

    def forward(self, input_and_condition: Dict[str, Any]) -> Dict[str, Any]:
        """Embed colours + task token, run ResNet, return logits [B, C, H, W]."""
        pixel_values = input_and_condition["input"]  # [B, H, W]
        task_ids = input_and_condition["condition"]["task_id"]  # [B]

        x = self.color_embed(pixel_values.long().clamp(0, self.num_colors - 1))
        # [B, H, W, hidden_dim]
        task_tok = self.task_embed(task_ids.long())  # [B, hidden_dim]
        x = x + task_tok[:, None, None, :]  # broadcast over H, W

        out = self.resnet({"input": x, "condition": None})
        logits = out["logits"].permute(0, 3, 1, 2)  # [B, num_colors, H, W]
        return {"logits": logits}
