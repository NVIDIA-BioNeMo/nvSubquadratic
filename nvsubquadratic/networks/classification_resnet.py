# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Simple implementation of a ResNet for classification."""

import torch

from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


class ClassificationResNet(ResidualNetwork):
    """Simple implementation of a ResNet for classification.

    It assumes:
    - the input tensor is of shape (batch_size, *spatial_dims, in_channels).
    - the output tensor is of shape (batch_size, num_classes).

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_blocks (int): Number of blocks
        hidden_dim (int): Number of hidden dimensions
        in_proj_cfg (LazyConfig): Configuration for the input projection
        out_proj_cfg (LazyConfig): Configuration for the output projection
        norm_cfg (LazyConfig): Configuration for the normalization
        block_cfg (LazyConfig): Configuration for the residual block
        dropout_in_cfg (LazyConfig): Configuration for the dropout in layer (applied to the input)
        condition_in_proj_cfg (LazyConfig | None): Configuration for the condition input projection or None if no condition is used.
            If provided, the condition tensor is of shape [B, * spatial_dims_condition, hidden_dim].
            If not provided, the condition tensor is None.
    """

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the ClassificationResNet.

        Args:
            input_and_condition: A dictionary containing the input and condition.
                Keys: "input" and "condition".

            - input: Input tensor of shape [B, * spatial_dims, hidden_dim].
            - condition: Condition tensor of shape [B, * spatial_dims_condition, hidden_dim] or None.

        Returns:
            Dict[str, torch.Tensor]:
                - "logits": tensor of shape [B, out_channels].
        """
        # Extract the input and condition from the dictionary
        x, condition = input_and_condition["input"], input_and_condition["condition"]

        # Apply in_dropout to the input
        x = self.dropout_in(x)
        # Apply input projection
        x = self.in_proj(x)

        # Apply condition input projection if provided
        if self.condition_in_proj is not None:
            assert condition is not None, "Condition must be provided if condition input projection is provided"
            condition = self.condition_in_proj(condition)

        # Apply residual blocks (with or without condition)
        for block in self.blocks:
            x = block(x, condition)

        # Average over the spatial dimensions
        x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
        x = x.mean(dim=1)
        # Apply output norm
        x = self.out_norm(x)
        # Apply output projection
        x = self.out_proj(x)
        return {"logits": x}
