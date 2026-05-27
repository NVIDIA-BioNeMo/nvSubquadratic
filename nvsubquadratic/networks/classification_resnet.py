# TODO: Add license header here

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""Classification residual network — global-average-pool readout.

:class:`ClassificationResNet` subclasses :class:`ResidualNetwork` and overrides
:meth:`forward` to add a **global average pooling** step before the output
projection, collapsing all spatial positions into a single per-sample vector:

.. code-block:: text

    input [B, *spatial, in_channels]
        ↓  (inherited) dropout_in, in_proj, blocks   → [B, *spatial, hidden_dim]
        ↓  GAP: reshape → [B, T, hidden_dim] → mean over T
        ↓  out_norm                                  → [B, hidden_dim]
        ↓  out_proj                                  → [B, num_classes]
    output {"logits": [B, num_classes]}

The pooling is layout-agnostic: it flattens all spatial dimensions into a
single token axis before taking the mean, so the same network class works for
1-D (sequence), 2-D (image), and 3-D (video / volume) inputs.

All constructor arguments are inherited from
:class:`~nvsubquadratic.networks.general_purpose_resnet.ResidualNetwork`;
``target_size`` is not used by this subclass (the GAP step replaces the
readout-crop mechanism).

Adapted from https://github.com/implicit-long-convs/ccnn_v2.
"""

import torch

from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


class ClassificationResNet(ResidualNetwork):
    """Residual network with global-average-pool readout for classification.

    Inherits the full constructor and backbone from
    :class:`~nvsubquadratic.networks.general_purpose_resnet.ResidualNetwork`.
    Overrides only :meth:`forward` to replace the spatial output with a
    single class-logit vector via global average pooling.

    **Output shape**: ``[B, out_channels]`` regardless of input spatial size —
    the model is therefore resolution-agnostic at inference time.

    **No ``target_size``**: the inherited ``target_size`` attribute is ignored;
    GAP serves as the spatial aggregation step.

    All constructor arguments are documented in
    :class:`~nvsubquadratic.networks.general_purpose_resnet.ResidualNetwork`.
    The typical value for ``out_channels`` here is the number of classes.
    """

    def forward(self, input_and_condition: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run classification forward pass: backbone → GAP → norm → projection.

        Args:
            input_and_condition: Dictionary with two keys:

                * ``"input"`` — signal tensor of shape
                  ``[B, *spatial, in_channels]``.
                * ``"condition"`` — optional conditioning tensor of shape
                  ``[B, *spatial_cond, hidden_dim]``, or ``None``.

        Returns:
            dict[str, torch.Tensor]: Single-key dict:

            * ``"logits"`` — shape ``[B, out_channels]`` (one logit vector per
              sample, all spatial information collapsed by global average pooling).
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
