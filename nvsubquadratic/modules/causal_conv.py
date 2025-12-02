import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    """
    1D Causal Convolution.
    
    This layer pads the input on the left side so that the output at time t
    only depends on inputs up to time t.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # For causal convolution with kernel_size K and dilation D,
        # we need to pad (K-1) * D on the left.
        # We assume stride=1 for now as is typical in these models.
        # If stride > 1, the output length will be reduced.
        
        # We ignore the 'padding' argument passed to super() and handle it manually.
        # However, nn.Conv1d stores it. We should set it to 0 to avoid double padding.
        # But we can't easily change it after super().__init__ if it affects internal state?
        # Actually we can just set self.padding = 0 and use F.pad.
        
        # Calculate required left padding
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]
        
        # Disable standard padding
        self.padding = (0,)

    def forward(self, x):
        # x: [B, C, L]
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)
