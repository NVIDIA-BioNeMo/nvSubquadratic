# Review: `nvsubquadratic/modules/residual_block.py`

Reviewed as a second reader looking for gaps that would confuse an external collaborator
reading alongside the research paper.

______________________________________________________________________

## Issues

### 1. Module docstring does not clarify that `ResidualBlock` and `AdaLNZeroResidualBlock` are *alternative* block types, not composed together

**Location**: module docstring, opening paragraph.

**Quote**: "Two block variants are provided: ResidualBlock … AdaLNZeroResidualBlock …"

**Fix**: Add one sentence explicitly stating that a network uses one *or* the other per
stage/config — they are not nested. New collaborators may assume `AdaLNZeroResidualBlock`
somehow wraps `ResidualBlock`.

______________________________________________________________________

### 2. `ResidualBlock.__init__` docstring: `condition_mixer_cfg` description does not name a concrete supported type

**Location**: `ResidualBlock.__init__` Args, `condition_mixer_cfg`.

**Quote**: "When active, the module's `forward` must accept `(x, condition)` positional
arguments."

**Fix**: Name the concrete supported type — e.g. `ConditionMixer` from
`nvsubquadratic.modules.condition_mixer` — or note that as of this writing no concrete
condition mixer class exists in the public API (so `torch.nn.Identity` is effectively the
only safe value). Without this, a reader does not know what to pass.

______________________________________________________________________

### 3. `ResidualBlock.forward`: the `condition` parameter should be typed `Optional[torch.Tensor]`

**Location**: `forward` signature, line `def forward(self, x: torch.Tensor, condition: torch.Tensor)`.

**Fix**: Change the type annotation to `condition: torch.Tensor | None` (or
`Optional[torch.Tensor]`) to match actual runtime semantics: `condition` is
silently ignored (and therefore safely `None`) when `condition_mixer` is
`torch.nn.Identity`. The current signature implies it is always required, which
contradicts the docstring prose.

______________________________________________________________________

### 4. `ResidualBlock.forward` docstring: tensor shape for `condition` is too vague

**Location**: `ResidualBlock.forward` Args, `condition`.

**Quote**: "Its shape depends on the conditioning operator — a common choice is
`(B, *spatial_dims_condition, C)` for cross-attention, or `(B, C)` for a global
conditioning vector."

**Fix**: This is reasonable, but add a note that `condition` must have the *same* `C`
(hidden channel dimension) as `x`, since the condition mixer is expected to project into
the residual stream width. Otherwise readers may pass a conditioning tensor with a
different channel count and get a confusing runtime error.

______________________________________________________________________

### 5. `ResidualBlock.__init__`: the Raises section is missing a description for the case where norm/mixer Identity constraint is violated

**Location**: `ResidualBlock.__init__` Raises, currently reads "AssertionError: If a norm
config does not match…"

**Fix**: The assertion message says "Sequence mixer norm must be Identity if sequence mixer
is Identity" — but the *converse* (norm is Identity while mixer is not) is silently
allowed and would cause the norm to become a no-op. Clarify the *exact* direction of the
constraint enforced: mixer is Identity ⟹ norm must also be Identity. The reverse is not
checked and is the user's responsibility.

______________________________________________________________________

### 6. `AdaLNZeroResidualBlock.__init__`: `hidden_dim` relationship to sub-module configs is not explained

**Location**: `AdaLNZeroResidualBlock.__init__` Args, `hidden_dim`.

**Quote**: "Channel dimension `C` shared by all sub-modules. Used to size the
`condition_proj` linear layer (`Linear(C, 6*C)`)."

**Fix**: Note that `hidden_dim` must match the `dim`/`hidden_dim` argument baked into the
instantiated `sequence_mixer_cfg`, `mlp_cfg`, etc. There is no runtime check; a mismatch
will produce a shape error deep inside `condition_proj`, which is hard to trace back to a
`hidden_dim` mismatch at block init. Suggest either adding an assertion or at minimum
documenting that they must agree.

______________________________________________________________________

### 7. `AdaLNZeroResidualBlock.forward`: the `expand` inner function is undocumented as a local helper

**Location**: `AdaLNZeroResidualBlock.forward`, `def expand(param, ref)`.

**Current docstring**: "Broadcast a \[B, hidden_dim\] vector across ref's spatial axes."
(one line, present).

**Fix**: The one-liner is fine, but add a note that this helper is defined *inside*
`forward` to avoid materialising a persistent buffer. It is a purely local broadcast
utility, not a module method. This avoids readers wondering why it is not a `staticmethod`
or class method.

______________________________________________________________________

### 8. `AdaLNZeroResidualBlock.forward` docstring: does not mention that `conditioning=cond` is passed *into* the sequence mixer

**Location**: `AdaLNZeroResidualBlock.forward`, Returns/prose section.

**Fix**: The forward pass calls `self.sequence_mixer(seq_mod, conditioning=cond)` — the
pooled conditioning vector is forwarded into the inner mixer (e.g. Hyena with FiLM). This
is a non-obvious secondary conditioning path beyond the AdaLN modulation itself. Document
it explicitly in the forward docstring so readers understand there are two conditioning
signals: AdaLN-Zero affine modulation (shift/scale/gate) *and* the raw `cond` vector
passed to the inner mixer.

______________________________________________________________________

### 9. Both classes: no cross-reference to the network classes that instantiate them

**Location**: class-level docstrings.

**Fix**: Add a `See Also` or `Note` pointing to
`nvsubquadratic.networks.general_purpose_resnet` and/or
`nvsubquadratic.networks.classification_resnet` as the canonical consumers of these
blocks. External collaborators reading this file in isolation will not know where to look
to understand how blocks are stacked.

______________________________________________________________________

### 10. `AdaLNZeroResidualBlock`: class docstring does not state that the condition branch (cross-attention) of `ResidualBlock` is *absent*

**Location**: `AdaLNZeroResidualBlock` class docstring.

**Fix**: Add a sentence noting that `AdaLNZeroResidualBlock` has no separate
`condition_mixer` cross-attention branch (unlike `ResidualBlock`). All conditioning is
handled through the single AdaLN-Zero projection. Without this, readers familiar with
`ResidualBlock` will wonder what happened to the condition-mixer slot.
