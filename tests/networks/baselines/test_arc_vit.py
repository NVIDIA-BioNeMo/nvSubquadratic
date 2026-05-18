import torch

from nvsubquadratic.networks.baselines.arc_vit import ARCViT


def test_arc_vit_initialization():
    net = ARCViT(
        num_tasks=10, max_size=32, num_colors=12, embed_dim=128, depth=2, num_heads=4, mlp_dim=256, patch_size=2
    )
    assert net is not None
    assert net.embed_dim == 128


def test_arc_vit_forward():
    net = ARCViT(
        num_tasks=10, max_size=32, num_colors=12, embed_dim=128, depth=2, num_heads=4, mlp_dim=256, patch_size=2
    )
    net.eval()

    batch_size = 2
    # Create mock inputs
    input_tensor = torch.randint(0, 10, (batch_size, 32, 32))  # [B, H, W]
    task_id = torch.randint(0, 10, (batch_size,))  # [B]
    attention_mask = torch.ones((batch_size, 32, 32))  # [B, H, W]

    input_and_condition = {"input": input_tensor, "condition": {"task_id": task_id, "attention_mask": attention_mask}}

    with torch.no_grad():
        output = net(input_and_condition)

    assert "logits" in output
    logits = output["logits"]

    # Check shape: [B, num_colors, H, W]
    assert logits.shape == (batch_size, 12, 32, 32)


def test_arc_vit_forward_no_mask():
    net = ARCViT(
        num_tasks=10, max_size=32, num_colors=12, embed_dim=128, depth=2, num_heads=4, mlp_dim=256, patch_size=2
    )
    net.eval()

    batch_size = 2
    input_tensor = torch.randint(0, 10, (batch_size, 32, 32))
    task_id = torch.randint(0, 10, (batch_size,))

    input_and_condition = {
        "input": input_tensor,
        "condition": {
            "task_id": task_id,
        },
    }

    with torch.no_grad():
        output = net(input_and_condition)

    logits = output["logits"]
    assert logits.shape == (batch_size, 12, 32, 32)
