# Grand Plan Overview Tracker

This document tracks our experimental roadmap across different datasets and tasks.

Status Legend:

- 📝 **Todo** (Not started)
- ⏳ **WIP** (In Progress)
- 🚧 **Blocked** (Needs help/discussion)
- ✅ **Done**

## The Well

**Tracker Document:** [examples/well/v2/TRACKER.md](examples/well/v2/TRACKER.md)

| Status  | Priority | Assignee     | Task                                                                              | Links / Configs / Notes |
| :------ | :------- | :----------- | :-------------------------------------------------------------------------------- | :---------------------- |
| 📝 Todo | P1       | @Olivia      | Replicate UNext on all sub-datasets using default learning rates from paper.      |                         |
| 📝 Todo | P1       | @Olivia      | Perform ablations for different patch sizes for the same epochs as the baselines. | default lr              |
| 📝 Todo | P2       | @Olivia      | Tune the learning rate and other parameters for the best-performing patch sizes.  |                         |
| 📝 Todo | P1       | @david&david | Implement Hyena UNext and perform ablations on a few selected sub-datasets.       |                         |
| 📝 Todo | P2       | @Olivia      | Tune learning rate and other hyperparameters for Hyena UNext.                     |                         |

## ImageNet (Classification)

**Tracker Document:** [examples/vit5_imagenet/TRACKER.md](examples/vit5_imagenet/TRACKER.md)

| Status  | Priority | Assignee     | Task                                                                               | Links / Configs / Notes |
| :------ | :------- | :----------- | :--------------------------------------------------------------------------------- | :---------------------- |
| 📝 Todo | P1       | @Alireza     | Run the patch-experiment defined in `examples/vit5_imagenet`.                      |                         |
| 📝 Todo | P2       | @david&david | Investigate whether Hyena with masking/gap can match the performance of Attention. |                         |
| 📝 Todo | P3       | @Unassigned  | Run best setups for different model-sizes                                          |                         |

## ADE21k (Segmentation)

**Tracker Document:** *To be defined*

| Status  | Priority | Assignee     | Task                                                                               | Links / Configs / Notes                   |
| :------ | :------- | :----------- | :--------------------------------------------------------------------------------- | :---------------------------------------- |
| 📝 Todo | P0       | @david&david | Define a tracker.md file containing all in-depth experiments.                      |                                           |
| 📝 Todo | P0       | @Unassigned  | Implement an efficient dataloader for ADE21k.                                      |                                           |
| 📝 Todo | P0       | @Unassigned  | Implement the ViT5 baseline and reproduce the result from the paper.               | [Paper](https://arxiv.org/pdf/2602.08071) |
| 📝 Todo | P1       | @Unassigned  | Evaluate if the baseline can be beaten using (masked) Hyena.                       |                                           |
| 📝 Todo | P2       | @Unassigned  | Conduct a patch size analysis.                                                     |                                           |
| 📝 Todo | P2       | @Unassigned  | Evaluate if Hyena UNext can achieve (i) better performance, and (ii) faster speed. |                                           |

## ImageNet (Diffusion) (Note: This needs some extra planning)

**Tracker Document:** *To be defined*

| Status  | Priority | Assignee     | Task                                                                               | Links / Configs / Notes |
| :------ | :------- | :----------- | :--------------------------------------------------------------------------------- | :---------------------- |
| 📝 Todo | P0       | @david&david | Define a tracker.md file containing all in-depth experiments.                      |                         |
| 📝 Todo | P0       | @Knigge?     | Obtain baseline vs standard Hyena values (e.g., 8 DiT vs 40 Hyena FID).            |                         |
| 📝 Todo | P1       | @Unassigned  | Run the masking experiment for Hyena.                                              |                         |
| 📝 Todo | P2       | @Unassigned  | Perform hyperparameter ablations (lr, omega0, weight decay, drop path rate).       |                         |
| 📝 Todo | P2       | @Unassigned  | Conduct a patch size analysis.                                                     |                         |
| 📝 Todo | P2       | @Unassigned  | Evaluate if Hyena UNext can achieve (i) better performance, and (ii) faster speed. |                         |
