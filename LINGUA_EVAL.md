# Evaluation with Lingua (lm-eval)

This repository includes an adapter to evaluate `Zyda` models using the [Lingua](https://github.com/facebookresearch/lingua) evaluation logic (powered by `lm-eval`).

## Setup

Ensure you have the necessary dependencies installed:

```bash
pip install lm-eval megatron-core
```

## Usage

Use the `scripts/evaluate_with_lingua.py` script to run evaluations.

```bash
python scripts/evaluate_with_lingua.py \
    --ckpt_path /path/to/checkpoint.ckpt \
    --config_path examples/text_pretraining/zyda_1d_attention.py \
    --tasks hellaswag,arc_easy \
    --batch_size 8 \
    --device cuda
```

### Arguments

- `--ckpt_path`: Path to the model checkpoint (`.ckpt` file).
- `--config_path`: Path to the experiment configuration file used to train the model.
- `--tasks`: Comma-separated list of tasks to evaluate (e.g., `hellaswag,arc_easy,piqa`).
- `--batch_size`: Batch size for evaluation (default: 1).
- `--device`: Device to run on (default: `cuda`).
- `--output_path`: Optional path to save results as JSON.

## Supported Tasks

The script supports all tasks available in `lm-eval`. Common benchmarks include:

- `hellaswag`
- `arc_easy`
- `arc_challenge`
- `piqa`
- `winogrande`
- `boolq`
- `sciq`

## Notes

- The script automatically handles loading the model configuration and instantiating the network.
- It uses `lm_eval.simple_evaluate` to run the evaluation.
- Ensure your environment has access to the `nvsubquadratic` package and other dependencies.
