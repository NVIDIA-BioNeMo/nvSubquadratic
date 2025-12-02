# Text Pretraining Metrics Summary

This document explains the metrics logged during the text pretraining experiments and their intended purpose.

## 1. Loss (`train/loss`, `val/loss`)
- **Definition**: Cross-Entropy Loss between the model's predictions and the actual next tokens.
- **Purpose**: The primary objective function being minimized. Lower is better.
- **Insight**: Indicates how well the model fits the training data. Divergence (loss going up) or stagnation (loss not going down) are key signals of training issues.

## 2. Perplexity (`train/ppl`, `val/ppl`)
- **Definition**: The exponential of the loss ($e^{\text{loss}}$).
- **Purpose**: A more intuitive measure of "surprise". A perplexity of $X$ means the model is as confused as if it were randomly guessing among $X$ equally likely options.
- **Target**: For this model (~40M params, 131k vocab), a PPL of **30-40** is a good target.
- **Insight**: 
    - **< 30**: Excellent performance.
    - **30 - 45**: Good/Solid performance.
    - **> 100**: Something is likely wrong (or training just started).

## 3. Accuracy (`train/acc`, `val/acc`)
- **Definition**: Top-1 Accuracy. The percentage of times the model's *highest probability* prediction matches the exact correct next token.
- **Purpose**: Measures precise correctness.
- **Insight**: 
    - **~50%** is excellent for a large vocabulary (131k tokens).
    - High accuracy with high perplexity suggests the model is good at "easy" tokens (stopwords, punctuation) but uncertain about content words.

## 4. Top-5 Accuracy (`train/acc_top5`, `val/acc_top5`)
- **Definition**: The percentage of times the correct next token is within the model's top 5 highest probability predictions.
- **Purpose**: Measures "near-miss" correctness.
- **Insight**: 
    - If Top-1 Acc is low but Top-5 Acc is high (e.g., > 80%), the model understands the context well but struggles with specific word choice synonyms.
    - If both are low, the model is failing to understand the context.

## 5. Gradient Norm (`grad_norm`)
- **Definition**: The L2 norm (magnitude) of the gradients calculated during the backward pass.
- **Purpose**: Monitoring training stability.
- **Insight**:
    - **Spikes**: Indicate instability, "bad batches", or learning rate issues.
    - **Vanishing (near 0)**: The model has stopped learning (gradients are too small).
    - **Exploding (huge values)**: The model is diverging. Gradient clipping helps prevent this, but consistent high norms suggest hyperparameter issues.
