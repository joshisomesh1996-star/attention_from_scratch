# Self-Attention & Multi-Head Attention — Mini README

This repository/notebook collection demonstrates the **step-by-step implementation of self-attention and multi-head attention** using PyTorch.  
The goal is to build intuition by starting from simplified attention and gradually moving to a full multi-head attention module with masking and dropout.

---

## Contents

- **simplified_self_attention.ipynb**  
  Implements basic self-attention using dot products and softmax to show how tokens attend to each other.

- **self_attention_with_trainable_weights.ipynb**  
  Extends self-attention by introducing learnable Query, Key, and Value projection matrices.

- **multi_head_attention_weight_splits.ipynb**  
  Explains how embeddings are split into multiple heads using `reshape` and `transpose`, and why these operations are necessary.

- **multi_head_attention.ipynb**  
  Full implementation of **Multi-Head Attention** with:
  - Scaled dot-product attention
  - Causal (future) masking
  - Dropout on attention weights
  - Head concatenation and output projection

---

## Key Concepts Covered

- Self-Attention vs Multi-Head Attention  
- Query, Key, Value (QKV) projections  
- Scaled dot-product attention  
- Causal masking for autoregressive models (GPT-style)  
- `reshape` and `transpose` for head-wise computation  
- Why attention is computed **per head**  
- Final head concatenation and output projection  

---

## Requirements

- Python 3.x  
- PyTorch  
- Jupyter Notebook / Google Colab  

---

## Learning Objective

By the end of these notebooks, you should be able to:
- Understand how attention works mathematically
- Implement self-attention from scratch
- Explain why `reshape` and `transpose` are required in multi-head attention
- Confidently read and write Transformer attention code

---

## Notes

These notebooks are **educational and explanatory**, designed for learning and clarity rather than optimized production use.
