Assignment: Fine Tuning a Large Language Model for Text Classification
===============================================

AUTHOR: Shubh Patel (002822971)



# Table of Contents
1. [Project Overview](#Project-Overview)
2. [Dataset](#Dataset-Source-And-Overview)
3. [Key Tools & Libraries](#key-Tools-&-Libraries)
4. [Models](#Models)
5. [Fine-Tuning Configuration and Limitations](#Fine-Tuning-Configuration-and-Limitations)
6. [Evaluation Metric](#Evaluation-Metric)
7. [Results & Summary](#Results-&-Summary)

# Project Overview

This project evaluates the performance of four pre-trained language models—BERT-base (110M), GPT2-XL (1.5B), Qwen2.5 (1.5B), and LLaMA-3.2 (1B)—on multi-class emotion classification using subsets of the GoEmotions dataset. The goal is to understand how model performance varies with:

A) Number of emotion labels (2, 4, 8, 16)

B) Training set size (800, 1600, 2400 samples)

C) Model scale and architecture

A key motivation is to compare BERT, a smaller and more lightweight model, with larger LLMs to assess whether it can remain competitive in low- and mid-resource settings. This helps highlight trade-offs between model size, performance, and data efficiency—crucial for real-world, resource-constrained applications.

# Dataset

The base dataset is GoEmotions — a human-annotated corpus of 58k Reddit comments labeled with 27 emotion categories.

For this project, 12 balanced subsets were created. Each subset varies by:

-- Number of label classes: 2, 4, 8, or 16

-- Number of training samples per class: 800, 1600, or 2400

This design forms a systematic evaluation grid across varying classification complexities and dataset sizes.

# Key Tools & Libraries

**Hugging Face Transformers** – for model loading, tokenization, and training

**PEFT (LoRA)** – for parameter-efficient fine-tuning

**Hugging Face Datasets** – for loading and preprocessing the dataset

**PyTorch** – as the deep learning backend

**Evaluate** – for computing evaluation metrics such as accuracy

# Models

The following pre-trained LLMs were fine-tuned for each dataset configuration:

-- Qwen 2.5

-- LLaMA 3.2

-- GPT-2 XL

-- Gemma 3

All models were trained using supervised learning with appropriate classification heads. Training incorporated early stopping based on validation accuracy to avoid overfitting.

# Fine-Tuning Configuration and Limitations

The models were fine-tuned on emotion classification datasets with varying label complexities (2, 4, 8, and 16 labels) and data sizes (800, 1600, and 2400 samples) under the following configuration:

A) **Quantization:** 4-bit quantized models using the BitsAndBytes library were employed to significantly reduce memory consumption and accelerate training.

B) **Adapter Tuning:** Low-Rank Adaptation (LoRA) was used for parameter-efficient fine-tuning. Model-specific attention modules (e.g., q_proj, v_proj, c_attn, or query/value) were targeted based on model architecture.

C) **Training Regimen:**

-- Optimizer: adamw_bnb_8bit

-- Batch size: 4 per device

-- Learning rate: 1e-4 with cosine scheduler

-- Epochs: 5

-- Gradient accumulation: 1 step

-- Mixed precision: bf16 used (with fallback to fp16=False)

-- Early stopping: Patience of 10 evaluation steps

**Note:** Fine-tuning was conducted under limited compute resources, with a single GPU and memory-optimized configurations (e.g., 4-bit quantization and adapter-based tuning). These constraints limited both the model size (to ≤2B parameters) and the maximum batch size.

# Evaluation Metric

The primary evaluation metric is accuracy, computed on a held-out validation set for each configuration. It reflects the model’s ability to correctly classify text across varying label granularities.

# Results & Summary

The fine-tuning experiments reveal consistent and interpretable trends in model performance across different classification complexities and dataset sizes. These trends underscore the trade-offs between model size, label granularity, and data availability.

**A) Effect of Label Complexity**

In 2-class classification, all models perform well, with LLaMA-3.2 reaching 100% accuracy at 2400 samples. As the number of labels rises—especially from 8 to 16—accuracy drops significantly due to the increased difficulty of fine-grained emotion recognition. At 16 labels, all models fall below 60% accuracy, even with the largest dataset, underscoring the challenge of complex classification under limited supervision.

**B) Effect of Training Data Size**

Increasing training data size consistently boosts performance across all models. For instance, Qwen2.5 improves from 43.75% (800 samples) to 56.67% (2400 samples) on the 16-label task. The impact is most significant in high-label settings, where more data helps models refine decision boundaries.

**C) Model Comparison: Performance vs. Scale**

The comparison between BERT (110M parameters) and larger models like LLaMA-3.2 (1B), Qwen2.5 (1.5B), and GPT2-XL (1.5B) reveals key trends:

-- LLaMA-3.2 and Qwen2.5 deliver the best performance and robustness across both low and high complexity tasks. For example, LLaMA-3.2 achieves 93.13% accuracy in the 4-label, 1600-sample setup.

-- In the challenging 16-label, 800-sample task, Qwen2.5 outperforms LLaMA-3.2 (43.75% vs. 30.00%) and BERT (8.75%).

-- GPT2-XL performs well in simpler tasks but is less stable in complex, low-resource settings. BERT excels in low-label, data-rich tasks (99.17% in 2-label, 2400-sample) but struggles with high-label complexity, dropping to 8.75% in the 16-label, 800-sample case.

**D) Heatmap Visualization**

The heatmap below shows:

Performance decline with increasing label complexity.

Improvement with larger training datasets.

Strong performance and robustness of LLaMA-3.2 and Qwen2.5.

BERT's strengths in simple tasks and weaknesses in complex ones.

![Model_Performnace](Model_Performance.png)
