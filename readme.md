# Attention Profiling: Resource Utilization of Self-Attention Variants

## Introduction

Attention mechanisms have become central to large language models (LLMs), significantly driving their performance but also introducing substantial computational and energy demands. This project provides a detailed benchmarking of various attention mechanisms, focusing on their training efficiency, GPU resource usage, and energy consumption.

## Project Description

Our study systematically evaluates nine self-attention mechanisms within the GPT-2 framework. Metrics including GPU memory usage, training time, FLOPs, power consumption, and model size are thoroughly compared to provide practical insights for selecting resource-efficient attention variants, aligning with the goals of Green AI.

## Key Contributions

- **Comprehensive Benchmarking**: Empirical comparison of nine popular self-attention mechanisms.
- **Resource Efficiency Analysis**: Detailed measurement of GPU memory usage, training time, energy consumption, and computational complexity.
- **Practical Recommendations**: Insights on the most energy-efficient attention mechanisms suitable for deployment in resource-constrained environments.
    

## Background and Motivation

Given the escalating complexity and size of modern LLMs, optimizing resource utilization has become critical. Despite various efficient attention proposals, there remains a gap in quantitative evaluations of their energy and resource demands. Our benchmarking addresses this gap, emphasizing the environmental and practical implications.

## Methods

### Attention Mechanisms Evaluated

- **Baseline Self-Attention** (GPT-2)   
- **Scaled Dot-Product Attention**
- **Grouped Query Attention**
- **Multi-Head Flexible Attention**
- **Linear Attention**
- **Sliding Window Attention**
- **Locality-Sensitive Hashing (LSH) Attention**
- **FlashAttention v2**
- **Multi-Head Latent Attention (MHLA)**

### Benchmarking Metrics

- **Training Time**
- **GPU Memory Usage**
- **FLOPs** (Floating Point Operations)
- **GPU Power Consumption**
- **Inference Latency**

### Hardware and Software

- **GPU**: NVIDIA RTX 4090, 24 GB VRAM    
- **CPU**: Intel Core i9, 3.30 GHz
- **RAM**: 256 GB
- **Operating System**: Ubuntu 20.04 LTS
- **CUDA Toolkit**: 12.0
    
### Dataset

- **Tulu-v2-sft-mixture** dataset from AllenAI (available on Hugging Face). 
- Structured multi-turn conversation data tokenized to a maximum length of 512 tokens.
    

## Results

### Training Efficiency

FlashAttention, Linear Attention, and ScaledDotProductAttention showed superior training speeds, contributing significantly to energy efficiency.

### GPU Resource Utilization

- **Lowest Power Consumption**: FlashAttention, Linear Attention
- **Lowest GPU Memory Usage**: FlashAttention, ScaledDotProductAttention
    

### Energy Consumption

FlashAttention demonstrated the lowest cumulative GPU energy consumption, balancing training speed and GPU power draw effectively.

## Conclusion

This benchmarking study underscores the importance of choosing attention mechanisms based on comprehensive efficiency metrics. Attention mechanisms such as FlashAttention, Linear Attention, and ScaledDotProductAttention emerge as leading choices for environmentally sustainable and resource-efficient AI.

## Future Work

Future work will focus on several important directions:

- **Scalability Analysis**: Evaluate the attention mechanisms on larger-scale open models to understand their performance in more complex scenarios.
- **Diverse Dataset Benchmarking**: Benchmark the attention mechanisms using more varied and multi-task datasets to assess their adaptability and performance across different workloads.
- **Hardware Variation Analysis**: Extend benchmarking to different GPU architectures to investigate how attention mechanisms perform across varying hardware environments.

## Getting Started

### Installation

```bash
git clone https://github.com/LoadingLoading/attention-benchmark

cd attention-benchmark

pip install -r requirements.txt
```

### Running Benchmark

```bash
python main.py
```

## Contributions

We invite researchers and developers to further explore these findings, extend benchmarking to additional attention mechanisms, and contribute to a greener AI future.

Please open an issue or submit a pull request for any queries, suggestions, or contributions.


