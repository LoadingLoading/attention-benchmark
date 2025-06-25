# Attention Profiling: Resource Utilization of Self-Attention Variants

## Introduction

Attention mechanisms have emerged as critical components in large language models (LLMs), dramatically enhancing their capabilities but significantly increasing computational demands and energy consumption. Despite numerous proposals for more efficient attention mechanisms, there is a notable lack of comprehensive quantitative evaluations comparing their performance, resource utilization, and environmental impact. This research systematically benchmarks eight prominent self-attention variants within the GPT-2 architecture, rigorously measuring GPU memory usage, training duration, computational complexity (FLOPs), energy consumption, and model size. By providing detailed empirical comparisons, we aim to deliver practical insights into selecting attention mechanisms that are both computationally efficient and environmentally sustainable, aligning with the broader objectives of Green AI.

## Methods

### Attention Mechanisms Evaluated

- **Baseline Self-Attention** (GPT-2)   
- **Scaled Dot-Product Attention**
- **Grouped Query Attention**
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
    
### Dataset

- **Tulu-v2-sft-mixture** dataset from AllenAI (available on Hugging Face). 
- Structured multi-turn conversation data tokenized to a maximum length of 512 tokens.

## Results

FlashAttention, Linear Attention, and ScaledDotProductAttention demonstrated superior training efficiency, with FlashAttention notably achieving the lowest cumulative GPU energy consumption by effectively balancing training speed and power draw. Additionally, FlashAttention and Linear Attention exhibited the lowest power consumption, while FlashAttention and ScaledDotProductAttention required the least GPU memory.

<h2>Visualizations</h2>

<h3>Training Time per Epoch</h3>
<img src="https://github.com/Zhengyu-Tian/attention-benchmark/blob/main/assets/training_time_s.png" alt="Training Time per Epoch" width="600"/>
<p><em>Baseline and sliding-window attention are the two slowest to train—hovering around 260 s per epoch—while LSH, Linear Flex, and Scaled Dot Product finish roughly 20 % faster, making them the clear choices if wall-clock speed is the priority.</em></p>

<hr/>

<h3>GPU Energy Consumption</h3>
<img src="https://github.com/Zhengyu-Tian/attention-benchmark/blob/main/assets/gpu_total_energy_sorted.png" alt="GPU Energy Consumption" width="600"/>
<p><em>Energy-wise, Flash Attention is the most frugal at about 1.07 MJ, consuming roughly 25 % less GPU energy than the power-hungry Sliding-Window and Baseline methods, which top the chart at 1.43 MJ and 1.39 MJ respectively.</em></p>

<hr/>

<h3>GPU Memory Usage</h3>
<img src="https://github.com/Zhengyu-Tian/attention-benchmark/blob/main/assets/gpu_memory_MB.png" alt="GPU Memory Usage" width="600"/>
<p><em>On memory footprint, Flash Attention again leads with the smallest peak usage (~16.9 GB), while Multi-Head Flex and Sliding-Window push past 20 GB, signaling that Flash is the best fit for tight-VRAM settings.</em></p>

<hr/>

<h3>Training Loss Curves</h3>
<img src="https://github.com/Zhengyu-Tian/attention-benchmark/blob/main/assets/loss_loss.png" alt="Training Loss Curves" width="600"/>
<p><em>Across twenty epochs, Multi-Head Latent Attention is the only variant whose loss keeps falling steadily past epoch 10, ending below 2, whereas every other mechanism flattens out near 3 and Sliding Window even drifts upward—highlighting Latent Attention’s superior convergence.</em></p>

<hr/>

<h3>FLOPs and Memory Usage</h3>
<img src="https://github.com/Zhengyu-Tian/attention-benchmark/blob/main/assets/FLOPS_FLOPS.png" alt="FLOPs and Memory Usage" width="600"/>
<p><em>Finally, in raw compute cost, Multi-Head Latent Attention registers the lowest FLOPs (~0.92 × 10¹²), whereas Linear Flex edges past 1.07 × 10¹², meaning Latent Attention delivers its strong learning curve with the lightest arithmetic burden of the group.</em></p>


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





