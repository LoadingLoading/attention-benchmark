# Attention Profiling for Efficient Transformer Variants

This project evaluates and compares various attention mechanisms in Transformer models from both computational and energy-efficiency perspectives. It includes experiments on GPU energy usage, convergence behavior, and memory consumption using a consistent benchmarking setup.

# Introduction
This project aims to measure all the usage while training and compare it between using different attention. 

This project uses one dataset: allenai/tulu-v2-sft-mixture. Uses one model: 'gpt2'. Uses 8 different attentions plus one baseline: Scaled Dot-Product Attention, Multi-Head Flex Attention, Multi-head Latent Attention, Linear Flex Attention, LSH Attention, Sliding Window Attention, Flash Attention, Group Query Attention, and Baseline Attention. Measures 12 different numbers: FLOPS, CPU usage percentage, disk I/O read in MB, disk I/O write in MB, GPU memory in MB, GPU power in watts, GPU utilization percentage, inference time in seconds, loss, memory usage in GB, model size in MB, and training time in seconds.

# Quick Start
All experiments can run on Collab, you can upload everything on Collab, and open the file for_colab.ipynb. There are 5 cells. You could run everything in sequence. Then run the third one again, as you can see in the cell: print the previous result. You can change the preset in config.py file.

You can also run the experiments in Ubuntu. After setting up all the required environments, you may run the main.py to start.



# About config.py

This line decide it is mini-batch SGD. 
only_use_first_serveral_batch=True

How many batch used in total. Because it detect everything each batch, so this number will be better to be higher. When I get the result, I set to 5. But for test, 2 is enough.
number_first_serveral_batch=2

batch_size_number=2

In main.py, choose True will lead only run flash attention and normal one. Or will run all 8 attentions plus one normal.
only_use_single_attention=True

max_epoch = 2

This is preset to test all things but quickly.
fast_test=True

This is preset for real tests and has higher priority than fast_test.
full_test=True




