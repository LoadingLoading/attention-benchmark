# Quick Start
All experiments run on Collab, you can upload everything on Collab, and open the file for_colab.ipynb. There are 5 cells. Just run everything in sequence. Then run the third one again, as you can see in the cell: print the previous result. You can change the preset in config.py file.

# Introduction
This project aims to measure all the usage while training and compare them between using the different attentions. 

This project uses one dataset: allenai/tulu-v2-sft-mixture. Uses 3 different models: 'gpt2-large', 'gpt2-medium', and 'gpt2'. Uses 8 different attentions plus one baseline: Scaled Dot-Product Attention, Multi-Head Flex Attention, Sparse Flex Attention, Linear Flex Attention, LSH Attention, Sliding Window Attention, Flash Attention, Group Query Attention, and Baseline Attention. Measures 12 different numbers: FLOPS, CPU usage percentage, disk I/O read in MB, disk I/O write in MB, GPU memory in MB, GPU power in watts, GPU utilization percentage, inference time in seconds, loss, memory usage in GB, model size in MB, and training time in seconds.

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

model_name_list=['gpt2-large','gpt2-medium','gpt2']

Choose 0 will be gpt2-large, 1 will be gpt2-medium, 2 will be gpt2.
model_name=model_name_list[2]

Other parts are quite straightforward. I will have a short introduction for the rest.

# Difficulties

Different version has different problems. All version has the same problem: need to fit the attention to the model.

# Updates:
2024 Dec 6: 
Adjust the code structure again, making it easier to read and understand.
Move a lot of things to config, so people only need to change the config file.
Create the file and save the result conveniently. Delete every unnecessary file. 
Change some ways to show the data, making it clear.
