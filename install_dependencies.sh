#!/bin/bash


# Install basic dependencies
pip install ptflops
pip install fvcore
pip install datasets
pip install packaging
pip install ninja

# Install specific PyTorch and CUDA version
pip install torch=='2.4.1+cu121' torchvision=='0.19.1+cu121' torchaudio=='2.4.1+cu121' --index-url https://download.pytorch.org/whl/cu121

# Set MAX_JOBS and install FlashAttention
MAX_JOBS=6 pip install flash-attn --no-build-isolation

