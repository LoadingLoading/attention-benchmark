import logging

only_use_first_serveral_batch=True

number_first_serveral_batch=2

batch_size_number=2

# for test bug on specific attention, use True
only_use_single_attention=True

# flops will fail on macbook, so for macbook use False for now
use_flops=True

max_epoch = 2

fast_test=False

full_test=True

model_name_list=['gpt2-large','gpt2-medium','gpt2']

model_name=model_name_list[2]

if fast_test:
    # for short test use True, for real experimennt, use False
    only_use_first_serveral_batch=True

    number_first_serveral_batch=2

    batch_size_number=16

    # for test bug on specific attention, use True
    only_use_single_attention=False

    # flops will fail on macbook, so for macbook use False for now
    use_flops=True

    max_epoch = 2

if full_test:
    # for short test use True, for real experimennt, use False
    only_use_first_serveral_batch=True

    number_first_serveral_batch=400

    batch_size_number=16

    # for test bug on specific attention, use True
    only_use_single_attention=False

    # flops will fail on macbook, so for macbook use False for now
    use_flops=True

    max_epoch = 30

# Logging configuration
logging.basicConfig(level=logging.CRITICAL, force=True)
