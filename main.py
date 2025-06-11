from numpy import False_  # (kept from your original)
import os, time, glob, gc, logging, pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import pandas as pd
import psutil

# ── Project modules ─────────────────────────────────────────────────────────
from data.dataset import TextDataset
from utils.resource_monitor import monitor_resource_usage
from utils.gpu_monitor import monitor_GPU_usage
from utils.flops_counter import measure_flops
from utils.plot_results import plot_result
from config import (
    only_use_single_attention, use_flops, only_use_first_serveral_batch,
    number_first_serveral_batch, batch_size_number, max_epoch,
    fast_test, full_test, model_name,
)
from attention_modules import (
    GroupQueryAttention, ScaledDotProductAttention, MultiHeadFlexAttention,
    LinearFlexAttention, LSHAttention, SlidingWindowAttention,
    FlashAttention, BaselineAttention, MultiHeadLatentAttention
)
from models.gpt2_custom import GPT2CustomAttentionModel

# ── Optional: Intel RAPL power ──────────────────────────────────────────────
def read_cpu_energy_uj():
    paths = (
        glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:?/energy_uj")
        + glob.glob("/sys/class/powercap/intel-rapl/energy_uj")
        + glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj")
    )
    for p in sorted(paths):
        try:
            with open(p) as f:
                return int(f.read().strip())
        except Exception:
            continue
    return None

# ── Training loop (now baseline-aware) ──────────────────────────────────────
def train_model(
    model_to_train, dataloader_instance, optimizer_instance,
    criterion_instance, device_instance, num_epochs_to_run,
    baseline_gb,                          # ← NEW PARAM
):
    cpu_percent_data   = []
    cpu_usage_data     = []
    cpu_power_data     = []
    memory_usage_data  = []
    flops_data         = []
    gpu_percent_data   = []
    gpu_memory_data    = []
    gpu_power_data     = []
    disk_io_read_data  = []
    disk_io_write_data = []
    inference_time_data= []
    training_time_data = []
    loss_data          = []

    for epoch in range(num_epochs_to_run):
        print(f"Epoch {epoch+1}/{num_epochs_to_run}")
        model_to_train.train()

        epoch_start     = time.time()
        cpu_energy_start= read_cpu_energy_uj()

        num_batch = 0
        totals = dict(mem=0.0, cpu_pct=0.0, cpu_time=0.0,
                      flops=0.0, gpu_pct=0.0,
                      gpu_mem=0.0, gpu_pwr=0.0)
        prev_cpu_time = None
        disk_read_start  = psutil.disk_io_counters().read_bytes
        disk_write_start = psutil.disk_io_counters().write_bytes
        total_inference_time = 0.0
        total_loss           = 0.0

        for batch in dataloader_instance:
            if only_use_first_serveral_batch and \
               num_batch == number_first_serveral_batch:
                break
            num_batch += 1

            inputs, masks = (t.to(device_instance) for t in batch)
            if inputs.shape[1] == 0:
                continue

            if torch.cuda.is_available():
                torch.cuda.synchronize(device_instance)

            optimizer_instance.zero_grad()
            inf_start = time.time()
            outputs = model_to_train(
                input_ids=inputs, attention_mask=masks, labels=inputs
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_instance)
            total_inference_time += time.time() - inf_start

            loss = outputs[0]
            loss.backward()
            optimizer_instance.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_instance)
            total_loss += loss.item()

            # ── Resource snapshots ─────────────────────────────────────
            mem_usage, *_cpu = monitor_resource_usage()
            cpu_percent, cpu_time_used = _cpu[1], _cpu[2]
            gpu_pct, gpu_mem, gpu_pwr = monitor_GPU_usage()

            # —— Baseline-shift host RAM so first point ≈ 0 GB ————
            mem_usage = max(mem_usage - baseline_gb, 0.0)

            totals["mem"]      += mem_usage
            totals["cpu_pct"]  += cpu_percent

            delta_cpu_time = cpu_time_used if prev_cpu_time is None \
                             else cpu_time_used - prev_cpu_time
            prev_cpu_time = cpu_time_used
            totals["cpu_time"] += delta_cpu_time

            if use_flops:
                totals["flops"] += measure_flops(model_to_train, inputs)

            totals["gpu_pct"] += gpu_pct
            totals["gpu_mem"] += gpu_mem
            totals["gpu_pwr"] += gpu_pwr

        # ── Epoch aggregates ──────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_instance)

        if num_batch:
            averages = {k: v/num_batch for k, v in totals.items()}
            loss_avg = total_loss/num_batch
        else:
            averages = {k: 0.0 for k in totals}
            loss_avg = 0.0

        disk_read_end  = psutil.disk_io_counters().read_bytes
        disk_write_end = psutil.disk_io_counters().write_bytes
        epoch_time     = time.time() - epoch_start

        cpu_energy_end = read_cpu_energy_uj()
        if (cpu_energy_start is not None and cpu_energy_end is not None and
            cpu_energy_end >= cpu_energy_start and epoch_time > 0):
            joules = (cpu_energy_end - cpu_energy_start) / 1e6
            avg_cpu_power = joules / epoch_time
        else:
            avg_cpu_power = 0.0

        # ── Log rows  ─────────────────────────────────────────────────
        cpu_percent_data.append(   {"Epoch": epoch+1, "CPU_Percent": averages["cpu_pct"]})
        cpu_usage_data.append(     {"Epoch": epoch+1, "CPU_Time":    averages["cpu_time"]})
        cpu_power_data.append(     {"Epoch": epoch+1, "CPU_Power":   avg_cpu_power})
        memory_usage_data.append(  {"Epoch": epoch+1, "Memory_Usage":averages["mem"]})
        flops_data.append(         {"Epoch": epoch+1, "FLOPS":       averages["flops"]})
        gpu_percent_data.append(   {"Epoch": epoch+1, "GPU_Utilization_Percentage": averages["gpu_pct"]})
        gpu_memory_data.append(    {"Epoch": epoch+1, "GPU_Memory":  averages["gpu_mem"]})
        gpu_power_data.append(     {"Epoch": epoch+1, "GPU_Power":   averages["gpu_pwr"]})
        disk_io_read_data.append(  {"Epoch": epoch+1, "Disk_IO_Read": disk_read_end - disk_read_start})
        disk_io_write_data.append( {"Epoch": epoch+1, "Disk_IO_Write":disk_write_end - disk_write_start})
        inference_time_data.append({"Epoch": epoch+1, "Inference_Time": total_inference_time})
        training_time_data.append( {"Epoch": epoch+1, "Training_Time":  epoch_time})
        loss_data.append(          {"Epoch": epoch+1, "Loss":           loss_avg})

    # ── Convert lists → DataFrames and return ────────────────────────
    return tuple(
        pd.DataFrame(lst) for lst in (
            cpu_percent_data, cpu_usage_data, cpu_power_data, memory_usage_data,
            flops_data, gpu_percent_data, gpu_memory_data, gpu_power_data,
            disk_io_read_data, disk_io_write_data,
            inference_time_data, training_time_data, loss_data
        )
    )

# ── Setup  ──────────────────────────────────────────────────────────────────
overall_start = time.perf_counter()

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
gpt2_config = GPT2LMHeadModel.from_pretrained(model_name).config

dataset = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
text_dataset = TextDataset(dataset, tokenizer)
dataloader = DataLoader(text_dataset, batch_size=batch_size_number, shuffle=True)

# total_examples = len(text_dataset)               
# total_batches = len(dataloader)                  
# print(f"Total training examples: {total_examples}")  
# print(f"Batch size:               {batch_size_number}")  
# print(f"Total batches per epoch:   {total_batches}\n")  

attention_module_classes = [
    ScaledDotProductAttention, BaselineAttention, MultiHeadFlexAttention,
    LinearFlexAttention, LSHAttention, SlidingWindowAttention,
    GroupQueryAttention, FlashAttention, MultiHeadLatentAttention
]
if only_use_single_attention:
    attention_module_classes = [BaselineAttention]

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
results_dict = {}

# ── Outer loop over attention variants ─────────────────────────────
for attn_cls in attention_module_classes:
    # -- RAM baseline before model allocates ─────────────────────────
    baseline_gb = psutil.virtual_memory().used / (1024**3)

    print(f"\n──────────────── Training with {attn_cls.__name__} ────────────────")
    current_model = GPT2CustomAttentionModel(gpt2_config, attn_cls).to(device)
    optimizer = optim.AdamW(current_model.parameters(), lr=5e-5)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    dfs = train_model(
        current_model, dataloader, optimizer, criterion,
        device, max_epoch, baseline_gb      # ← baseline passed in
    )
    (
        cpu_percent_df, cpu_usage_df, cpu_power_df, memory_usage_df, flops_df,
        gpu_percent_df, gpu_memory_df, gpu_power_df, disk_io_read_df,
        disk_io_write_df, inference_time_df, training_time_df, loss_df
    ) = dfs

    # ── Model size record  ───────────────────────────────────────────
    tmp_path = f"{attn_cls.__name__}.pth"
    torch.save(current_model.state_dict(), tmp_path)
    model_size_mb = os.path.getsize(tmp_path) / (1024**2)
    os.remove(tmp_path)
    model_size_df = pd.DataFrame({"Epoch": ["n/a"], "Model size": [model_size_mb]})

    # ── Collect results ──────────────────────────────────────────────
    key = (attn_cls.__name__,)
    results_dict[key + ("cpu_percent", "percent")] = cpu_percent_df
    results_dict[key + ("cpu_usage", "seconds")]   = cpu_usage_df
    results_dict[key + ("cpu_power", "W")]         = cpu_power_df
    results_dict[key + ("memory_usage", "GB")]     = memory_usage_df
    results_dict[key + ("FLOPS", "FLOPS")]         = flops_df
    results_dict[key + ("gpu_utilization_percentage", "percent")] = gpu_percent_df
    results_dict[key + ("gpu_memory", "MB")]       = gpu_memory_df
    results_dict[key + ("gpu_power", "W")]         = gpu_power_df
    results_dict[key + ("disk_io_read", "MB")]     = disk_io_read_df
    results_dict[key + ("disk_io_write", "MB")]    = disk_io_write_df
    results_dict[key + ("model_size", "MB")]       = model_size_df
    results_dict[key + ("inference_time", "s")]    = inference_time_df
    results_dict[key + ("training_time", "s")]     = training_time_df
    results_dict[key + ("loss", "loss")]           = loss_df

    # ── Clean-up before next module ──────────────────────────────────
    del current_model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

# ── Plot + serialize results ────────────────────────────────────────
test_type = "full_test" if full_test else "fast_test" if fast_test else "normal_test"
run_stamp = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
base_name = (
    f"{run_stamp},{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'},"
    f"{test_type}_final_results_{model_name}_batchNumber{number_first_serveral_batch}"
    f"_batchSize{batch_size_number}"
)
results_dir = os.path.join("results", base_name)
os.makedirs(results_dir, exist_ok=True)

plot_result(results_dict, results_dir)

with open(
    os.path.join(
        results_dir,
        f"final_results_{model_name}_batchNumber{number_first_serveral_batch}"
        f"_batchSize{batch_size_number}.pkl",
    ),
    "wb",
) as f:
    pickle.dump(results_dict, f)

# ── Console summary ────────────────────────────────────────────────
print("\nConfiguration:")
print(f"only_use_first_serveral_batch = {only_use_first_serveral_batch}")
print(f"number_first_serveral_batch    = {number_first_serveral_batch}")
print(f"batch_size_number              = {batch_size_number}")
print(f"only_use_single_attention      = {only_use_single_attention}")
print(f"use_flops                      = {use_flops}")
print(f"max_epoch                      = {max_epoch}")
print(f"fast_test                      = {fast_test}")
print(f"full_test                      = {full_test}")

elapsed = time.perf_counter() - overall_start
print(f"\nTotal run time: {elapsed:.2f} seconds")