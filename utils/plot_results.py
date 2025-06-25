import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pickle
import glob
from collections import defaultdict
import pandas as pd

extra_font = 40  # Font size adjustment
line_minus=0
ylable_text="Att Mechanism"
ylable_text=''
label_map = {
    "ScaledDotProductAttention": "Scaled Dot Product Att",
    "BaselineAttention": "Baseline Att",
    # "MultiHeadFlexAttention": "Multi-Head Flex Att",  # Hidden
    "LinearFlexAttention": "Linear Att",
    "LSHAttention": "LSH Att",
    "SlidingWindowAttention": "Sliding Window Att",
    "GroupQueryAttention": "Group Query Att",
    "FlashAttention": "Flash Att",
    "MultiHeadLatentAttention": "Multi-Head Latent Att",
    "cpu_percent": "CPU Percentage",
    "cpu_usage": "CPU Usage",
    "cpu_power": "CPU Power",
    "memory_usage": "Memory Usage",
    "FLOPS": "FLOPS",
    "gpu_utilization_percentage": "GPU Utilization",
    "gpu_memory": "GPU Memory",
    "gpu_power": "GPU Power",
    "disk_io_read": "Disk IO Read",
    "disk_io_write": "Disk IO Write",
    "model_size": "Model Size",
    "inference_time": "Inference Time",
    "training_time": "Training Time",
    "loss": "Loss"
}

def merge_results(results_list):
    merged = defaultdict(lambda: defaultdict(list))
    for result in results_list:
        for key, df in result.items():
            merged[key]['dataframes'].append(df)

    merged_dict = {}

    for key, val in merged.items():
        dfs = val['dataframes']
        if len(dfs) == 1:
            merged_dict[key] = dfs[0]
        else:
            common_index = dfs[0].iloc[:, 0]
            values_stack = np.stack([df.iloc[:, 1].values for df in dfs])
            mean = np.mean(values_stack, axis=0)
            std = np.std(values_stack, axis=0)

            merged_df = pd.DataFrame({
                dfs[0].columns[0]: common_index,
                'mean': mean,
                'std': std
            })
            merged_dict[key] = merged_df

    return merged_dict

def plot_result(results_dict, save_path="plots"):
    import matplotlib.ticker as ticker

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    keep_line_plot = {"loss_loss", "gpu_utilization_percentage_percent"}

    plt.rcParams["font.family"] = "Arial"

    grouped_results = {}
    for key, value in results_dict.items():
        model_name = key[0]
        metric_name = key[1]
        unit = key[2]
        if (metric_name, unit) not in grouped_results:
            grouped_results[(metric_name, unit)] = {}
        grouped_results[(metric_name, unit)][model_name] = value

    for (metric_name, unit), models_data in grouped_results.items():
        metric_unit_key = f"{metric_name}_{unit}"
        if metric_unit_key in keep_line_plot:
            plt.figure(figsize=(12, 8))
            # Show all borders to form box with thick lines
            plt.gca().spines['top'].set_visible(True)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['top'].set_linewidth(3)
            plt.gca().spines['right'].set_linewidth(3)
            plt.gca().spines['bottom'].set_linewidth(3)
            plt.gca().spines['left'].set_linewidth(3)

            for model_name, data in models_data.items():
                # Skip MultiHeadFlexAttention but keep LinearFlexAttention
                if model_name == "MultiHeadFlexAttention":
                    continue

                x = data.iloc[:, 0]
                y = data.iloc[:, 1]
                # Increase line thickness
                plt.plot(x, y, marker='o', linestyle='-', linewidth=4, markersize=6,
                         label=label_map.get(model_name, model_name))

            pretty_metric = label_map.get(metric_name, metric_name)
            plt.text(
                0.5, 1.08,  # x=0.5 center, y>1 above the plot
                f"{pretty_metric} ({unit})",
                fontsize=-1 + extra_font//2.5 * 2,  # Reduce by 5 more: from 4 to -1
                ha='center',
                va='bottom',
                transform=plt.gca().transAxes
            )

            plt.xlabel("Epoch", fontsize=-1 + extra_font//2.5 * 2)  # Reduce by 5 more: from 4 to -1
            plt.xlim(1, 20)
            plt.xticks([1, 7, 14, 20], fontsize=-3 + extra_font//2.5 * 2)  # Reduce by 5 more: from 2 to -3
            plt.yticks(fontsize=-3 + extra_font//2.5 * 2)  # Reduce by 5 more: from 2 to -3
            plt.grid(False)

            plt.legend(
                fontsize=-3 + extra_font//2.5 * 2,  # Reduce by 5 more: from 2 to -3
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=1,
                frameon=False
            )

            filename = f"{metric_name.replace(' ', '_')}_{unit}.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
        else:
            attention_names = []
            values = []
            errors = []

            for model_name, data in models_data.items():
                # Skip MultiHeadFlexAttention but keep LinearFlexAttention
                if model_name == "MultiHeadFlexAttention":
                    continue

                name = label_map.get(model_name, model_name)
                attention_names.append(name)

                if metric_name == "FLOPS":
                    avg_value = data["mean"].mean() / 1e12
                    std_value = data["std"].mean() / 1e12
                elif metric_name == "model_size":
                    avg_value = data["mean"].values[0]
                    std_value = data["std"].values[0]
                elif metric_name == "gpu_memory":
                    avg_value = data["mean"].mean() / 1024  # Convert to GB
                    std_value = data["std"].mean() / 1024 if 'std' in data.columns else 0
                else:
                    avg_value = data["mean"].mean()
                    std_value = data["std"].mean() if 'std' in data.columns else 0

                values.append(avg_value)
                errors.append(std_value)

            # Sort by value (ascending), maintaining error bar correspondence
            sorted_indices = np.argsort(values)
            sorted_names = [attention_names[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            sorted_errors = [errors[i] for i in sorted_indices]

            xmax = max([v + e for v, e in zip(sorted_values, sorted_errors)]) * 1.25  # Increase from 1.15 to 1.25

            # Dynamically adjust chart height based on number of bars
            fig_height = max(8, len(sorted_names) * 1.7)  # 1.7 inches per bar
            plt.figure(figsize=(12, fig_height))

            # Show all borders to form box with thick lines
            plt.gca().spines['top'].set_visible(True)
            plt.gca().spines['right'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['top'].set_linewidth(3)
            plt.gca().spines['right'].set_linewidth(3)
            plt.gca().spines['bottom'].set_linewidth(3)
            plt.gca().spines['left'].set_linewidth(3)

            # Increase bar spacing by manually setting y-axis positions
            y_positions = np.arange(len(sorted_names)) * 2.5  # 2.5x spacing
            bars = plt.barh(y_positions, sorted_values, color="skyblue", height=1.5)  # Increase bar thickness

            # Add thicker error bars separately
            for i, (name, val, err) in enumerate(zip(sorted_names, sorted_values, sorted_errors)):
                plt.errorbar(val, y_positions[i], xerr=err, fmt='none', capsize=8,
                             elinewidth=4, capthick=4, color='black')  # Increase error bar thickness

            plt.yticks(y_positions, sorted_names, fontsize=14 + extra_font)
            plt.xlim(0, xmax)

            # Add value labels to the right of bars
            for i, (bar, val, err) in enumerate(zip(bars, sorted_values, sorted_errors)):
                if metric_name == "FLOPS":
                    label = f"{val:.3g}"
                elif val >= 1e6:
                    label = f"{val / 1e6:.3g}M"
                elif val >= 1e3:
                    label = f"{val / 1e3:.3g}K"
                else:
                    label = f"{val:.3g}"
                # Place label outside error bar to avoid overlap
                label_x_position = bar.get_width() + err + (xmax * 0.01)
                plt.text(label_x_position, y_positions[i], label,
                         ha='left', va='center', fontsize=14 + extra_font)

            pretty_metric = label_map.get(metric_name, metric_name)

            plt.ylabel(ylable_text, fontsize=16 + extra_font)
            if metric_name == "FLOPS":
                plt.xlabel(f"{pretty_metric} (×10¹²)", fontsize=16 + extra_font)
            elif metric_name == "gpu_memory":
                plt.xlabel(f"{pretty_metric} (GB)", fontsize=16 + extra_font)
            else:
                plt.xlabel(f"{pretty_metric} ({unit})", fontsize=16 + extra_font)

            # Y-axis labels already set above
            # Reduce x-axis label count to avoid overlap
            ax = plt.gca()
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))  # Maximum 4 labels
            plt.xticks(fontsize=14 + extra_font)  # Ensure x-axis font size matches y-axis
            plt.grid(False)

            filename = f"{metric_name.replace(' ', '_')}_{unit}.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()

def plot_energy_product(results_dict, save_path="plots"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.rcParams["font.family"] = "Arial"

    power_data = {}
    time_data = {}

    for (model_name, metric, unit), df in results_dict.items():
        if metric == "gpu_power":
            power_data[model_name] = df
        elif metric == "training_time":
            time_data[model_name] = df

    model_names = []
    total_energies_kj = []
    total_energy_errors = []

    for model in power_data:
        if model == "SparseFlexAttention":
            continue
        # Skip MultiHeadFlexAttention but keep LinearFlexAttention
        if model == "MultiHeadFlexAttention":
            continue
        if model in time_data:
            df_power = power_data[model]
            df_time = time_data[model]
            if len(df_power) == len(df_time):
                # Calculate energy consumption mean and standard deviation
                if 'mean' in df_power.columns and 'std' in df_power.columns:
                    power_mean = df_power["mean"].values
                    power_std = df_power["std"].values
                    time_mean = df_time["mean"].values
                    time_std = df_time["std"].values
                else:
                    # If no mean/std columns, assume second column is value
                    power_mean = df_power.iloc[:, 1].values
                    power_std = np.zeros_like(power_mean)
                    time_mean = df_time.iloc[:, 1].values
                    time_std = np.zeros_like(time_mean)

                # Energy = Power × Time
                energy_mean = power_mean * time_mean
                total_energy_joules = np.sum(energy_mean)
                total_energy_mj = total_energy_joules / 1e6

                # Error propagation: for product z = x*y, σz = sqrt((y*σx)² + (x*σy)²)
                energy_std_per_epoch = np.sqrt((time_mean * power_std)**2 + (power_mean * time_std)**2)
                # Total error: assuming independent epoch errors, total variance = sum of epoch variances
                total_energy_std_joules = np.sqrt(np.sum(energy_std_per_epoch**2))
                total_energy_std_mj = total_energy_std_joules / 1e6

                model_names.append(model)
                total_energies_kj.append(total_energy_mj)
                total_energy_errors.append(total_energy_std_mj)

    # Sort by energy consumption (ascending)
    sorted_indices = np.argsort(total_energies_kj)
    sorted_models = [label_map.get(model_names[i], model_names[i]) for i in sorted_indices]
    sorted_energies = [total_energies_kj[i] for i in sorted_indices]
    sorted_errors = [total_energy_errors[i] for i in sorted_indices]
    xmax = max([e + err for e, err in zip(sorted_energies, sorted_errors)]) * 1.25

    # Dynamically adjust chart height based on number of models
    fig_height = max(8, len(sorted_models) * 1.7)  # 1.7 inches per bar
    plt.figure(figsize=(12, fig_height))
    # Show all borders to form box with thick lines
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['top'].set_linewidth(3)
    plt.gca().spines['right'].set_linewidth(3)
    plt.gca().spines['bottom'].set_linewidth(3)
    plt.gca().spines['left'].set_linewidth(3)

    # Increase bar spacing by manually setting y-axis positions
    y_positions = np.arange(len(sorted_models)) * 2.5  # 2.5x spacing
    bars = plt.barh(y_positions, sorted_energies, color="skyblue", height=1.5)  # Increase bar thickness

    # Add thicker error bars separately
    for i, (model, energy, err) in enumerate(zip(sorted_models, sorted_energies, sorted_errors)):
        plt.errorbar(energy, y_positions[i], xerr=err, fmt='none', capsize=8,
                     elinewidth=4, capthick=4, color='black')  # Increase error bar thickness

    plt.xlim(0, xmax)

    plt.yticks(y_positions, sorted_models, fontsize=14 + extra_font)

    # Add value labels to the right of bars
    for i, (bar, val, err) in enumerate(zip(bars, sorted_energies, sorted_errors)):
        label = f"{val:.3g}"
        # Place label outside error bar to avoid overlap
        label_x_position = bar.get_width() + err + (xmax * 0.01)
        plt.text(label_x_position, y_positions[i], label,
                 ha='left', va='center', fontsize=14 + extra_font)

    plt.ylabel("", fontsize=16 + extra_font)
    plt.xlabel("Total Energy (MJ)", fontsize=16 + extra_font)
    # Reduce x-axis label count to avoid overlap
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))  # Maximum 4 labels
    plt.xticks(fontsize=14 + extra_font)
    plt.grid(False)

    filename = "gpu_total_energy_sorted.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved energy chart to {filepath}")

def main():
    all_results = []
    for pkl_file in glob.glob("pkl/*.pkl"):
        with open(pkl_file, "rb") as f:
            results_dict = pickle.load(f)
            all_results.append(results_dict)

    merged_results = merge_results(all_results)

    plot_result(merged_results)
    plot_energy_product(merged_results)

if __name__ == "__main__":
    main()
