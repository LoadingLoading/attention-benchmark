import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

def plot_result(results_dict, save_path="plots"):
    """
    Plot metrics for different attention modules.
      - Bar charts: FLOPS, GPU memory, model size, CPU frequency, CPU percent, CPU power.
      - Time-series: GPU utilization, CPU usage, CPU percent, CPU power,
                     inference time, training time, loss, disk I/O, etc.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Group DataFrames by (metric_name, unit)
    grouped_results = {}
    for (model_name, metric_name, unit), df in results_dict.items():
        grouped_results.setdefault((metric_name, unit), {})[model_name] = df

    # Metrics to plot as bar charts (one value per module)
    bar_chart_metrics = [
        "FLOPS",
        "gpu_memory",
        "model_size",
        "cpu_frequency",
        "cpu_percent",
        "cpu_power"
    ]

    for (metric_name, unit), models_data in grouped_results.items():
        # Bar-chart metrics
        if metric_name in bar_chart_metrics:
            labels = []
            values = []
            for model_name, df in models_data.items():
                labels.append("\n".join(model_name.split()))
                if metric_name == "FLOPS":
                    values.append(df["FLOPS"].mean())
                elif metric_name == "model_size":
                    values.append(df["Model size"].iloc[0])
                elif metric_name == "cpu_frequency":
                    values.append(df["CPU_Frequency"].mean())
                elif metric_name == "cpu_percent":
                    values.append(df["CPU_Percent"].mean())
                elif metric_name == "cpu_power":
                    values.append(df["CPU_Power"].mean())
                else:
                    # gpu_memory or other metrics stored in second column
                    values.append(df.iloc[:, 1].mean())

            plt.figure(figsize=(12, 8))
            plt.bar(labels, values)
            plt.title(f"Comparison of {metric_name}", fontsize=16)
            plt.xlabel("Attention Module", fontsize=14)
            plt.ylabel(f"{metric_name} ({unit})", fontsize=14)
            plt.xticks(rotation=60, ha="right")
            plt.grid(axis="y")
            filepath = os.path.join(save_path, f"{metric_name}_{unit}_bar.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()

        # Time-series metrics per epoch
        else:
            plt.figure(figsize=(12, 8))
            for model_name, df in models_data.items():
                x = df["Epoch"]
                if metric_name == "cpu_usage":
                    y = df["CPU_Time"] if "CPU_Time" in df.columns else df.iloc[:, 1]
                elif metric_name == "cpu_percent":
                    y = df["CPU_Percent"]
                elif metric_name == "cpu_power":
                    y = df["CPU_Power"]
                else:
                    # default to second column
                    y = df.iloc[:, 1]

                plt.plot(x, y, marker='o', linestyle='-', label=model_name)

            plt.title(f"Comparison of {metric_name}", fontsize=16)
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel(f"{metric_name} ({unit})", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)
            filepath = os.path.join(save_path, f"{metric_name}_{unit}.png")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()
