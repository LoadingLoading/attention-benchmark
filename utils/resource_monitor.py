import psutil

def monitor_resource_usage():
    # Get memory info
    memory_info = psutil.virtual_memory()
    used_memory_gb = memory_info.used / (1024 ** 3)
    total_memory_gb = memory_info.total / (1024 ** 3)
    
    # Get CPU percentage
    cpu_percent = psutil.cpu_percent(interval=0)
    
    # Get CPU times to calculate actual usage (cumulative since boot)
    cpu_times = psutil.cpu_times()
    cpu_time_used = sum(cpu_times) - cpu_times.idle
    
    # Get current CPU clock frequency (MHz)
    cpu_freq = psutil.cpu_freq()
    cpu_freq_current = cpu_freq.current if cpu_freq else 0
    
    # Number of logical cores
    cpu_count = psutil.cpu_count(logical=True)
    
    return used_memory_gb, total_memory_gb, cpu_percent, cpu_time_used, cpu_freq_current, cpu_count