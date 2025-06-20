import time
import tracemalloc
import psutil
import os
from contextlib import contextmanager

class ForwardPassCounter:
    def __init__(self):
        self.count = 0
    def increment(self, n=1):
        self.count += n
    def reset(self):
        self.count = 0
    def get(self):
        return self.count

@contextmanager
def measure_time():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

@contextmanager
def measure_memory():
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    try:
        yield lambda: max(tracemalloc.get_traced_memory()[1], process.memory_info().rss)
    finally:
        tracemalloc.stop()

def get_peak_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def profile_hardware():
    # Returns dict of cpu/gpu utilization (simple version)
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory().percent
    gpu_stats = get_gpu_usage()
    return {'cpu_percent': cpu, 'ram_percent': mem, **gpu_stats}

def get_gpu_usage():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'gpu_util_percent': util.gpu,
            'gpu_mem_used_MB': mem.used // 1048576,
            'gpu_mem_total_MB': mem.total // 1048576
        }
    except Exception:
        return {'gpu_util_percent': None, 'gpu_mem_used_MB': None, 'gpu_mem_total_MB': None}

def estimate_flops(model, tokenizer, text, device):
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity
        inputs = tokenizer(text, return_tensors='pt').to(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU], record_shapes=True) as prof:
            with torch.no_grad():
                model(**inputs)
        flops = None
        for evt in prof.key_averages():
            if hasattr(evt, 'flops') and evt.flops is not None:
                flops = evt.flops
        return flops
    except Exception:
        return None
