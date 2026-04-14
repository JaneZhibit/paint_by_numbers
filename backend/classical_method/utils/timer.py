import time
from functools import wraps


class PipelineTimer:
    def __init__(self):
        self.timings = {}

    def time_stage(self, stage_name):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                start_time = time.perf_counter()
                result = func(self, *args, **kwargs)
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                self.timings[stage_name] = elapsed
                return result
            return wrapper
        return decorator


def time_stage(stage_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()
            result = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            self.timings[stage_name] = elapsed
            return result
        return wrapper
    return decorator


def print_timing_report(timings):
    if not timings:
        print("No timing data available.")
        return
    
    print("\n" + "="*50)
    print("TIMING REPORT")
    print("="*50)
    
    total_time = sum(timings.values())
    
    for stage_name, elapsed_time in timings.items():
        percentage = (elapsed_time / total_time * 100) if total_time > 0 else 0
        print(f"{stage_name:<20} {elapsed_time:>10.4f}s  ({percentage:>6.2f}%)")
    
    print("-"*50)
    print(f"{'TOTAL':<20} {total_time:>10.4f}s  (100.00%)")
    print("="*50 + "\n")
