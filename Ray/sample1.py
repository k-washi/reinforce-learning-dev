
import ray
import time

@ray.remote
def worker_func(pid):
    time.sleep(5)
    return f"pid {pid} finished"

ray.init()

start = time.time()
results = [worker_func.remote(i) for i in range(3)]
print(results)
print(f"途中: {time.time() - start}")
print(ray.get(results))
print("Elapsed:", time.time() - start) 