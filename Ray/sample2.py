import ray
import random
import time

@ray.remote
def worker_func(pid):
    time.sleep(random.randint(3, 15))
    return f"pid {pid} fin."

ray.init()
start = time.time()

work_in_progress = [worker_func.remote(i) for i in range(10)]

for i in range(10):
    fin, work_in_progress = ray.wait(work_in_progress, num_returns=1) # 終了したタスクのリスト、終了していないタスクのリストを返す
    orf = fin[0]
    print(fin)
    print(ray.get(orf))
    print(f"## Elapsed: {time.time() - start}")
    