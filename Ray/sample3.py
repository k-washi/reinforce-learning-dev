import ray
import time

@ray.remote
class Worker():
    def __init__(self, worker_id) -> None:
        self._worker_id = worker_id
        self.n = 0

    
    def add(self, n):
        time.sleep(5)
        self.n += n
    
    def get_value(self):
        return f"Process {self._worker_id}: value: {self.n}"    
    

ray.init()
# ray.init(num_cpus=20, num_gpus=2) # gpuは明示が必要
# local_mode=True # worker内でprintを使用できるようになる
start = time.time()


workers = [Worker.remote(i) for i in range(5)]
for worker in workers:
    worker.add.remote(10)

for worker in workers:
    worker.add.remote(5)

for worker in workers:
    print(ray.get(worker.get_value.remote()))

print(f"Elapsed: {time.time() - start}")