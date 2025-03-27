from multiprocessing import Process, Queue
import random
from time import time

# Worker function: does the job and puts the result into a queue
def generate_and_sum(n, output_queue):
    total = sum(random.random() for _ in range(n))
    output_queue.put(total)

if __name__ == "__main__":
    N = 100_000_000
    num_workers = 4
    chunk_size = N // num_workers

    # Queue to collect results
    q = Queue()
    processes = []

    start = time()

    # Launch processes manually
    for _ in range(num_workers):
        p = Process(target=generate_and_sum, args=(chunk_size, q))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Collect results from the queue
    total = 0
    while not q.empty():
        total += q.get()

    print(f"Total: {total}")
    print(f"Time: {time() - start:.2f} s")
