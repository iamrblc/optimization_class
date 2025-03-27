from concurrent.futures import ProcessPoolExecutor
import random
from time import time
import os

# Function that each worker runs
def generate_and_sum(n):
    return sum(random.random() for _ in range(n))

if __name__ == "__main__":
    N = 100_000_000
    num_workers = 100
    chunk_size = N // num_workers

    start = time()

    # Start parallel execution
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(generate_and_sum, [chunk_size] * num_workers)

    total = sum(results)
    print(f"Total: {total}")
    print(f"Time: {time() - start:.2f} s")
