import time
start = time.time()
total = sum(range(1_000_000_000))
print(total)
print(f"Time: {time.time() - start} s")