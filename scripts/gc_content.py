import os
from time import time
from concurrent.futures import ProcessPoolExecutor
import psutil

# ---------- GC Utilities ----------

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq)
    return sequences

def compute_gc_content(seq):
    return sum(base in 'GC' for base in seq) / len(seq) if len(seq) > 0 else 0

# ---------- Single-threaded version ----------

def gc_single(sequences):
    return [compute_gc_content(seq) for seq in sequences]

# ---------- Chunked for parallel version ----------

def gc_parallel_chunk(chunk):
    return [compute_gc_content(seq) for seq in chunk]

def gc_parallel(sequences, num_workers):
    chunk_size = len(sequences) // num_workers
    chunks = [sequences[i:i+chunk_size] for i in range(0, len(sequences), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(gc_parallel_chunk, chunks)
    
    # Flatten the list of results
    return [gc for chunk in results for gc in chunk]

# ---------- Main Script ----------

if __name__ == "__main__":
    fasta_path = "data/sequences.fasta"
    num_workers = os.cpu_count() or 4  # fallback to 4 if detection fails

    print(f"Using {num_workers} workers")

    # Load sequences
    sequences = read_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequences")

    # Single-threaded
    start = time()
    gc_single_result = gc_single(sequences)
    single_time = time() - start

    # Parallel version
    start = time()
    gc_parallel_result = gc_parallel(sequences, num_workers)
    parallel_time = time() - start

    # Results (check they're the same)
    assert len(gc_single_result) == len(gc_parallel_result)
    avg_gc = sum(gc_single_result) / len(gc_single_result)

    print(f"\nGC content (average): {avg_gc:.4f}")
    print(f"Single-threaded time: {single_time:.2f} s")
    print(f"Parallel time ({num_workers} workers): {parallel_time:.2f} s")
