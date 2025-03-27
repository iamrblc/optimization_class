import torch
import time

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

# Encode ACGT as integers
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
def encode_sequences(sequences):
    encoded = []
    for seq in sequences:
        numeric = [BASE_MAP.get(base, 0) for base in seq]  # fallback A for unknown
        encoded.append(numeric)
    return encoded

# Compute GC content using PyTorch
def compute_gc_torch(encoded_seqs, device):
    tensor = torch.tensor(encoded_seqs, dtype=torch.int32, device=device)
    # count how many elements are 1 (C) or 2 (G)
    is_gc = (tensor == 1) | (tensor == 2)
    gc_counts = is_gc.sum(dim=1)
    gc_fraction = gc_counts / tensor.shape[1]
    return gc_fraction.cpu()

if __name__ == "__main__":
    sequences = read_fasta("data/sequences.fasta")
    print(f"Loaded {len(sequences)} sequences")
    encoded = encode_sequences(sequences)

    # CPU version
    start = time.time()
    gc_cpu = compute_gc_torch(encoded, device="cpu")
    cpu_time = time.time() - start

    # MPS (GPU on Mac) version
    if torch.backends.mps.is_available():
        start = time.time()
        gc_gpu = compute_gc_torch(encoded, device="mps")
        gpu_time = time.time() - start
    else:
        gc_gpu = None
        gpu_time = None

    print(f"CPU GC avg: {gc_cpu.mean():.4f}, Time: {cpu_time:.2f} s")
    if gc_gpu is not None:
        print(f"GPU (MPS) GC avg: {gc_gpu.mean():.4f}, Time: {gpu_time:.2f} s")
    else:
        print("GPU not available")