import sys
import numpy as np
import csv
from idQ import identifiability

def process_batch(batch_file, output_csv):
    batch = np.load(batch_file, allow_pickle=True)
    identifiable_matrices = []
    for Q in batch:
        if identifiability(Q):
            identifiable_matrices.append(Q)
    
    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for Q in identifiable_matrices:
            flat = Q.flatten() if hasattr(Q, "flatten") else [item for row in Q for item in row]
            writer.writerow(flat)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_batch.py <batch_file.npy> <output_csv>")
        sys.exit(1)
    batch_file = sys.argv[1]
    output_csv = sys.argv[2]
    process_batch(batch_file, output_csv)
