import itertools
import numpy as np
import os
from expr_function import binary_matrix_to_string, sort_lexicographically, random_generate_Q, prop_check, sort_binary_matrix
import networkx as nx


def generate_canonical_matrices_gen(J, K):
    """
    Generator for canonical Q-matrices of dimension J x K.
    
    This version yields one matrix at a time rather than building the full list in memory.
    
    Note: This simplified version uses a single canonicalization step (using sort_binary_matrix)
    and assumes that’s sufficient. If you need the iterative refinement as in your original code,
    you may need to adapt this approach further (e.g., by writing intermediate results to disk).
    """
    # Generate all possible binary sequences of length K (excluding the all-zeros vector)
    all_sequences = [seq for seq in itertools.product([0, 1], repeat=K) if any(seq)]
    seen = set()  # To track canonical matrices we've already yielded
    
    # Iterate over combinations of J unique sequences
    for selected_sequences in itertools.combinations(all_sequences, J):
        Q = np.array(selected_sequences)
        # Get the canonical form of Q.
        # Adjust the parameters as needed. Here we use a fixed ordering.
        Q_canonical = sort_binary_matrix(Q, columnfirst=False)
        # Convert to a string representation for uniqueness checking.
        Q_string = binary_matrix_to_string(Q_canonical)
        if Q_string not in seen:
            seen.add(Q_string)
            yield Q_canonical
            
            
            
            
def batch_iterator(generator, batch_size=1000):
    """
    Yields batches of matrices from a generator.
    """
    batch = []
    for Q in generator:
        batch.append(Q)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


        
        
        


def main():
    J, K = 11, 10
    # Use the generator version so matrices are produced on the fly.
    matrix_generator = generate_canonical_matrices_gen(J, K)
    
    os.makedirs("batches", exist_ok=True)
    os.makedirs("jobs", exist_ok=True)
    
    batch_id = 0
    for batch in batch_iterator(matrix_generator, batch_size=1000):
        batch_file = f"batches/batch_{batch_id}.npy"
        # Save the current batch; using dtype=object to properly save each matrix.
        np.save(batch_file, np.array(batch, dtype=object))
        
        # Create an sbatch job script for this batch.
        job_file = f"jobs/job_batch_{batch_id}.sh"
        with open(job_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name=check_id_batch_{batch_id}\n")
            f.write(f"#SBATCH --output=jobs/job_batch_{batch_id}.out\n")
            f.write(f"#SBATCH --error=jobs/job_batch_{batch_id}.err\n")
            f.write("#SBATCH --time=01:00:00\n")
            f.write("#SBATCH --mem=4G\n")
            f.write("\n")
            f.write(f"/sw/pkgs/arc/python3.10-anaconda/2023.03/bin/python /home/lemonkey/idQ/py/ process_batch.py {batch_file} identifiable_matrices.csv\n")
        
        # Submit the job.
        os.system(f"sbatch {job_file}")
        batch_id += 1

if __name__ == "__main__":
    main()

        
        
        
        
