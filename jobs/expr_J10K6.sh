#!/bin/bash

J=10
K=6

# Submit jobs
for SEED in {0..20}
do
    sbatch exprQ_job.sh $J $K $SEED
done

