#!/bin/bash

J=15
K=7

# Submit jobs
for SEED in {0..20}
do
    sbatch exprQ_job.sh $J $K $SEED
done

