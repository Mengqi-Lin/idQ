# Raw experiment output

Local experiment commands write into `bernoulli/` and `row_sparsity/`.
Cluster arrays add a `run_<array-job-id>/` directory and a separate CSV per seed.
Do not combine seed outputs by appending concurrently to a shared CSV.
The row-sparsity summary command recursively reads completed matching CSVs;
`.part` files indicate incomplete writes and are excluded.
