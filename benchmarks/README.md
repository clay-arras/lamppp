## Benchmarks:

Across the board, my library is pretty much the same speed as Pytorch for the CUDA implementations, and many orders of magnitudes slower for the CPU implementation (but I mean who uses CPU for machine learning anyway ;))

Commands run: 
`build/benchmarks/reg_bench_long --benchmark_format=csv --benchmark_time_unit=us --benchmark_min_time=0.01s > bench_lmp.csv`
`python -m pt.all_tests --output-csv bench_torch.2.csv --min-time-per-test 10`

GPU: RunPod, NVIDIA RTX 4000 Ada
Docker container: https://hub.docker.com/repository/docker/clayarras/lamppp/general