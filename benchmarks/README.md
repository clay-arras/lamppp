## Benchmarks:

Across the board, my library consistently faster than Pytorch for the CUDA implementations, and many orders of magnitudes slower for the CPU implementation (but I mean who uses CPU for machine learning anyway ;)). Would like to note that these benchmarks just look at operators --Pytorch also has optimizations when dealing with groups of operators, which was not benchmarked here. 

Commands run: 

```bash
build/benchmarks/reg_bench_long --benchmark_format=csv --benchmark_time_unit=us --benchmark_min_time=0.01s > bench_lmp.csv
python -m pt.all_tests --output-csv bench_torch.2.csv --min-time-per-test 10
```

- GPU: RunPod, NVIDIA RTX 4000 Ada
- CPU: RunPod, 47 GB RAM + 9 vCPU

Docker container: https://hub.docker.com/repository/docker/clayarras/lamppp/general

![image](https://github.com/user-attachments/assets/a88e19b3-103e-4e74-ae3e-444d0a35e059)
![image](https://github.com/user-attachments/assets/38618e8a-2aba-449b-a0f7-6a6fe63c32f3)
![image](https://github.com/user-attachments/assets/625b5ac0-6226-43bb-bb3b-02a3ee6b03ac)

