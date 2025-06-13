## Benchmarks:

Notes: forward is around the same order of magnitude as Pytorch. Have not benched Pytorch backwards yet.
Some results are 1-2 orders of magnitude slower than what they should be. not sure why, maybe due to Pytorch's tensor management / reusing memory, or just due to tighter loops

GPU: RunPod, NVIDIA RTX 4000 Ada


Sum_axis0_64x512_Float32_CUDAForward/min_warmup_time:1.000        39515 ns        39330 ns        18488
Sum_axis0_64x512_Float32_CUDABackward/min_warmup_time:1.000      102126 ns       101923 ns         5892
Sum_axis1_64x512_Float32_CUDAForward/min_warmup_time:1.000        43113 ns        43051 ns        16839
Sum_axis1_64x512_Float32_CUDABackward/min_warmup_time:1.000      119886 ns       119351 ns         6748
Min_axis0_64x512_Float32_CUDAForward/min_warmup_time:1.000        40221 ns        40027 ns        17612
Min_axis0_64x512_Float32_CUDABackward/min_warmup_time:1.000      178442 ns       177835 ns         4119
Min_axis1_64x512_Float32_CUDAForward/min_warmup_time:1.000        42521 ns        42147 ns        13324
Min_axis1_64x512_Float32_CUDABackward/min_warmup_time:1.000      140681 ns       139967 ns         5120
Max_axis0_64x512_Float32_CUDAForward/min_warmup_time:1.000        37458 ns        37399 ns        18306
Max_axis0_64x512_Float32_CUDABackward/min_warmup_time:1.000      135925 ns       135680 ns         5340
Max_axis1_64x512_Float32_CUDAForward/min_warmup_time:1.000        36271 ns        36078 ns        18838
Max_axis1_64x512_Float32_CUDABackward/min_warmup_time:1.000      146665 ns       146478 ns         5092
Prod_axis0_64x512_Float32_CUDAForward/min_warmup_time:1.000       35262 ns        35170 ns        20506
Prod_axis0_64x512_Float32_CUDABackward/min_warmup_time:1.000     166742 ns       166684 ns         4865
Prod_axis1_64x512_Float32_CUDAForward/min_warmup_time:1.000       43605 ns        43585 ns        15996
Prod_axis1_64x512_Float32_CUDABackward/min_warmup_time:1.000     135566 ns       135331 ns         5133
