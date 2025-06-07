## Benchmarks:

Notes: forward is around the same order of magnitude as Pytorch. Have not benched Pytorch backwards yet.
Some results are 1-2 orders of magnitude slower than what they should be. not sure why, maybe due to Pytorch's tensor management / reusing memory, or just due to tighter loops

GPU: RunPod, NVIDIA RTX 4000 Ada