## Benchmarks:

Notes: forward is around the same order of magnitude as Pytorch. Have not benched Pytorch backwards yet.
Some results are 1-2 orders of magnitude slower than what they should be. not sure why, maybe due to Pytorch's tensor management / reusing memory, or just due to tighter loops

```
GPU Details:
NVIDIA GeForce RTX 4060
Driver Version: 570.133.07
CUDA Version: 12.8

2025-05-22T20:56:45-04:00
Run on (32 X 1425.88 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
Load Average: 3.21, 2.83, 2.84
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
AddForward128        88517 ns        88468 ns        11000
AddForward256        83267 ns        83224 ns         7800
AddForward1024      716446 ns       716134 ns          993
SubForward128        64639 ns        64582 ns        11685
SubForward256        88435 ns        88372 ns         9148
SubForward1024      792415 ns       790656 ns          901
MulForward128        94903 ns        94789 ns         6351
MulForward256       121192 ns       120903 ns         5042
MulForward1024      776050 ns       775246 ns          843
DivForward128        65349 ns        65330 ns         9141
DivForward256        79477 ns        79438 ns         9204
DivForward1024      722530 ns       722305 ns          966
AddBackward128      143378 ns       143353 ns         5048
AddBackward256      335813 ns       335434 ns         2387
AddBackward1024    1384170 ns      1383828 ns          442
SubBackward128       60023 ns        59981 ns        11513
SubBackward256      133340 ns       133256 ns         5287
SubBackward1024    1305569 ns      1305352 ns          544
MulBackward128       66957 ns        66929 ns        10844
MulBackward256      138687 ns       138641 ns         5120
MulBackward1024    1341321 ns      1340794 ns          563
DivBackward128       85054 ns        85031 ns         8465
DivBackward256      155463 ns       155441 ns         4525
DivBackward1024    1346857 ns      1346707 ns          550
AbsForward128         8540 ns         8530 ns        93965
AbsForward256         9751 ns         9714 ns        92052
AbsForward1024       13512 ns        13327 ns        43850
SinForward128         8004 ns         7996 ns       102000
SinForward256         8332 ns         8310 ns        75485
SinForward1024       16663 ns        16413 ns        50434
CosForward128         9616 ns         9598 ns        84326
CosForward256         8028 ns         8006 ns        70315
CosForward1024       11805 ns        11624 ns        73388
AbsBackward128       99925 ns        99911 ns         6749
AbsBackward256      217114 ns       217097 ns         3251
AbsBackward1024    2242052 ns      2241744 ns          310
SinBackward128       49484 ns        49477 ns        14005
SinBackward256      121156 ns       121120 ns         5849
SinBackward1024    1292504 ns      1292355 ns          522
CosBackward128       57885 ns        57878 ns        11749
CosBackward256      129519 ns       129487 ns         5667
CosBackward1024    1249491 ns      1249300 ns          584


```
