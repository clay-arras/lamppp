use: https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html

- then make sure you add the shared_ptr as a custom type
- next step is to change the Layer Implementation to accept the new values

- need to add linting, comments, cleaner code, assertions, etc.
- integrating Eigen expression templates
- memory pool or custom allocator
- using manual pointer allocation
- replace lambda functions with static function pointers

documentation:

- creating many layers:

```cpp
    int nin = 108 * 108 * 3;
    Layer W1(nin, 1024);
    Layer W2(1024, 512);
    Layer W3(512, 256);
    Layer W4(256, 128);
    Layer W5(128, 10);
```

    > FastLayer is 1.604, Layer is 13.399

- setting up cpp tools and project folders

### 1. Benchmarking Frameworks

- **Google Benchmark:** Framework for microbenchmarks in C++.
- **Catch2 Benchmarking:** Built-in features for testing with Catch2.
- **Custom Timers:** Use `std::chrono::high_resolution_clock` for timing.

### 2. Profiling Tools

- **gprof:** GNU profiler for time breakdown.
- **perf:** Linux tool for hardware performance insights.
- **Valgrind (Callgrind):** Simulates CPU for function call profiling.
- **Intel VTune Amplifier:** Commercial tool for performance bottleneck analysis.
