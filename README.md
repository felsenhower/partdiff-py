# partdiff-py

This is a Python port of [`partdiff`](https://github.com/parcio/partdiff).

## Usage

```shell
$ git clone https://github.com/felsenhower/partdiff-py.git
$ cd partdiff-py/simple
$ uv run main.py 1 2 100 1 2 5
```

## Variants

The repository contains three variants:
- `simple`: An intentionally naïve and straightforward implementation (simple but slow)
- `np_vectorize`: An implementation that uses numpy's fast factorized math for the Jacobi method. For the Gauß-Seidel method, this is not possible[^1], so we're only having some minor simplifications here.
- `numba`: An implementation where the main loop has been JIT-compiled with numba.

All variants above use some shared code that can be found in `partdiff_common`.

[^1]: In general, it is not possible to parallelize the Gauß-Seidel without some form of synchronization if bitwise accuracy is needed. MPI can be used to parallelize Gauß-Seidel efficiently which works well for large problem sizes.

## Correctness

This project uses [partdiff_tester](https://github.com/parcio/partdiff_tester) via CI to ensure that the output matches the reference implementation.
It passes the correctness tests with `--strictness=4` (exact match).

Since the performance of some variants (especially `simple`) is not great, we run fewer tests here (e.g. only `interlines=0` for `simple`).

## Performance

See the table below for a runtime comparison of the variants that has been created with the scripts inside the `benchmark` directory. The C reference implementation serves as a comparison.

For all benchmarks, the arguments `1 {1,2} 100 2 2 100` were used. Therefore, this only serves to give you a rough overview.

`runtime_internal` shows the runtime that partdiff measured (the `Calculation time` field in the output) and `runtime_total` shows the runtime measured with `time`.

All Python implementations below have a larger runtime in total than the reference implementation. Since all of the startup code (arg-parsing, matrix initialization) were written in a pythonic and straightforward way, this is not surprising. With that in mind, I will only look at the internally measured runtime below.

As expected, the naïve implementation (`simple`) performs very badly. Here, the reference implementation is roughly 100x faster.

Same goes for the `np_vectorize` version with the Gauß-Seidel method which is even slightly slower than `simple`. Although it's not surprising that this is the case (since it contains an extra function call), it _is_ surprising that this is adding over 3 seconds of runtime.

With the Jacobi method, the `np_vectorize` version is even faster than the reference implementation, thanks to numpy's optimized vectorized math.

Finally, the `numba` version shows a comparable performance to the reference implementation, being slightly faster for Jacobi and slightly slower for Gauß-Seidel. 

| variant      | method      | runtime_internal   |                         | runtime_total      |                      |
|--------------|-------------|--------------------|-------------------------|--------------------|----------------------|
| reference    | Gauß-Seidel | (0.563 ± 0.023) s  | 100.00%                 | (0.567 ± 0.029) s  | 100.00%              |
| reference    | Jacobi      | (0.490 ± 0.017) s  | 100.00%                 | (0.493 ± 0.023) s  | 100.00%              |
| simple       | Gauß-Seidel | (51.817 ± 0.273) s | 9198.22%                | (52.107 ± 0.273) s | 9195.29%             |
| simple       | Jacobi      | (52.287 ± 0.508) s | 10670.75%               | (52.250 ± 1.087) s | 10591.22%            |
| numba        | Gauß-Seidel | (0.703 ± 0.023) s  | 124.85%                 | (1.150 ± 0.017) s  | 202.94%              |
| numba        | Jacobi      | (0.417 ± 0.006) s  | 85.03%                  | (0.860 ± 0.010) s  | 174.32%              |
| np_vectorize | Gauß-Seidel | (55.177 ± 0.303) s | 9794.67%                | (55.467 ± 0.303) s | 9788.24%             |
| np_vectorize | Jacobi      | (0.213 ± 0.006) s  | 43.54%                  | (0.497 ± 0.012) s  | 100.68%              |

