# partdiff-py

This is a Python port of [`partdiff`](https://github.com/parcio/partdiff).

## Usage

```shell
$ git clone https://github.com/felsenhower/partdiff-py.git
$ cd partdiff-py/simple
$ uv run main.py 1 2 100 1 2 5
```

## Variants

The repository contains multiple variants:
- `simple`: An intentionally naïve and straightforward implementation (simple but slow)
- `np_vectorize`: An implementation that uses numpy's fast factorized math for the Jacobi method. For the Gauß-Seidel method, this is not possible[^1], so we're only having some minor simplifications here.
- `numba`: An implementation where the main loop has been JIT-compiled with numba.
- `cython`: An implementation where the `calculate()` function has been rewritten in `Cython` and is effectively used as an external C module.
- `nuitka`: An implementation that uses Nuitka to compile the Python code into a single binary.

All variants above use some shared code that can be found in `partdiff_common`.

## Correctness

This project uses [partdiff_tester](https://github.com/parcio/partdiff_tester) via CI to ensure that the output matches the reference implementation.
It passes the correctness tests with `--strictness=4` (exact match).

Since the performance of some variants (especially `simple`) is not great, we run fewer tests here (e.g. only `interlines=0` for `simple`).

## Performance

See the table below for a runtime comparison of the variants that has been created with the scripts inside the `benchmark` directory. The C reference implementation serves as a comparison.

For all benchmarks, the arguments `1 {1,2} 100 2 2 100` were used. Therefore, this only serves to give you a rough overview.

`runtime_internal` shows the runtime that partdiff measured itself (the `Calculation time` field in the output) and `runtime_total` shows the runtime measured via `time.perf_counter()`.

All Python implementations below have a larger runtime in total than the reference implementation.
Since all of the startup code (arg-parsing, matrix initialization) is written in a pythonic and straightforward way, this is not surprising.
Since we're mainly interested in the calculation part, we'll only look at the internally measured runtime from now on.

As expected, the naïve implementation (`simple`) performs very badly. Here, the reference implementation is over 100x faster.

The `nuitka` variant is _slightly_ faster than the `simple` variant (looking at the standard deviation, take that with a grain of salt).

For the Gauß-Seidel method, the `np_vectorize` variant is about as fast as the `simple` variant which is not surprising since the `calculate_gauss_seidel()` method is nearly identical to the standard `calculate` method and only contains some obvious tweaks (e.g. no matrix swapping).
However, for the Jacobi method, the `np_vectorize` variant is even faster than the reference implementation (about 25% faster). This is thanks to numpy's optimized vectorized math while the reference implementation does not use vectorization. The perturbation matrix is also precomputed once inside the `calculate()` method instead of all the values being recomputed for each element (caching the matrix _does_ make sense in general, but since the reference implementation doesn't do it, we're also not doing it as long as it's not necessary).

The `numba` variant performs a bit worse than the reference implementation, the reference implementation being about 2x faster.

Finally, the performance of the `cython` variant is nearly identical to the reference implementation. Also not surprising, since the Cython code is an almost straight port of the C code, and Cython then translates that back to C and compiles it with exactly the same optimizations that the C version uses by default.

| variant      | method      | runtime_internal     |                         | runtime_total        |                      |
|--------------|-------------|----------------------|-------------------------|----------------------|----------------------|
| reference    | Gauß-Seidel | (0.5626 ± 0.0063) s  | (1.0000 ± 0.0159)       | (0.5640 ± 0.0061) s  | (1.0000 ± 0.0154)    |
| reference    | Jacobi      | (0.4921 ± 0.0032) s  | (1.0000 ± 0.0091)       | (0.4939 ± 0.0031) s  | (1.0000 ± 0.0089)    |
| simple       | Gauß-Seidel | (58.9728 ± 0.0725) s | (104.8218 ± 1.1836)     | (59.1671 ± 0.0721) s | (104.9124 ± 1.1500)  |
| simple       | Jacobi      | (59.9409 ± 0.2698) s | (121.7980 ± 0.9576)     | (60.1439 ± 0.2765) s | (121.7734 ± 0.9513)  |
| nuitka       | Gauß-Seidel | (56.1754 ± 0.7793) s | (99.8496 ± 1.7818)      | (56.3520 ± 0.7951) s | (99.9208 ± 1.7811)   |
| nuitka       | Jacobi      | (58.2192 ± 1.7328) s | (118.2996 ± 3.6026)     | (58.3895 ± 1.7231) s | (118.2213 ± 3.5678)  |
| np_vectorize | Gauß-Seidel | (60.2394 ± 1.2457) s | (107.0732 ± 2.5194)     | (60.4389 ± 1.2378) s | (107.1674 ± 2.4860)  |
| np_vectorize | Jacobi      | (0.3656 ± 0.0018) s  | (0.7429 ± 0.0060)       | (0.5601 ± 0.0025) s  | (1.1340 ± 0.0088)    |
| numba        | Gauß-Seidel | (1.1685 ± 0.0039) s  | (2.0770 ± 0.0243)       | (1.5011 ± 0.0153) s  | (2.6617 ± 0.0397)    |
| numba        | Jacobi      | (1.1692 ± 0.0116) s  | (2.3758 ± 0.0281)       | (1.5184 ± 0.0254) s  | (3.0743 ± 0.0549)    |
| cython       | Gauß-Seidel | (0.5207 ± 0.0086) s  | (0.9256 ± 0.0185)       | (0.7169 ± 0.0094) s  | (1.2712 ± 0.0217)    |
| cython       | Jacobi      | (0.5208 ± 0.0077) s  | (1.0583 ± 0.0171)       | (0.7197 ± 0.0160) s  | (1.4571 ± 0.0337)    |

## Conclusion

I think we can get some interesting insights from this data:

1. Python loops are very very very slow.
2. Numpy is really great when fully exploited, but it does _not_ solve all of our problems: While it's true that Python code written with Numpy can be extremely fast as long as vectorized math is used, there are simply some algorithms that can not be vectorized. The Gauß-Seidel method used in partdiff is an excellent example for an algorithm that can _not_ be vectorized[^1]. To make Gauß-Seidel as fast as the C version, we need to use some fancier (but also more complex) tricks (see below). But who knows, maybe the Numpy developers will introduce an iterator supporting stencil calculations somehow in the future. And of course, I have to admit that when Numpy vectorized math _does_ work, the process is relatively pain free since we're still just writing relatively simple Python code.
3. With numba, we can relatively easily get performance that is in the same ballpark as the C version. For `partdiff`, the only things it had problems with was the `dataclass` arguments and the `time()` method. As long as your algorithm only uses simple types (including enums), using numba is relatively easy. Therefore, **`numba` definitely has the highest ratio of performance gain per hours wasted**.
4. With Cython, we can get even quicker than with numba, getting performance relatively identical to C. This comes at the cost of being relatively annoying to write:
    - All the `cdef` directives add a lot of clutter and you need to `cdef` _everything_.
    - If you don't know what you're doing, Cython can just fall back to slow Python if it doesn't know how to handle something. This can be _very_ annoying.
    - Tooling is a bit cumbersome. While `uv` makes the compilation process quite easy (if you know how), there is no formatter for Cython yet (but ruff might support it in the future [[1]](https://github.com/astral-sh/ruff/issues/10250)).
  While I would argue that writing modern C is _less_ annoying that writing Cython, I still think it's great that we can have these little islands of native code in our Python applications and still get the nice bits of Python for the non-sensitive stuff. So that might ultimately be worth it.
5. Nuitka is a tool that might make deployment of Python applications slightly easier, but the claim that it boosts your performance is relatively hollow. Setting up a buggy, oversensitive and errorprone toolchain and waiting several minutes (for a fully optimized build) per build for a 2–5% performance boost is _not_ worth it if you ask me.

[^1]: In general, it is not possible to parallelize the Gauß-Seidel method without some form of synchronization while still retaining bitwise reproducability. MPI can be used to parallelize Gauß-Seidel efficiently which works well for large problem sizes.
