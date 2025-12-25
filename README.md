# partdiff-py

This is a C++ port of [`partdiff`](https://github.com/parcio/partdiff).

## Usage

<!--
    Shorter example used because of bad performance.
    TODO: Select longer example if performance is improved.
-->

```shell
$ git clone https://github.com/felsenhower/partdiff-py.git
$ cd partdiff-py
$ uv run main.py 1 2 100 1 2 5
```

## Correctness

This project uses [partdiff_tester](https://github.com/parcio/partdiff_tester) via CI to ensure that the output matches the reference implementation.
It passes the correctness tests with `--strictness=4` (exact match).

We are currently using `--max-num-tests=10` because the performance is quite bad.

## Performance

Currently, `partdiff-py` is slower than the reference implementation by a factor of about 200.

This makes it unsuitable for real-world use cases.

This is probably largely due to the very naïvely implemented main loop. This can probably be improved by leaning more heavily on numpy features.
