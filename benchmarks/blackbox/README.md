# Standard Blackbox Optimization Benchmark

This benchmark provides utilities to query some well-established benchmark functions in the blackbox optimization field.
The benchmark can be used for testing new algorithms and models for blackbox optimization.

There two available `Benchmark` subclasses:

* `blackbox.Ackley`, which implements the Ackley function
* `blackbox.Rosenbrock`, which implements the Rosenbrock function

## Ackley Function

The [Ackley function](https://www.sfu.ca/~ssurjano/ackley.html) is a non-convex function used as a performance test problem for optimization algorithms.
It was proposed by David Ackley in his 1987 PhD dissertation.

It can be computed as:
$$f(x) = -a \cdot e^{-b \cdot \sqrt{\frac{1}{d} \sum_i x_i^2}} - e^{\frac{1}{d} \sum_i cos(c \cdot x_i)} + a + e$$
and has its minimum in $f(0, ..., 0) = 0$.

## Rosenbrock Function

The [Rosenbrock function](https://www.sfu.ca/~ssurjano/rosen.html) (a.k.a. Rosenbrock's valley or Rosenbrock's banana function) is a non-convex function.
It was introduced by Howard H. Rosenbrock in 1960, and it is used as a performance test for optimization algorithms.
The global minimum is inside a long, narrow, parabolic shaped flat valley.
Hence, to find the valley is trivial, while to converge to the global minimum is difficult.

It can be computed as:
$$f(x) = \sum_{i=1}^{d-1} [b \cdot (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]$$
and has its minimum in $f(1, ..., 1) = 0$.
