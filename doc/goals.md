# MCCL Goals

In short MMCL strives to be a flexible and optimized framework to perform code cryptanalysis.

## Structure

The aimed structured is:
- a low-level library for core operations
- various parametrized cryptanalysis algorithms implemented on top
- accompanying tool-suite:

## Low-level operations

- matrix, vector operations
- permutation, submatrix tools
- gaussian elimination
- birthday search

Sub-goals:
- flexible: to easily implement wide scope of algorithms mentioned below.
- allow low-level optimizations (minimize inner-loop operations, cache effects, simd)
- (allow to bitslice and/or offload operations on simd/gpu?)

## Desired algorithms

- Prange
- Lee-Brickell
- Stern/Dumer
- May-Meurer-Thomae
- Becker-Joux-May-Meurer

## Desired tool suite

- easy command line tool
- parser
- benchmarking
- computing optimal run-time parameters
- python layer
