# MCCL Goals

## One sentence description

MMCL strives to be a flexible and optimized framework to perform code cryptanalysis.

## Goals

1. Propose a general framework to develop reference implementations of information set decoding (ISD) algorithms.
2. These implementations should be able to solve instances of the decoding challenge.
3. Rely on strong optimized algorithms in back end to be used as shared building blocks for the different syndrome decoding algorithms.
4. The reference implementation should enable people to quickly plug-in ideas of tweaks as well as theoretical improvements and compare their efficiency, instead of starting a new implementation from scratch.
5. This project aims at providing good reference points to benchmark improvements and
new implementations. Moreover, it should encourage people to provide new benchmark
information on their implementations.
6. Finally, the ultimate goal of this project is to broaden the community of people working on information set decoding algorithms and trying to solve challenges. This means that a particular attention should be given to make the project accessible, documented, easy to use and to contribute to.

## Structure

The aimed structured is:
- a low-level library for core operations
- various parametrized cryptanalysis algorithms implemented on top
- accompanying tool-suite

## Low-level operations

- matrix, vector operations
- permutation, submatrix tools
- (partial) gaussian elimination
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
- instance generator
- parser
- benchmarking
- computing optimal run-time parameters
- python layer
