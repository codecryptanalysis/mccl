# Phase 1

## Documentation
- [X] description of the goals
- [ ] pseudo-code description of the ISD algorithms
- [ ] small bibliography
- [ ] see how the "nearest neighbor" approach can fit in the API

## Core
- [ ] API description of the core objects
  - [X] matrix
  - [ ] permutation and permuted matrix
  - [X] ISD
- [ ] use the [m4ri](https://bitbucket.org/malb/m4ri/src/master/) library to instantiate the matrix class

## Algorithms
- [ ] implement Prange's algorithm

## Tools
- [ ] Python tool to optimize parameter choice (success probability etc.)

Once this is achieved:
- We should be able to solve some small instances using Prange's algorithm.

# Phase 2

## Core
- [ ] Birthday search
- [ ] Canteaut-Chabaud: only permute a small number of columns instead of a full permutation

## Algorithms
- [ ] Lee Brickel
- [ ] Stern/Dumer

## Tools
- [ ] Parser
- [ ] Instance generator
- [ ] Documentation of the implemented algorithms
- [ ] Tests

Once this is achieved:
- See how this performs on challenge instances.
- Present the project to other members of the community for feedback.

# Phase 3

## Algorithms
- [ ] MMT
- [ ] BJMM
