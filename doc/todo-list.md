# Phase 1

## 14/01/21

Discussion
- challenge website
- API-oriented approach
- main objects: matrix, permutation
- low level blocks: binary matrix operations, birthday search 
- use of the [M4RI](https://bitbucket.org/malb/m4ri/src/master/) library for matrix operations

## 28/01/21

- [x] creation of the [git repository](https://github.com/cr-marcstevens/mccl/) 
- [x] build system and skeleton
- [x] description of the [goals](https://github.com/cr-marcstevens/mccl/blob/main/doc/goals.md) of the project and [different steps](https://github.com/cr-marcstevens/mccl/blob/main/doc/todo-list.md)
- [x] [matrix API](https://github.com/cr-marcstevens/mccl/blob/main/doc/lowlevel.md)

Discussion
- M4RI requires adaptation
- permuted matrix object
- ISD API using callback functions
- bibliography
- compatibility of the nearest neighbor approach

## 11/02/21

- [x] interface to M4RI row reduction, multiplication, etc for matrices with 64 bits int
- [x] list of references [biblio.md](https://github.com/cr-marcstevens/mccl/blob/main/doc/biblio.md)
- [x] [script](https://github.com/cr-marcstevens/mccl/blob/main/tools/challenges.py) to generate/download challenges

Discussion
- Tools to compute the complexity and optimize the choice of parameters
- Pseudo-code of the ISD algorithms
- Optimized permutations

## 25/02/21

- [x] [pseudo-code](https://github.com/cr-marcstevens/mccl/blob/main/doc/algorithms.md) for the ISD_generic loop
- [x] [python tool](https://github.com/cr-marcstevens/mccl/blob/main/tools/probability.py) to compute the expected number of solutions returned by sub-ISD

Discussion:
- transposition of the H1/H2 part after the Gaussian elimination (for all versions but Prange)
- optimizing permutations: three options
	- random permutation
 	- permuting only a small set of random columns
	- permuting only a small set of columns chosen in a particular way to optimize the  
- pre-filtering in the sub-ISD loop
	- Instead of giving the matrix H2 of size (k+ell)*ell, we give as input to Sub-ISD a the matrix H2' of size (k+ell)*ell2 such that ell2 is the smallest multiple of 64 greater or equal to ell. The first rows of H2' correspond to the last rows of H1.
	- In Sub-ISD, the condition H2 * e2 = s2 is replaced by checking that the last ell bits of H2' * e2 are equal to s2. We also compute the weight w2 of the first (ell2-ell) bits of H2 * e2. If the relative weight value w2 is too high, we discard the solution. Else, we return in L the solution (e2,w2).

## 11/03/21
- [x] parser to use the instances of the challenge as input
- [x] write a more detailed pseudocode for the sub_isd (including optimization tricks)

## 25/03/21
- [x] basic echelonize function
- [x] non-optimized random permutation of columns

## 08/04/21
- [x] first implementation of Prange's algorithm
- [x] presentation of MTT and BJMM papers

Discussion:
- tools needed for MMT/BJMM
	- generating base lists
	- merging
	- python tool to evaluate complexity given parameters
- merging lists
	- lookup tables vs sorting
	- [Landais](https://tel.archives-ouvertes.fr/tel-01142563/document) (Chap 11): lookup tables more efficient, in case of collision only store one element, to simplify the data structure
	- direct addressing (using the last l bits as the address)
- base lists
	- construct two lists at level 1 using the same left/right baselists (with different targets) 
- expected size of the parameters?
	- see [Hobach](https://hackingthe.net/downloads/isd.pdf) (pp. 70-71)
	
## 26/04/21
- [x] fix Prange test
- [x] write Prange following the ISD API
- [x] presentation of Hobach's experimental results

Discussion
- Prange works
- fast enumeration of sums of columns: 
	- add the 'origin' vector to all sums
	- no callback: store the results
	- same API can directly return a hashmap
- Hobach's work
	- most proposed optimizations use binary operations and do not take into account the fact that we have 64 bits integers 
	- early abort on multiples of 64 bits
	- full Gaussian elimination? no because we will not treat pivot columns differently from random columns in H2

## 10/05/21
- [x] Fast enumeration of sums of columns
- [x] Lee-Brickell (not in API)

## 31/05/21
- [x] API single generic callback function
- [x] Prange and Lee-Brickell following the API

## 10/06/21
- [x] simplification of the `matrix` type
- [x] command line utility
- [x] cleaning existing code for Prange/LB

Discussion
- New matrix types: `mat_view` and `vec_view` are like pointers to a non-const data (you can still modify data through `const mat_view`), `cmat_view` and `cvec_view` have const data. `mat` and `vec` manage memory and behave like actual matrix/vector objects: automatic resizing on assignment and const data for const objects. Also no more need to handle the scratch columns.
- Decoding algorithm split in `prange.hpp` and `LB.hpp`.
- Command line utility `isdsolver.cpp`, need to check the solution and add a dedicated test. 
- M4RI library is now useless, we can remove it (or at lease make it optional).
- Next step: new setting. Transposed H (so that we have the row permutation for free) and reverse (column reduction starting from the right) to easily obtain a multiple of 64 for H1T.
- The `echelon` and `permute` functions are going to be dependant. Should we merge them?
- Statistics. We have some Python code to compute the success probability. We could  try to see if we obtain the same results in practice. Also later we will need more statistical data (use .json format?). 

## 24/06/21
- [x] drop the M4RI dependency
- [x] add a parameter to run the solver multiple times 
- [x] compute the success probability (compare with theoretical value from the Python tool)

Discussion
- Comparison of success probability (observed vs theoretic) match for Prange but not Lee Brickell. There is a but in LB (callback function). 
- Statistics: add a counter to each object, gather all counters at the end. One should be able to run the algorithms several times on the same object and keep track of all the counters / the average of the counters.
- Need to make an average on different instances, not just rerun the algorithm on the same instance: we need an instance generator (random instances in the form of the SD problem). 
- Merging `echelon` and `permute`: ongoing work.
- Add the case p=0 to row enumeration (corner case).

## 08/07/21
- [x] fix Lee Brickell bugs
- [x] check the success probability for fixed Lee-Brickell
- [x] generator of random SD instances
- [x] handling options for command line interface
- [x] merge `echelon` and `permute`, allow reverse column reduction (anti-diagonal)
- [x] rewrite Prange+LB in the new transposed setting
- [x] optimized column randomization: swap a small number of columns
- [x] Lee-Brickell with `l`>0
- [x] print more statistics

Discussion
- new HST setting
- new `sparserange` format for vectors
- improved column permutation by swaping a small number of columns. How to compute the success probability? Using Markov chains, see [BLP08](https://eprint.iacr.org/2008/318.pdf) (and associated code [here](https://github.com/christianepeters/isdf2/)).

## TODO for next session
- [ ] delete the code related to the non-transposed setting
- [ ] compute success probablity for LB with `l`>0
- [ ] improve the `update` (column swap), add an option (`u=-1`?) for full random permutation
- [ ] compute success probability with optimized column permutation using Markov chain (see if it matches the observations)
- [ ] add an option to stop the algorithm after a fixed number of iterations (and/or time limit)  
- [ ] statistics: count number of calls to each function
- [ ] store a table giving the value `w` in terms of `n` for the SD challenge and make this the default value of `w`
- [ ] `isdsolver.cpp` check that the solution is correct
- [ ] add a test for `isdsolver.cpp`
- [ ] documentation of the new HST setting

# Phase 2

- [ ] Birthday search
- [ ] Stern/Dumer

Once this is achieved:
- See how this performs on large challenge instances.
- Present the project to other members of the community for feedback.

# Phase 3

- [ ] MMT
- [ ] BJMM
