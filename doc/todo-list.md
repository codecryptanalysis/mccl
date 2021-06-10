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
- New matrix type: `mat` and `vec` are pointers, `cmat` and `cvec` have const data. No more need to handle the scratch columns.
- Decoding algorithm split in `prange.hpp` and `LB.hpp`.
- Command line utility `isdsolver.cpp`, need to check the solution and add a dedicated test. 
- M4RI library is now useless, we can remove it (or at lease make it optional).
- Next step: new setting. Transpose (so that we have the row permutation for free) and reverse (column reduction starting from the right) to easily obtain a multiple of 64.
- The `echelon` and `permute` functions are going to be dependant. Should we merge them?
- Statistics. We have some Python code to compute the success probability. We could  try to see if we obtain the same results in practice. Also later we will need more statistical data (use .json format?). 

## 24/06/21
- [ ] drop the M4RI dependency
- [ ] implement (reverse) column reduction
- [ ] rewrite Prange+LB in the new transpose/reverse setting
- [ ] write LB for l>0 using the new transpose/reverse setting
- [ ] documentation of the new setting and API
- [ ] `isdsolver.cpp` check that the solution is correct
- [ ] add a test for `isdsolver.cpp`
- [ ] add a parameter to run the solver multiple times and compute the success probability (compare with theoretical value from the Python tool)
- [ ] merge `echelon` and `permute`?
- [ ] python tool to compute the cost given the parameters
- [ ] more flexible and optimized row-reduction
- [ ] optimized (block) permutation

# Phase 2

- [ ] Birthday search
- [ ] Canteaut-Chabaud: only permute a small number of columns instead of a full permutation
- [ ] Stern/Dumer

Once this is achieved:
- See how this performs on challenge instances.
- Present the project to other members of the community for feedback.

# Phase 3

- [ ] MMT
- [ ] BJMM
