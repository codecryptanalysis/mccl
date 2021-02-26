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

- [x] adapting the row reduction of M4RI to 64 bits int
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
- [ ] finish adapting M4RI to have efficient matrix transposition
- [ ] implementation of Prange's algorithm
- [ ] parser to use the instances of the challenge as input
- [ ] write a more detailed pseudocode for the sub_isd (including optimization tricks)
- [ ] python tool to compute the cost given the parameters (with the actual cost of each operation as a variable)

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
