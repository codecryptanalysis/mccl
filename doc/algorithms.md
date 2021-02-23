# Algorithms

## Objects

- `matrix`
- `vector`
- `permutation`
- `permuted_matrix`: this objects corresponds to a matrix and a permutation. The permutation keeps track of the column permutations that were applied to the matrix. A constructor takes a matrix as input and puts the identity permutation as default.

## Functions

- `gaussian_elimination`
  - Input: permuted_matrix pH, vector s, int c
  - Output: null
  - This operation takes the matrix from the permuted_matrix object and performs gaussian elimination so as to make the top-left c x c submatrix the identity matrix, and the bottom-left (n-k-c) x c submatrix the all-zero matrix. The same row-operations are applied to the vector s (viewed as a (n-k) x 1 vector). All of these operation are applied in place. For this reason, nothing is output.
- `inverse_permutation`
  - Input: permutation
  - Output: permutation
  - Returns the inverse permutation.
- `permute_vector`
  - Input: vector, permutation
  - Output: null
  - Applies the permutation to the vector.
- `permute_matrix`
  - Input: permuted_matrix pH, permutation
  - Output: null
  - Applies a column permutation on the matrix and changes the permutation stored in the permuted_matrix to keep track of this operation.
- `random_permute_matrix`
  - Input permuted_matrix pH
  - Output: null
  - Same as permute_matrix but with a random permutation.

## ISD generic algorithm

``` cpp
vector ISD_generic(matrix H, vector s, int w, int p, int ell):
    int n = H.dimensions[0]
    int k = H.dimensions[1]
    permuted_matrix pH = permuted_matrix(H)
    while(true)
        random_permute_matrix(pH)
        gaussian_elimination(pH, s, n-k-ell)
        //       ||     Id    |   H1   || s1 (n-k-ell) 
        //  pH=  ||     0     |   H2   || s2 (ell)
        //       ----------------------
        //  e=   ||     e1    |   e2   ||
        //         (n-k-ell)    (k+ell)
        L = Sub_ISD(H2, s2, p)
        // Returns a set of vectors e2 of length k+ell
        // s.t. H2 * e2 = s2 and e2.weight = p
        for e2 in L:
            e1 = H1 * e2 - s1
            if (e1.weight <= w-p):
                e = (e1 || e2)
                permute_vector(e, inverse_permutation(pH.permutation))
                return e
```

## The Sub_ISD loop

- Prange
  - p = 0, ell = 0
  - returns the full-zero vector of length k
- Lee-Brickell
  - ell = 0
  - exhaustive list of all vectors of length k and weight p
- Stern/Dumer
  - birthday search, disjoint support
- MMT
  - without the disjoint support restriction
  - filtering to avoid multiple representations
- BJMM
  - use the "1+1=0" idea to have even more representations
