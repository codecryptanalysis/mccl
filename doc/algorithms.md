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
  - Input: permuted_matrix pH
  - Output: null
  - Same as permute_matrix but with a random permutation.
- `transpose`
	- Input: a x b matrix
	- Output: b x a matrix
	- Transposes the matrix

## ISD generic algorithm

``` cpp
vector ISD_generic(matrix H, vector s, int w, int p, int ell):
    int n = H.dimensions[0]
    int r = H.dimensions[1]
    int k = n-r
    permuted_matrix pH = permuted_matrix(H)
    while(true)
        random_permute_matrix(pH)
        gaussian_elimination(pH.matrix, s, n-k-ell)
		// pH is now of the form
		//         -------------------
  	      	//       ||     Id    |   *   || (n-k-ell) 
	       	//  pH=  ||     0     |   *   || (ell)
		//         -------------------
        	//          (n-k-ell)    (k+ell)
		// We will denote e1 e2 such that 
        	//  e=   ||     e1    |   e2   ||
		// Let Htranspose be the transposition of the right block matrix of pH
		Htranspose = transpose(pH.matrix.submatrix([:],[n-k-ell:]))
		// Let ell2 be the smallest integer such that 
		// ell+ell2 is a multiple of 64
		int ell2 = ell / (int) 64 * 64 - ell
		// Define H1 H2 the submatrices of Ht such that
		//          ----------------------
	        //  Ht =  ||     H1      |   H2   || (k+ell)
		//          ----------------------
		//         (n-k-ell-ell2) (ell+ell2)
		// and similarly
	        //  s =   ||     s1      |   s2   ||
		L = Sub_ISD(H2, s2, p)
	        // Returns a set of vectors e2 of length k+ell+ell2 s.t.
        	// 		1. e2[ell2:] * H2[ell2:] = s2[ell2:]   
		// 		2. e2[ell2:].weight <= p
		//		3. e2[:ell2] = e2[ell2:]*H2[:ell2] - s2[:ell2]
		//		3. w2 < w2max,
		//			where w2 = e2[:ell2].weight
        for (e2,w2) in L:
            e1 = H1 * e2 - s1
            if (e1.weight <= w-w2-p):
                e = (e1 || e2)
                permute_vector(e, inverse_permutation(pH.permutation))
                return e
```

## The Sub_ISD loop

### Prange
- p = 0, ell = 0
- returns the full-zero vector of length k

``` cpp
List<vector> Sub_ISD_Prange():
    vector e2 = zero_vector(k)
	L.insert((e2,0))
	return L
```

Note: it does not make sense to use a Sub_ISD loop in this case.

### Lee-Brickell
- ell = 0
- exhaustive list of all vectors of length k and weight p

``` cpp
List<vector> Sub_ISD_LB(matrix H2, vector s2, int p):
    // for all vectors e2 of weight p:
		L.insert((e2,w2))
	return L
```

### Leon
- ell > 0
- exhaustive list of all vectors of length k and weight p

``` cpp
List<vector> Sub_ISD_Leon(matrix H2, vector s2, int p, int ell, int w2max):
    // for all vectors e2 of length k+ell and weight p:
		vector v = e2*H2 - s2
		if v[ell2:] == zero_vector(ell):
			w2 = v[:ell2].weight
			if w2 < w2max: // pre-filtering
				L.insert((v[:ell2],e2),w2)
	return L
```

### Stern/Dumer
``` cpp
List<vector> Sub_ISD_Dumer(matrix H2, vector s2, int p, int ell, int w2max):
	int L1 = (k+ell)/2
	int L2 = k+ell-L1
	int p1 = p/2
	int p2 = p-p1
	matrix H21 = H2[:L1]
	matrix H22 = H2[L1:]
	// Let H be a hash-table
	// for all vectors e1 of length L1 and weight p1:
		vector v1 = e1*H21
		H[v1[ell2:]].append(e1,v1[ell2:])
	// for all vectors e2 of length L2 and weight p2:
		vector v2 = e2*H22
		for (e1,v11) in H[v2[ell2:]]:
			v = v11 + v2[:ell2]
			w2 = v.weight
			if w2 < w2max: // pre-filtering
				L.insert((v,e1,e2),w2)
	return L
```
- birthday search, disjoint support

### MMT
- without the disjoint support restriction
- filtering to avoid multiple representations

### BJMM
- use the "1+1=0" idea to have even more representations

