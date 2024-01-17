# Algorithms

## Objects

- `matrix`
- `vector`
- `permutation`
- `permuted_matrix`: this objects corresponds to a matrix and a permutation. The permutation keeps track of the column permutations that were applied to the matrix. A constructor takes a matrix as input and puts the identity permutation as default.

## Functions

- `gaussian_elimination`
  - Input: matrix H, vector s, int c
  - Output: null
  - This operation takes a matrix and performs Gaussian elimination so as to make the top-left c x c submatrix the identity matrix, and the bottom-left (n-k-c) x c submatrix the all-zero matrix. The same row-operations are applied to the vector s (viewed as a (n-k) x 1 vector). All of these operation are applied in place.
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
- `decompose`
  - Input: a x b matrix H, vector s of length a, integer c
  - Output:
    - H1 a (b-a-c') x (a-c') matrix
    - H2 a (b-a-c') x c' matrix
    - s1 a vector of length a-c'
    - s2 a vector of length c'
  - c' denotes the smallest multiple of 64 such that c' >= c
  - H1 is the transpose of the top-right (a-c')x(b-a-c') submatrix of H
  - H2 is the reverse-transpose of the bottom-right c'x(b-a-c') submatrix of H (by "reverse-transpose" we mean the transpose matrix written in reverse order, i.e. each row read from left to right becomes a column read from bottom to top)
  - s1 is the first (a-c') bits of s
  - s2 is the last c' bits of s in reverse order
  - Comments:
    - Computing H1 and s1 could be performed only when a solution is found?
- `random_vector`
  - Input: none
  - Output: random int(64) vector  
- `build_LR_lists`
  - Input:
    - matrix H
    - int p the total weight (each list should be of weight p/2)
  - Output: two lists L1,L2 of elements of the form (v,e) such that
    - v = H*e.transpose()
    - S1 and S2 form a random partition of the matrix columns
    - for L1,
      - Support(e) $\subseteq$ S1,
      - weight(e) = p//2
    - for L2,
      - Support(e) $\subseteq$ S2,
      - weight(e) = p-p//2
  - Comments:
    - should the set S1 be specified as a parameter?
    - in case p is odd, it could be better to break the symmetry of S1 and S2 to obtain lists of the same size.
- `merge_lists`
  - Input:
    - a list L1 of couples (v,e)
    - a list L2 of couples (v,e)
    - a target t (the target)
    - an integer r (the number of bits on which the sum should agree with the target)
    - an integer p (the weight)
  - Output: a list $L = \{(v=v1+v2, e=e1+e2) | (vi,ei) \in Li, v[:r]=t[:r], weight(e)=p \}$.
  - Comments:
    - could be worth to group the creation of base lists and the first merge in a single function
    - when we already know that the sums will agree on r2 bits we only need to check the condition for r-r2 bits, could this be taken into account?
    - merging can be done either by ordering or using hash tables. Landais finds hashtables more efficients. 
    - Landais notes that not handling collisions in the hashtable can be more efficient. 

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
        H1, H2, s1, s2 = decompose(pH.matrix,ell) // Useless to compute H1 here?
        callback = lambda (e2,w2) -> check_solution(H1,s1,perm,w,e2,w2)
        Sub_ISD(H2,s2,p,w2max,callback)
```

``` cpp
vector check_solution(matrix H1, vector s1, permutation perm, int w, vector e2, int w2):
    vector e1 = H1 * e2 - s1
    if (e1.weight <= w-w2-p):
        vector e = (e1 || e2)
        permute_vector(e, inverse_permutation(perm))
        return e
```

## The Sub_ISD loop

```cpp
  vector Sub_ISD(matrix H2, vector s2, int p, int ell, int w2max, vector (...) callback)
```

Let ell0 denote the smallest integer such that ell+ell0 is a multiple of 64.

This function finds vectors e2 of length k+ell+ell0 s.t.
  - e2[:k+ell] * H2[:ell] = s2[:ell]
  - e2[:k+ell].weight <= p
  - e2[k+ell:].weight <= w2max.

It calls the callback function on `(e2,w2)` where w2 is e2[ell:].weight.

### Prange
- p = 0, ell = 0
- returns the full-zero vector of length k

``` cpp
vector Sub_ISD_Prange(vector (...) callback):
    vector e2 = zero_vector(k)
    callback (e2, 0)
```

Note: it does not make sense to use a Sub_ISD loop in this case.

### Lee-Brickell
- ell = 0
- exhaustive list of all vectors of length k and weight p

``` cpp
vector Sub_ISD_LB(matrix H2, vector s2, int p, vector (...) callback):
    // for all vectors e2 of weight p:
	   callback(e2, 0)
```

### Leon
- ell > 0
- exhaustive list of all vectors of length k and weight p

``` cpp
vector Sub_ISD_Leon(matrix H2, vector s2, int p, int ell, int w2max, vector (...) callback):
    // for all vectors e2 of length k+ell and weight p:
		vector v = e2*H2 - s2
		if v[:ell] == zero_vector(ell):
			int w2 = v[ell:].weight
			if w2 <= w2max: // pre-filtering
				callback((vector_reverse(v[ell:]),e2),w2)
	return L
```

### Stern/Dumer
``` cpp
vector Sub_ISD_Dumer(matrix H2, vector s2, int p, int ell, int w2max, vector (...) callback):
	int L1 = (k+ell)/2
	int L2 = k+ell-L1
	int p1 = p/2
	int p2 = p-p1
	matrix H21 = H2[:L1]
	matrix H22 = H2[L1:]
	// Let H be a hash-table
	// for all vectors e1 of length L1 and weight p1:
		vector v1 = e1*H21
		H[v1[:ell]].append(e1,v1[ell:])
	// for all vectors e2 of length L2 and weight p2:
		vector v2 = e2*H22
		for (e1,v11) in H[v2[:ell]]:
			v = v11 + v2[ell:]
			w2 = v.weight
			if w2 <= w2max: // pre-filtering
				callback((vector_reverse(v),e1,e2),w2)
```
- birthday search, disjoint support

### MMT
- without the disjoint support restriction
- filtering to avoid multiple representations

```cpp
vector Sub_ISD_MMT(matrix H2, vector s2, int p, int ell, int r1, int w2max, vector (...) callback):
  // We suppose that s2 and H2 are of length 64.

  // initiate lists
  List[(vector,List(int))] Li;

  // random target vectors
  vector  t = random_vector(64);

  int p1 = p//2;
  int p2 = p - p1;

  L21, L22 = build_LR_lists(H2,p1);
  L11 = merge_lists(L21,L22,t,r1,p1);

  L23, L24 = build_LR_lists(H2,p2);
  L12 = merge_lists(L23,L24,t+s2,r1,p2);

  L0  = merge_lists(L11,L12,s,l,p);

  for (v,e) in L0:
    if (v[l:]).weight() <= w2max: // pre-filtering
      callback((vector_reverse(v),e),w2)
```

### BJMM
- use the "1+1=0" idea to have even more representations

```cpp
vector Sub_ISD_BJMM(matrix H2, vector s2, int p, int p1, int p2, int ell, int r1, int r2, int w2max, vector (...) callback):
  // We suppose that s2 and H2 are of length 64.

  // initiate lists
  List[(vector,List(int))] Li;

  // random target vectors
  vector t21 = random_vector(64);
  vector t22 = random_vector(64);
  vector  t1 = random_vector(64);

  L311, L312 = build_LR_lists(H2,p2);
  L21 = merge_lists(L311,L312,t21,r2,p2);

  L321, L322 = build_LR_lists(H2,p2);
  L22 = merge_lists(L321,L322,t1+t21,r2,p2);

  L331, L332 = build_LR_lists(H2,p2);
  L23 = merge_lists(L331,L332,t22,r2,p2);

  L341, L342 = build_LR_lists(H2,p2);
  L22 = merge_lists(L341,L342,t1+t22+s2,r2,p2);

  L11 = merge_lists(L21,L22,t1,r1,p1);
  L12 = merge_lists(L23,L24,t1+s,r1,p1);
  L0  = merge_lists(L11,L12,s,l,p);

  for (v,e) in L0:
    if (v[l:]).weight() <= w2max: // pre-filtering
      callback((vector_reverse(v),e),w2)
```

### Sieving
``` cpp
// solves an instance of Sub_ISD and returns a list of potential solutions
list Sub_ISD_sieving(matrix H, vector s, int w, int N):
	
	// sampling N random vectors of weight w
	L_ini = list()
	for i in 0..N-1:
		e.sample(w)
		L_ini.append(e)
		
	// sieving
	L_out = list()
	for i in 0..n-k-1:

		// check if any of the previously sampled e satisfy the first i constraints
		for e in L_ini:
			if H[:i] * e = s[:i]:
				L_out.append(e)

		// near neighbour search
		Centers = Sample_Filters() // version dependent (GJN, Hash, RPC)
		Pairs = Near_Neighbour_Search(L_ini, Centers, alpha, w)

		// check if any of the summed vectors from NNS satisfy the first i constraints
		for (e1, e2) in Pairs:
			if H[:i] * (e1 + e2) = s[:i]:
				L_out.append(e1 + e2)

		L_ini.copy(L')
		L_out.empty()
		
	return L_out // Should be modified in order to just check the solution not to return a list.
```

``` cpp
// solves an instance of the near neighbour search problem and return a list of solutions pairs	
list Near_Neighbour_Search(list L, list Centers, int alpha):
	
	// initialize buckets
	Buckets = map()
	for c in Centers:
		Buckets.key.add(c)
	
	// bucketing
	for x in L:
		for c in Valid_Filters(Centers, x, alpha):
			Buckets[c].value.append(x)
				
	// checking
	Pairs = list()
	for x in L:
		for for c in Valid_Filters(Centers, x, alpha):
			for y in Buckets[c].value:
				if x & y = w/2:
					Pairs.append(x, y)
					
	// return
	return Pairs
```

``` cpp
// returns valid filters correspoding to an element from a list
list Valid_Filters(list Centers, vector x, int alpha):
	
	L_out.list()
	for c in Centers:
		if x & c == alpha:
			L_out.append(c)
			
	return L_out
```

