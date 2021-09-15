# ISD

In MCCL, ISD attack implementations should derive from the `syndrome_decoding_API` (see [decoding.hpp](/mccl/algorithm/decoding.hpp)) and implement the following members functions:
- `void initialize(const cmat_view& H, const cvec_view& S, unsigned int w)`: obtain the H, S and w of the decoding problem
- `void prepare_loop(bool benchmark = false)`: prepare internal structures for the first main loop execution. If the option benchmark is set then found solutions should be ignored, so that the main loop can be benchmarked.
- `bool loop_next()`: execute one main loop iteration. return false to stop the main loop (a solution is found or search space is exhausted).
- `void solve()`: optional. calls `prepare_loop()` once and then `loop_next` repeatedly until either a solution is found or the main loop ends.
- `cvec_view get_solution()`: return the found solution or empty vector otherwise

In addition the following member functions are required to pass configuration and statistics:
- `decoding_statistics get_stats()`: return a statistics object (see [statistics.hpp](/mccl/tools/statistics.hpp))
- `void load_config(const configmap_t& configmap)`: read its configuration options from configmap, use default values for option values that are missing.
- `void save_config(configmap_t& configmap)`: write *all* its configuration options to configmap.

See [isdgeneric.hpp](/mccl/algorithm/isdgeneric.hpp) for an example implementation.

# isd_generic

ISD attacks can typically be described and implemented in a ISD_generic and subISD algorithms, where ISD_generic can be described by the following pseudocode:
```
vector ISD_generic(matrix H, vector S, int w, int ell)
{
    sol = vector()
    
    while sol.empty():
        pi = random_permutation( H.cols )
        PH = permute( H, pi )

        // perform row reduction and obtain submatrices H1, H2 and corresponding subsyndromes s1, s2
        ((I H1 s1),(0 H2 s2)) = row_reduce( (PH || S), H.rows - ell )
            // (I H1 s1) is a (H.rows-ell) x (H.columns+1) matrix
            // (0 H2 s2) is a ell x (H.columns+1) matrix
    
        // the callback checks for each e2 if it can be extended in a full solution. 
        // if a solution is found then this is stored in sol.
        callback = ( lamda: (e2) -> check_solution(H1,s1,pi,w,e2,sol) )
    
        subISD( H2, s2, callback )

    return sol
}
```

MCCL has an optimized implementation for ISD_generic such that most ISD attacks only need to implement the subISD part using `subISDT_API` described later on below.
SubISD's typtically work on H2^T such that it can work efficiently on individual H2-columns.
To avoid overhead due to transpositions, ISD_generic maintains PH in transposed form.

It also uses row reduction (as seen over non-transposed PH) that chooses pivots from last to first column, such that instead of `I` on the left hand side, a reversed identity submatrix appears on the right hand side.
The reason for this is that normal row reduction followed by transposition (i.e. `row_reduce(PH)^T`) results in a submatrix (H1^T || H2^T),
while this reversed row reduction followed by transposition (i.e. `revcol_row_reduce(PH)^T`) results in a submatrix (H2^T || H1^T).
This has two benefits:
- H2^T always starts at column 0, and thus can be passed as submatrix
- H1^T follows H2^T, so subISDs can let additional H1^T columns piggy back in the last word of H2^T and additionally prefilter on the weight of this part.

As is common, instead of using a completely new permutation each loop, 
ISD_generic can use one of several strategies to only perform a limited number of swaps `u` to randomize the permutation:
- type 1: `u` times choose an echelon & ISD column and swap
- type 2: choose `u` distinct echelon columns, swap with `u` random (non-distinct) ISD columns
- type 3: choose `u` distinct echelon columns, swap with `u` random distict ISD columns

Note that using type 3 with `u = B` where `B:=(n-k)*k/n` is essentially equivalent to using a completely random permutation each loop.
However, note that two swap iterations of type 3 with `u=B/2` is not equivalent to one swap iteration of type 3 with `u=B`.
We define a new strategy that improves this:
- type 4 (new): start by choosing a random sequence `EC` of `B` echelon columns, and a random sequence `IC` of `B` ISD columns. 
  In each iteration, consume the first `u` remaining elements of `EC` and `IC` and swap the respective columns.
  Everytime after `B` swaps, when `EC` and `IC` are empty, choose new sequences `EC` and `IC` of `B` columns.

In addition we define additional new strategies where echelon columns are not chosen at random, but are chosen in round robin fashion (each swap iteration picks up where the previous swap iteration ended):
- type 12: like type 2, but round-robin echelon column selection
- type 13: like type 3, but round-robin echelon column selection
- type 14: like type 4, but round-robin echelon column selection
- type 10: use round-robin echelon column selection, but perform swaps with an ISD column just-in-time during row reduce. To-be-swapped ISD columns will be chosen by considering them in round-robin fashion and picking the first with the correct pivot bit having value `1`.

In initial experiments for random matrices and rate=1/2, type 10 seems to perform best among all strategies. This may not be so for other cases.

# subISD

subISD attack implementations should derive from the `subISDT_API` (see [decoding.hpp](/mccl/algorithm/decoding.hpp)) and implement the following members functions:
- `void initialize(const cmat_view& H2TH1T, size_t ell, const cvec_view& S2S1, unsigned int w, callback_t callback, void* ptr = nullptr)`: obtain the subISD problem `H2TH1T`, `ell` `S2S1`, `w` as well as the callback function `callback` together with a ISD_generic internal pointer `ptr` that has to be supplied with each call to `callback`.
- `void prepare_loop()`: prepare internal structures for the first main loop execution.
- `bool loop_next()`: execute one main loop iteration. return `false` to stop the main loop (indicated by `callback` or subISD search space is exhausted).
- `void solve()`: optional. calls `prepare_loop()` once and then `loop_next` repeatedly until it returns `false`.

Note that `callback_t` is the function pointer type `bool (*callback_t)(void*, const uint32_t*, const uint32_t*, unsigned int)`, if it returns `false` then a solution has been found and subISD should stop the main loop.

Furthermore, ISD_generic passes the matrix `(H2^T || H1^T)`, subISD only need to consider the first `ell` columns that corresponds to `H2^T`.
Obtaining the `H2T` submatrix view can be easily done as follows: `auto H2T = H2TH1T.submatrix(0, H2TH1T.rows(), 0, ell);`

In addition the following member functions are required to pass configuration and statistics:
- `decoding_statistics get_stats()`: return a statistics object (see [statistics.hpp](/mccl/tools/statistics.hpp))
- `void load_config(const configmap_t& configmap)`: read its configuration options from configmap, use default values for option values that are missing.
- `void save_config(configmap_t& configmap)`: write *all* its configuration options to configmap.

See [prange.hpp](/mccl/algorithm/prange.hpp) for an example implementation.
