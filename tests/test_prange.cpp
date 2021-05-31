#include <mccl/config/config.hpp>

#include <mccl/tools/parser.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/core/matrix_permute.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <utility>

using namespace mccl;

typedef mccl::matrix_ref_t<uint64_t> matrixref;
typedef mccl::matrix_ptr_t<uint64_t> matrixptr;
typedef mccl::vector_ref_t<uint64_t> vectorref;
typedef mccl::vector_ptr_t<uint64_t> vectorptr;
typedef mccl::matrix_t<uint64_t> mat_t;
typedef mccl::vector_t<uint64_t> vec_t;
typedef mccl::detail::matrix_base_ref_t<uint64_t> base_t;

int main(int, char**)
{
    int status = 0;

    Parser<uint64_t> parse;
    status |= !parse.load_file("./tests/data/SD_100_0");

    auto Hraw = parse.get_H();
    auto S = parse.get_S();
    size_t n = parse.get_n();
    size_t k = parse.get_k();
    size_t w = parse.get_w();
    size_t ell = 0;

    std::vector<size_t> rowweights(n-k);
    for( size_t r = 0; r < n-k; r++)
        rowweights[r] = hammingweight(Hraw[r]);
    auto total_hw = hammingweight(Hraw);

    // test subISD_prange
    subISD_prange<uint64_t> prange;
    ISD_single_generic<uint64_t,subISD_prange<uint64_t>> ISD_single(prange);
    ISD_single.initialize(Hraw, S, w);
    try {
        ISD_single.solve();
    } catch(Solution<uint64_t>& sol) {
        status |= not(hammingweight(sol.get_solution()) <= w);
        std::cerr << hammingweight(sol.get_solution()) << std::endl;
        vec_t eval_S(Hraw.rows());
        vec_t r(Hraw.columns());
        for(size_t i = 0; i < Hraw.rows(); i++ ) {
            bool x = hammingweight(r.op_and(Hraw[i],sol.get_solution()))%2;
            if(x)
                eval_S.bitset(i);
        }
        status |= not(eval_S==S);
    }

    // test decoding API
    LB<uint64_t> decoder;
    std::function<bool(vector_ref_t<uint64_t>&)> callback = nullptr;
    decoder.initialize(Hraw, S, w, callback);
    decoder.solve();

    // prepare H for ISD, split into two parts with appropriate column padding
    size_t rows = n-k;
    size_t cols0 = n-k-ell;
    size_t scratch0 = get_scratch(cols0, 64);
    size_t cols1 = k+ell;
    mat_t H(n-k, cols0+scratch0+cols1+1); // +1 to store S
    for( size_t i = 0; i < rows; i++ ) {
        for( size_t j = 0; j < cols0; j++ ) {
            H.bitset(i,j+scratch0,Hraw(i,j));
        }
        for( size_t j = 0; j < cols1; j++ ) {
            H.bitset(i, cols0+scratch0+j, Hraw(i, cols0+j));
        }
        H.bitset(i, cols0+scratch0+cols1, S[i]);
    }

    size_t ell0 = get_scratch(ell, 64);
    size_t rows1 = std::min(ell+ell0, rows);
    size_t rows0 = rows-rows1;

    size_t scratch1 = get_scratch(cols1+1, 64);
    matrixref H01_S_view(H.submatrix(0, rows0, cols0+scratch0, cols1+1, scratch1));
    matrixref H11_S_view(H.submatrix(rows0, rows1, cols0+scratch0, cols1+1, scratch1));

    size_t scratch11T = get_scratch(rows1, 64);
    mat_t H11T(k+ell+1, ell+ell0);
    matrixref H11T_view(H11T.submatrix(0, k+ell, 0, rows1, scratch11T));
    matrixref H11T_S_view(H11T.submatrix(0, k+ell+1, 0, rows1, scratch11T));

    mat_t H01T(cols1+1, rows0);
    size_t scratch01T = get_scratch(rows0, 64);
    matrixref H01T_S_view(H01T.submatrix(0, cols1+1, 0, rows0, scratch01T));
    matrixref H01T_view(H01T.submatrix(0, cols1, 0, rows0, scratch01T));

    matrix_permute_t<uint64_t> permutator(H);
    size_t cnt = 0;
    while(true) {
        cnt++;

        permutator.random_permute(scratch0, scratch0+cols0, scratch0+n);
        auto pivotend = echelonize(H, scratch0, scratch0+cols0);
        if(pivotend != cols0)
            continue;

        H01T_S_view.transpose(H01_S_view);
        H11T_S_view.transpose(H11_S_view);

        auto S0 = H01T_S_view[cols1];
        auto S1 = H11T_S_view[k+ell];

        // for prange we only have to check the weight of S0 as S1 is empty
        if(hammingweight(S0) <= w) {
            auto perm = permutator.get_permutation();
            mccl::vector_t<uint64_t> sol(n);
            for( size_t i = 0; i < n-k; i++ ) {
                if (S0[i])
                    sol.bitset(perm[scratch0+i]-scratch0);
            }
            std::cerr << "Found solution after " << cnt << " iterations." << std::endl;
            std::cerr << S0 << std::endl;
    	    std::cerr << sol << std::endl;

            // check solution
            status |= not(hammingweight(sol) <= w);
            std::cerr << hammingweight(sol) << std::endl;
            vec_t eval_S(Hraw.rows());
            for(size_t i = 0; i < Hraw.rows(); i++ ) {
                bool x = hammingweight(Hraw[i].op_and(sol))%2;
                if(x)
                    eval_S.bitset(i);
            }
            status |= not(eval_S==S);
            break;
        }
    }
    
    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
