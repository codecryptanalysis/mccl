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

int main(int, char**)
{
    int status = 0;

    Parser parse;
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
//    auto total_hw = hammingweight(Hraw);

    // test subISD_prange
    subISD_prange<> prange;
    ISD_single_generic<subISD_prange<>> ISD_single_prange(prange);
    ISD_single_prange.initialize(Hraw, S, w);
    try 
    {
        ISD_single_prange.solve();
    } catch(Solution& sol)
    {
        status |= not(hammingweight(sol.get_solution()) <= w);
        std::cerr << hammingweight(sol.get_solution()) << std::endl;
        vec eval_S(Hraw.rows());
        vec r(Hraw.columns());
        for(size_t i = 0; i < Hraw.rows(); i++ ) 
        {
            bool x = hammingweight(r.vand(Hraw[i],sol.get_solution()))%2;
            if(x)
                eval_S.setbit(i);
        }
        status |= not(eval_S==S);
    }

    // test subISD_LB
    subISD_LB<> subLB;
    ISD_single_generic<subISD_LB<>> ISD_single_LB(subLB);
    ISD_single_LB.initialize(Hraw, S, w);
    try 
    {
        ISD_single_LB.solve();
    } catch(Solution& sol) 
    {
        status |= not(hammingweight(sol.get_solution()) <= w);
        std::cerr << hammingweight(sol.get_solution()) << std::endl;
        vec eval_S(Hraw.rows());
        vec r(Hraw.columns());
        for(size_t i = 0; i < Hraw.rows(); i++ ) {
            bool x = hammingweight(r.vand(Hraw[i],sol.get_solution()))%2;
            if(x)
                eval_S.setbit(i);
        }
        status |= not(eval_S==S);
    }

    // test decoding API
    LB<> decoder;
    decoder.initialize(Hraw, S, w, nullptr, nullptr);
    decoder.solve();

    // prepare H for ISD, split into two parts with appropriate column padding
    size_t rows = n-k;
    size_t cols0 = n-k-ell;
    size_t scratch0 = get_scratch(cols0, 64);
    size_t cols1 = k+ell;
    mat H(n-k, cols0+scratch0+cols1+1); // +1 to store S
    for( size_t i = 0; i < rows; i++ ) {
        for( size_t j = 0; j < cols0; j++ ) {
            H.setbit(i,j+scratch0,Hraw(i,j));
        }
        for( size_t j = 0; j < cols1; j++ ) {
            H.setbit(i, cols0+scratch0+j, Hraw(i, cols0+j));
        }
        H.setbit(i, cols0+scratch0+cols1, S[i]);
    }

    size_t ell0 = get_scratch(ell, 64);
    size_t rows1 = std::min(ell+ell0, rows);
    size_t rows0 = rows-rows1;

//    size_t scratch1 = get_scratch(cols1+1, 64);
    mat_view H01_S_view(H.submatrix(0, rows0, cols0+scratch0, cols1+1));
    mat_view H11_S_view(H.submatrix(rows0, rows1, cols0+scratch0, cols1+1));

//    size_t scratch11T = get_scratch(rows1, 64);
    mat H11T(k+ell+1, ell+ell0);
    mat_view H11T_view(H11T.submatrix(0, k+ell, 0, rows1));
    mat_view H11T_S_view(H11T.submatrix(0, k+ell+1, 0, rows1));

    mat H01T(cols1+1, rows0);
//    size_t scratch01T = get_scratch(rows0, 64);
    mat_view H01T_S_view(H01T.submatrix(0, cols1+1, 0, rows0));
    mat_view H01T_view(H01T.submatrix(0, cols1, 0, rows0));

    matrix_permute_t permutator(H);
    size_t cnt = 0;
    while(true) {
        cnt++;

        permutator.random_permute(scratch0, scratch0+cols0, scratch0, scratch0+n);
        auto pivotend = echelonize(H, scratch0, scratch0+cols0);
        if(pivotend != cols0)
            continue;

        H01T_S_view.transpose(H01_S_view);
        H11T_S_view.transpose(H11_S_view);

        auto S0 = H01T_S_view[cols1];
//        auto S1 = H11T_S_view[k+ell];

        // for prange we only have to check the weight of S0 as S1 is empty
        if(hammingweight(S0) <= w) {
            auto perm = permutator.get_permutation();
            vec sol(n);
            for( size_t i = 0; i < n-k; i++ ) {
                if (S0[i])
                    sol.setbit(perm[scratch0+i]-scratch0);
            }
            std::cerr << "Found solution after " << cnt << " iterations." << std::endl;
            std::cerr << S0 << std::endl;
    	    std::cerr << sol << std::endl;

            // check solution
            status |= not(hammingweight(sol) <= w);
            std::cerr << hammingweight(sol) << std::endl;
            vec eval_S(Hraw.rows());
            for(size_t i = 0; i < Hraw.rows(); i++ ) {
                bool x = hammingweight_and(Hraw[i],sol)%2;
                if(x)
                    eval_S.setbit(i);
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
