#include <mccl/config/config.hpp>

#include <mccl/tools/parser.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/algorithm/prange.hpp>
#include <mccl/algorithm/lee_brickell.hpp>

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
    {
        subISDT_prange prange;
        ISD_generic<subISDT_prange> ISD_prange(prange);
        ISD_prange.initialize(Hraw, S, w);
        ISD_prange.solve();
        status |= not(hammingweight(ISD_prange.get_solution()) <= w);
        std::cerr << hammingweight(ISD_prange.get_solution()) << std::endl;
        vec eval_S(Hraw.rows());
        vec r(Hraw.columns());
        for(size_t i = 0; i < Hraw.rows(); i++ ) 
        {
            bool x = hammingweight(r.vand(Hraw[i],ISD_prange.get_solution()))%2;
            if(x)
                eval_S.setbit(i);
        }
        status |= not(eval_S==S);
    }

    // test subISD_LB
    {
        subISDT_lee_brickell subLB;
        ISD_generic<subISDT_lee_brickell> ISD_single_LB(subLB);
        ISD_single_LB.initialize(Hraw, S, w);
        ISD_single_LB.solve();
        status |= not(hammingweight(ISD_single_LB.get_solution()) <= w);
        std::cerr << hammingweight(ISD_single_LB.get_solution()) << std::endl;
        vec eval_S(Hraw.rows());
        vec r(Hraw.columns());
        for(size_t i = 0; i < Hraw.rows(); i++ ) {
            bool x = hammingweight(r.vand(Hraw[i],ISD_single_LB.get_solution()))%2;
            if(x)
                eval_S.setbit(i);
        }
        status |= not(eval_S==S);
    }

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
