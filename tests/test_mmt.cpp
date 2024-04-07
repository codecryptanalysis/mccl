#include <mccl/config/config.hpp>

#include <mccl/tools/parser.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/algorithm/mmt.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <vector>

using namespace mccl;

int main(int, char**)
{
    int status = 0;

    file_parser parse;
    status |= !parse.parse_file("./tests/data/SD_100_0");

    auto Hraw = parse.H();
    auto S = parse.S();
    size_t n = parse.n();
    size_t k = parse.k();
    size_t w = parse.w();

    std::vector<size_t> rowweights(n-k);
    for( size_t r = 0; r < n-k; r++)
        rowweights[r] = hammingweight(Hraw[r]);
//    auto total_hw = hammingweight(Hraw);

    configmap_t configmap = { {"p", "4"}, {"l", "14"} };
    {
        subISDT_mmt mmt;
        ISD_generic<subISDT_mmt> ISD_mmt(mmt);
        
        ISD_mmt.load_config(configmap);
        mmt.load_config(configmap);
        
        ISD_mmt.initialize(Hraw, S, w);
        ISD_mmt.solve();
        status |= not(hammingweight(ISD_mmt.get_solution()) <= w);
        std::cerr << hammingweight(ISD_mmt.get_solution()) << std::endl;
        vec eval_S(Hraw.rows());
        vec r(Hraw.columns());
        for(size_t i = 0; i < Hraw.rows(); i++ ) 
        {
            bool x = hammingweight(r.v_and(Hraw[i],ISD_mmt.get_solution()))%2;
            if(x)
                eval_S.setbit(i);
        }
        status |= not(eval_S.is_equal(S));
    }

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
