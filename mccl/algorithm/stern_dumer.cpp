#include <mccl/algorithm/stern_dumer.hpp>

MCCL_BEGIN_NAMESPACE

stern_dumer_config_t stern_dumer_config_default;

vec solve_SD_stern_dumer(const cmat_view& H, const cvec_view& S, unsigned int w)
{
    subISDT_stern_dumer subISDT;
    ISD_stern_dumer<> ISD(subISDT);
    
    return solve_SD(ISD, H, S, w);
}

vec solve_SD_stern_dumer(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap)
{
    subISDT_stern_dumer subISDT;
    ISD_stern_dumer<> ISD(subISDT);
    
    subISDT.load_config(configmap);
    ISD.load_config(configmap);
    
    return solve_SD(ISD, H, S, w);
}

MCCL_END_NAMESPACE
