#include <mccl/algorithm/mmt.hpp>

MCCL_BEGIN_NAMESPACE

mmt_config_t mmt_config_default;

vec solve_SD_mmt(const cmat_view& H, const cvec_view& S, unsigned int w)
{
    subISDT_mmt subISDT;
    ISD_mmt<> ISD(subISDT);
    
    return solve_SD(ISD, H, S, w);
}

vec solve_SD_mmt(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap)
{
    subISDT_mmt subISDT;
    ISD_mmt<> ISD(subISDT);
    
    subISDT.load_config(configmap);
    ISD.load_config(configmap);
    
    return solve_SD(ISD, H, S, w);
}

MCCL_END_NAMESPACE
