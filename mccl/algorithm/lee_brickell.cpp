#include <mccl/algorithm/lee_brickell.hpp>

MCCL_BEGIN_NAMESPACE

lee_brickell_config_t lee_brickell_config_default;

vec solve_SD_lee_brickell(const cmat_view& H, const cvec_view& S, unsigned int w)
{
    subISDT_lee_brickell subISDT;
    ISD_lee_brickell<> ISD(subISDT);
    
    return solve_SD(ISD, H, S, w);
}

vec solve_SD_lee_brickell(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap)
{
    subISDT_lee_brickell subISDT;
    ISD_lee_brickell<> ISD(subISDT);
    
    subISDT.load_config(configmap);
    ISD.load_config(configmap);
    
    return solve_SD(ISD, H, S, w);
}

MCCL_END_NAMESPACE
