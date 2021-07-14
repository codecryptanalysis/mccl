#include <mccl/algorithm/prange.hpp>

MCCL_BEGIN_NAMESPACE

vec solve_SD_prange(const cmat_view& H, const cvec_view& S, unsigned int w)
{
    subISDT_prange subISDT;
    ISD_prange<> ISD(subISDT);

    return solve_SD(ISD, H, S, w);
}

vec solve_SD_prange(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap)
{
    subISDT_prange subISDT;
    ISD_prange<> ISD(subISDT);

    subISDT.load_config(configmap);
    ISD.load_config(configmap);

    return solve_SD(ISD, H, S, w);
}

MCCL_END_NAMESPACE
