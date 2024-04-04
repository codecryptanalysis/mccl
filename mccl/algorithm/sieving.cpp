#include <mccl/algorithm/sieving.hpp>

MCCL_BEGIN_NAMESPACE

sieving_config_t sieving_config_default;

vec solve_SD_sieving(const cmat_view& H, const cvec_view& S, unsigned int w)
{
    subISDT_sieving subISDT;
    ISD_sieving<> ISD(subISDT);

    return solve_SD(ISD, H, S, w);
}

vec solve_SD_sieving(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap)
{
    subISDT_sieving subISDT;
    ISD_sieving<> ISD(subISDT);

    subISDT.load_config(configmap);
    ISD.load_config(configmap);

    return solve_SD(ISD, H, S, w);
}

MCCL_END_NAMESPACE