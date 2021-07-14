#include <mccl/algorithm/isdgeneric.hpp>

MCCL_BEGIN_NAMESPACE

template class ISD_generic<subISDT_API,64>;
template class ISD_generic<subISDT_API,128>;
template class ISD_generic<subISDT_API,256>;
template class ISD_generic<subISDT_API,512>;

ISD_generic_config_t ISD_generic_config_default;

MCCL_END_NAMESPACE
