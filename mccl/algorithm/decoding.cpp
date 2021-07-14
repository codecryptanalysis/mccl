#include <mccl/algorithm/decoding.hpp>

MCCL_BEGIN_NAMESPACE

bool check_SD_solution(const cmat_view& H, const cvec_view& S, unsigned int w, const cvec_view& E)
{
    if (S.columns() != H.rows())
        throw std::runtime_error("check_SD_solution(): H and S do not have matching dimensions");
    if (E.columns() != H.columns())
        throw std::runtime_error("check_SD_solution(): H and E do not have matching dimensions");
    // first check if weight of E is less or equal to w
    if (hammingweight(E) > w)
        return false;
    // then check if columns of H marked by E sum up to S
    vec tmp(S.columns());
    for (size_t i = 0; i < H.rows(); ++i)
        if (hammingweight_and(H[i], E) % 2)
            tmp.setbit(i);
    return tmp == S;
}

bool syndrome_decoding_problem::check_solution(const cvec_view& E) const
{
    return check_SD_solution(H, S, w, E);
}

MCCL_END_NAMESPACE
