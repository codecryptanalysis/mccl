#ifndef MCCL_ALGORITHM_PRANGE_HPP
#define MCCL_ALGORITHM_PRANGE_HPP

#include <mccl/algorithm/decoding.hpp>

MCCL_BEGIN_NAMESPACE

class subISD_prange
    : public ISD_API_exhaustive_sparse_t
{   
public:        
    using ISD_API_exhaustive_sparse_t::callback_t;

    void initialize(const mat_view& H_, const vec_view& S, unsigned int w_, callback_t _callback, void* _ptr)
    {
        callback = _callback;
        ptr = _ptr;
    }

    bool loop_next()
    {
        (*callback)(ptr,E1_sparse, 0);
        return false;
    }
private:
    callback_t callback;
    void* ptr;
    std::vector<uint32_t> E1_sparse;
};

MCCL_END_NAMESPACE

#endif
