#ifndef MCCL_ALGORITHM_PRANGE_HPP
#define MCCL_ALGORITHM_PRANGE_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>

MCCL_BEGIN_NAMESPACE

class subISDT_prange
    : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    void configure(size_t)
    {
    }
    
    void initialize(const cmat_view&, size_t HTcolumns, const cvec_view&, unsigned int, callback_t _callback, void* _ptr) final
    {
        // should only be used with l=0
        if (HTcolumns != 0)
            throw std::runtime_error("subISDT_prange::initialize(): Prange doesn't support l>0");
        callback = _callback;
        ptr = _ptr;
    }
    
    void prepare_loop() final
    {
    }
    
    bool loop_next() final
    {
        (*callback)(ptr, nullptr, nullptr, 0);
        return false;
    }
    
    void solve() final
    {
        loop_next();
    }
    
private:
    callback_t callback;
    void* ptr;
};

MCCL_END_NAMESPACE

#endif
