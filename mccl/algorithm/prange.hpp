#ifndef MCCL_ALGORITHM_PRANGE_HPP
#define MCCL_ALGORITHM_PRANGE_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>

MCCL_BEGIN_NAMESPACE

class subISDT_prange
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    // API member function
    ~subISDT_prange() final
    {
    }
    
    void configure(size_t)
    {
    }

    // API member function
    void initialize(const cmat_view&, size_t HTcolumns, const cvec_view&, unsigned int, callback_t _callback, void* _ptr) final
    {
        // should only be used with l=0
        if (HTcolumns != 0)
            throw std::runtime_error("subISDT_prange::initialize(): Prange doesn't support l>0");
        callback = _callback;
        ptr = _ptr;
    }
    
    // API member function
    void prepare_loop() final
    {
    }
    
    // API member function
    bool loop_next() final
    {
        (*callback)(ptr, nullptr, nullptr, 0);
        return false;
    }
    
    // API member function
    void solve() final
    {
        loop_next();
    }
    
private:
    callback_t callback;
    void* ptr;
};

template<size_t _bit_alignment = 64>
using ISD_prange = ISD_generic<subISDT_prange,_bit_alignment>;

// TODO: provide further Prange convience functions

MCCL_END_NAMESPACE

#endif
