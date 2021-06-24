#ifndef MCCL_ALGORITHM_LB_HPP
#define MCCL_ALGORITHM_LB_HPP

#include <mccl/algorithm/decoding.hpp>

MCCL_BEGIN_NAMESPACE

class subISD_LB
    : public ISD_API_exhaustive_sparse_t
{   
public:
    using ISD_API_exhaustive_sparse_t::callback_t;

    void initialize(const mat_view& H_, const vec_view& S, unsigned int w_, callback_t _callback, void* _ptr) final
    {
        callback = _callback;
        ptr = _ptr;
        rowenum.reset(H_, p, 1);
        E1_sparse.resize(p);
    }

    void prepare_loop() final 
    {
        rowenum.reset(p, 1);
    }

    bool loop_next() final
    {
        rowenum.compute();

        // todo: optimize and pass computed error sum
        unsigned sz = rowenum.selectionsize();
        for(unsigned i = 0; i < sz; i++) E1_sparse[i] = rowenum.selection()[i];
        (*callback)(ptr, E1_sparse, 0);
        return rowenum.next();
    }

    void solve() final
    {
        prepare_loop();
        while (loop_next())
            ;
    }
private:
    callback_t callback;
    void* ptr;
    matrix_enumeraterows_t rowenum;
    static const size_t p = 3;
    std::vector<uint32_t> E1_sparse;
};

MCCL_END_NAMESPACE

#endif
