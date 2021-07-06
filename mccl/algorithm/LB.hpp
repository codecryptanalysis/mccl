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
        if (sz != E1_sparse.size())
            E1_sparse.resize(sz);
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


class subISDT_LB
    : public ISD_API_exhaustive_transposed_sparserange_t
{
public:
    using ISD_API_exhaustive_transposed_sparserange_t::callback_t;
    
    void configure(size_t _p = 3)
    {
        p = _p;
    }
    
    void initialize(const cmat_view& HTpadded, size_t HTcolumns, const cvec_view& Spadded, unsigned int w, callback_t _callback, void* _ptr) final
    {
        if (HTcolumns != 0)
            throw std::runtime_error("LB: l > 0 not yet supported");
        callback = _callback;
        ptr = _ptr;
        rows = HTpadded.rows();
    }
    
    void prepare_loop() final
    {
        curidx.resize(p + 1);
        cp = 0;
        curidx[0] = rows - 1;
    }
    
    bool loop_next() final
    {
        (*callback)(ptr, &curidx[0], &curidx[0] + cp, 0);
        return next();
    }

    bool next()
    {
//        std::cout << "!" << cp << "!" << curidx[0] << "." << curidx[1] << "." << curidx[2] << "   " << std::flush;
        if (++curidx[0] < rows)
            return true;
        unsigned i = 1;
        for (; i < cp; ++i)
            if (++curidx[i] < rows)
                break;
        if (i >= cp || curidx[i]+i >= rows)
        {
            if (++cp > p)
                return false;
            curidx[cp - 1] = 0;
            i = cp - 1;
        }
        for (; i > 0;)
        {
            --i;
            curidx[i] = curidx[i+1]+1;
        }
        return true;
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
    std::vector<uint32_t> curidx;
    size_t p, cp, rows;
};

MCCL_END_NAMESPACE

#endif
