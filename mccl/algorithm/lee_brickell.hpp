#ifndef MCCL_ALGORITHM_LEE_BRICKELL_HPP
#define MCCL_ALGORITHM_LEE_BRICKELL_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>

MCCL_BEGIN_NAMESPACE

class subISDT_lee_brickell
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    // API member function
    ~subISDT_lee_brickell() final
    {
    }
    
    subISDT_lee_brickell()
    {
        p = 3;
    }
    
    void configure(size_t _p = 3)
    {
        p = _p;
    }
    
    // API member function
    void initialize(const cmat_view& _HTpadded, size_t _HTcolumns, const cvec_view& _Spadded, unsigned int w, callback_t _callback, void* _ptr) final
    {
        HTpadded.reset(_HTpadded);
        Spadded.reset(_Spadded);
        columns = _HTcolumns;
        callback = _callback;
        ptr = _ptr;
        wmax = w;
        
        if (HTpadded.columns()-columns >= 64)
            throw std::runtime_error("subISDT_lee_brickell: HTpadded must round up columns to multiple of 64");
        // maybe we can generalize HTpadded.columns() not to be multiple of 64
        if (HTpadded.columns()%64 != 0)
            throw std::runtime_error("subISDT_lee_brickell: HTpadded must have columns multiple of 64");
        // TODO: allow HTpadded.columns()>64 by prefiltering using 1 word, then computing remaining words
        if (HTpadded.columns() > 64)
            throw std::runtime_error("subISDT_lee_brickell: currently doesn't support HTpadded with columns > 64");

        rows = HTpadded.rows();
        words = HTpadded.columns()/64;

        // need to check these are correct in all cases!
        lastwordmask = detail::lastwordmask( columns );
        firstwordmask = ((words == 1) ? lastwordmask : ~uint64_t(0));
        padmask = detail::lastwordmask( HTpadded.columns() ) & ~lastwordmask;
    }

    // API member function
    void solve() final
    {
        prepare_loop();
        if (words == 0)
        {
            while (_loop_next<false>())
                ;
        }
        else
        {
            while (_loop_next<true>())
                ;
        }
    }
    
    // API member function
    void prepare_loop() final
    {
        curidx.resize(p);
        curpath.resize(p+1, 0);
            
        cp = 1;
        curidx[0] = 0;
        if (words > 0)
        {
            firstwords.resize(rows);
            for (unsigned i = 0; i < rows; ++i)
                firstwords[i] = *HTpadded.data(i);
            curpath[0] = *Spadded.data();
            curpath[1] = curpath[0] ^ firstwords[0];
        }
    }

    // API member function
    bool loop_next() final
    {
        if (words == 0)
            return _loop_next<false>();
        else
            return _loop_next<true>();
    }
    
    template<bool use_curpath>
    bool _loop_next()
    {
        if (use_curpath)
        {
            if ((curpath[cp] & firstwordmask) == 0) // unlikely
            {
                unsigned int w = hammingweight(curpath[cp] & padmask);
                if (cp + w <= wmax)
                    if (!(*callback)(ptr, &curidx[0], &curidx[0] + cp, w))
                        return false;
            }
        }
        else
            if (!(*callback)(ptr, &curidx[0], &curidx[0] + cp, 0))
                return false;
        return next<use_curpath>();
    }

    template<bool use_curpath>
    inline bool next()
    {
        if (++curidx[cp - 1] < rows) // likely
        {
            if (use_curpath)
                curpath[cp] = curpath[cp-1] ^ firstwords[ curidx[cp-1] ];
            return true;
        }
        unsigned i = cp - 1;
        while (i >= 1)
        {
            if (++curidx[i-1] >= rows - (cp-i)) // likely
                --i;
            else
            {
                if (use_curpath)
                    curpath[i] = curpath[i-1] ^ firstwords[ curidx[i-1] ];
                break;
            }
        }
        if (i == 0)
        {
            if (++cp > p) // unlikely
                return false;
            curidx[0] = 0;
            if (use_curpath)
                curpath[1] = curpath[0] ^ firstwords[0];
            i = 1;
        }
        for (; i < cp; ++i)
        {
            curidx[i] = curidx[i-1] + 1;
            if (use_curpath)
                curpath[i+1] = curpath[i] ^ firstwords[ curidx[i] ];
        }
        return true;
    }
    
private:
    callback_t callback;
    void* ptr;
    cmat_view HTpadded;
    cvec_view Spadded;
    size_t columns, words;
    unsigned int wmax;
    
    std::vector<uint32_t> curidx;
    std::vector<uint64_t> curpath;
    std::vector<uint64_t> firstwords;
    
    uint64_t lastwordmask, firstwordmask, padmask;
    
    size_t p, cp, rows;
};

template<size_t _bit_alignment = 64>
using ISD_lee_brickell = ISD_generic<subISDT_lee_brickell,_bit_alignment>;

// TODO: provide further Lee Brickell convience functions

MCCL_END_NAMESPACE

#endif
