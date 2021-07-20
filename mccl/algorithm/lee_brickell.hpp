#ifndef MCCL_ALGORITHM_LEE_BRICKELL_HPP
#define MCCL_ALGORITHM_LEE_BRICKELL_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>

MCCL_BEGIN_NAMESPACE

struct lee_brickell_config_t
{
    const std::string modulename = "lee_brickell";
    const std::string description = "Lee-Brickell configuration";
    const std::string manualstring = 
        "Lee-Brickell:\n"
        "\tParameters: p\n"
        "\tAlgorithm:\n"
        "\t\tReturns all sets of at most p column indices of H2 that sum up to S2\n"
        ;

    unsigned int p = 3;

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 3, "subISDT parameter p");
    }
};

// global default. modifiable.
// at construction of subISDT_lee_brickell the current global default values will be loaded
extern lee_brickell_config_t lee_brickell_config_default;


class subISDT_lee_brickell
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    // API member function
    ~subISDT_lee_brickell() final
    {
        cpu_prepareloop.refresh();
        cpu_loopnext.refresh();
        cpu_callback.refresh();
        if (cpu_loopnext.total() > 0)
        {
            std::cerr << "prepare : " << cpu_prepareloop.total() << std::endl;
            std::cerr << "nextloop: " << cpu_loopnext.total() - cpu_callback.total() << std::endl;
            std::cerr << "callback: " << cpu_callback.total() << std::endl;
        }
    }
    
    subISDT_lee_brickell()
        : config(lee_brickell_config_default), stats("Lee-Brickell")
    {
    }

    void load_config(const configmap_t& configmap) final
    {
        mccl::load_config(config, configmap);
    }
    void save_config(configmap_t& configmap) final
    {
        mccl::save_config(config, configmap);
    }

    // API member function
    void initialize(const cmat_view& _H12T, size_t _H2Tcolumns, const cvec_view& _S, unsigned int w, callback_t _callback, void* _ptr) final
    {
        if (stats.cnt_initialize._counter != 0)
            stats.refresh();
        stats.cnt_initialize.inc();
        // copy parameters from current config
        p = config.p;
        if (p == 0)
            throw std::runtime_error("subISDT_lee_brickell::initialize: Lee Brickell does not support p = 0");

        H12T.reset(_H12T);
        S.reset(_S);
        columns = _H2Tcolumns;
        callback = _callback;
        ptr = _ptr;
        wmax = w;
        
        rows = H12T.rows();
        words = (columns+63)/64;

        if (words > 1)
            throw std::runtime_error("subISDT_lee_brickell::initialize(): Lee Brickell does not support l > 64");

        firstwordmask = lastwordmask(columns);
        padmask = ~firstwordmask;
    }

    // API member function
    void solve() final
    {
        stats.cnt_solve.inc();
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
        stats.cnt_prepare_loop.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_prepareloop);
        
        curidx.resize(p);
        curpath.resize(p+1, 0);
            
        cp = 1;
        curidx[0] = 0;
        if (words > 0)
        {
            firstwords.resize(rows);
            for (unsigned i = 0; i < rows; ++i)
                firstwords[i] = *H12T.data(i);
            curpath[0] = *S.data();
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
        stats.cnt_loop_next.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);
        
        if (use_curpath)
        {
            if ((curpath[cp] & firstwordmask) == 0) // unlikely
            {
                unsigned int w = hammingweight(curpath[cp] & padmask);
                if (cp + w <= wmax)
                {
                    MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
                    if (!(*callback)(ptr, &curidx[0], &curidx[0] + cp, w))
                        return false;
                }
            }
        }
        else
        {
            MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
            if (!(*callback)(ptr, &curidx[0], &curidx[0] + cp, 0))
                return false;
        }
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
            if (++curidx[i-1] >= rows - (cp-i)) // unlikely
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
    decoding_statistics get_stats() const { return stats; };

private:
    callback_t callback;
    void* ptr;
    cmat_view H12T;
    cvec_view S;
    size_t columns, words;
    unsigned int wmax;
    
    std::vector<uint32_t> curidx;
    std::vector<uint64_t> curpath;
    std::vector<uint64_t> firstwords;
    
    uint64_t firstwordmask, padmask;
    
    size_t p, cp, rows;
    
    lee_brickell_config_t config;
    decoding_statistics stats;
    cpucycle_statistic cpu_prepareloop, cpu_loopnext, cpu_callback;
};

template<size_t _bit_alignment = 64>
using ISD_lee_brickell = ISD_generic<subISDT_lee_brickell,_bit_alignment>;

vec solve_SD_lee_brickell(const cmat_view& H, const cvec_view& S, unsigned int w);
vec solve_SD_lee_brickell(const syndrome_decoding_problem& SD)
{
    return solve_SD_lee_brickell(SD.H, SD.S, SD.w);
}

vec solve_SD_lee_brickell(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap);
vec solve_SD_lee_brickell(const syndrome_decoding_problem& SD, const configmap_t& configmap)
{
    return solve_SD_lee_brickell(SD.H, SD.S, SD.w, configmap);
}



MCCL_END_NAMESPACE

#endif
