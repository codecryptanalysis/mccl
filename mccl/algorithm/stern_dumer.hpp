#ifndef MCCL_ALGORITHM_STERN_DUMER_HPP
#define MCCL_ALGORITHM_STERN_DUMER_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>

MCCL_BEGIN_NAMESPACE

struct lee_brickell_config_t
{
    const std::string modulename = "stern_dumer";
    const std::string description = "Stern/Dumer configuration";
    const std::string manualstring = 
        "Stern/Dumer:\n"
        "\tParameters: p\n"
        "\tAlgorithm:\n"
        "\t\tPartition columns of H2 into two sets.\n\t\tCompare p/2-columns sums from both sides.\n\t\tReturn pairs that sum up to S2.\n"
        ;

    unsigned int p = 3;

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 3, "subISDT parameter p");
    }
};

// global default. modifiable.
// at construction of subISDT_stern_dumer the current global default values will be loaded
extern stern_dumer_config_t stern_dumer_config_default;


// three stage collision bitfield:
// stage 1: compute all L1 values and set the first bit of the 2-bit value at the corresponding address
// stage 2: compute all L2 values, check the address. if the first bit is set (a collision with L1) then set the second bit and store the L2 value
// stage 3: compute all L1 values again, store the ones with second bit set at the corresponding address
template<bool usefilter1 = false, bool usefilter2 = false>
struct stern_dumer_bitfield
{
    // each 'address' is mapped to a bit position of a uint32_t word
    // the hashmap actually consists of a vector of uint64_t
    // the bottom half is the L1 uint32_t word
    // the top half is the L2 uint32_t word
    std::vector<uint64_t> hashmap;
    // the filter is another hashmap, but with a shorter address space
    // to obtain a speed-up:
    //  (1) the hashmap should not fit any cachelevel
    //  (2) it should be small enough to fit entirely in some cachelevel
    //  (3) it should still be big enough to filter a significant factor of look-ups
    // now each address uses only 1 bit, so filter1 for L1, filter2 for L2
    // filter1 can be used in stage 2 to quickly filter non-collisions of L2 values
    // filter2 can be used in stage 3 to quickly filter non-collisions of L1 values
    std::vector<uint64_t> filter1;
    std::vector<uint64_t> filter2;

    uint64_t addressmask_hashmap;
    uint64_t addressmask_filter1;
    uint64_t addressmask_filter2;
    uint32_t hashmap_bitshift;
    
    void clean()
    {
        std::fill(hashmap.begin(), hashmap.end(), uint64_t(0));
        std::fill(filter1.begin(), filter1.end(), uint64_t(0));
        std::fill(filter2.begin(), filter2.end(), uint64_t(0));
    }

    void resize(size_t hashmapaddressbits, size_t filter1addressbits = 0, size_t filter2addressbits = 0)
    {
        // check inputs
        if (hashmapaddressbits < 5)
            throw std::runtime_error("stern_dumer_map::resize(): hashmapaddressbits should be at least 5");
        if (usefilter1 == true && filter1addressbits < 6)
            throw std::runtime_error("stern_dumer_map::resize(): filter1 will be used, so filter1addressbits must be >= 6");
        if (usefilter2 == true && filter2addressbits < 6)
            throw std::runtime_error("stern_dumer_map::resize(): filter2 will be used, so filter2addressbits must be >= 6");
        if (usefilter1 == false && filter1addressbits != 0)
            throw std::runtime_error("stern_dumer_map::resize(): filter1 will NOT be used, so filter1addressbits must be 0");
        if (usefilter2 == false && filter2addressbits != 0)
            throw std::runtime_error("stern_dumer_map::resize(): filter2 will NOT be used, so filter2addressbits must be 0");

        hashmap.resize(size_t(1) << (hashmapaddressbits - 5));
        addressmask_hashmap = uint64_t(hashmap.size() - 1);

        filter1.resize(size_t(1) << (filter1addressbits - 6));
        addressmask_filter1 = uint64_t(filter1.size() - 1);

        filter2.resize(size_t(1) << (filter2addressbits - 6));
        addressmask_filter2 = uint64_t(filter2.size() - 1);

        // always call cleanup
        cleanup();
    }
        
    inline void filter1set(uint64_t L1val)
    {
        if (!usefilter1)
            return;
        filter1[ (L1val/64) & addressmask_filter1 ] |= uint64_t1(1) << (L1val%64);
    }
    inline void filter2set(uint64_t L2val)
    {
        if (!usefilter2)
            return;
        filter2[ (L2val/64) & addressmask_filter2 ] |= uint64_t1(1) << (L2val%64);
    }
    inline bool filter1get(uint64_t L2val)
    {
        if (!usefilter1)
            return true;
        return 0 != (filter1[ (L2val/64) & addressmask_filter1 ] |= uint64_t1(1) << (L2val%64));
    }
    inline bool filter2get(uint64_t L1val)
    {
        if (!usefilter2)
            return true;
        return 0 != (filter2[ (L1val/64) & addressmask_filter2 ] |= uint64_t1(1) << (L1val%64));
    }
    
    inline void stage1(uint64_t L1val)
    {
        hashmap[ (L1val/32) & addressmask_hashmap ] |= uint64_t(1) << (L1val%32);
        filter1set(L1val);
    }
    inline bool stage2(uint64_t L2val)
    {
        if (!filter1get(L2val))
            return false;
        uint64_t& x = hashmap[ (L2val/32) & addressmask_hashmap ];
        uint64_t L1bitval = uint64_t(1) << (L2val%32);
        if (0 == (x & L1bitval))
            return false;
        x |= L1bitval<<32;
        filter2set(L2val);
        return true;
    }
    inline bool stage3(uint64_t L1val)
    {
        if (!filter2get(L1val))
            return false;
        return 0 != (hashmap[ (L1val/32) & addressmask_hashmap ] & ((uint64_t(1)<<32) << (L1val%32)));
    }
};

// simple two-sided collision hashmap
// only all values from one side are stored in hashmap
// note that this expects each value to occur exactly once
// thus also each value-match produces exactly 1 collision
/* fastprimes[] = { 67, 131, 263, 521, 1031, 2053, 4099, 8209, 16417, 32771, 65539, 131113, 262147, 524309, 1048583, 2097169, 4194329, 8388617, 16777289, 33554473, 67108913, 134217989, 268435459, 536870923 }; */
template<typename SelT = uint64_t, typename ValT = uint64_t>
struct stern_dumer_hashmap_SC
{
    typedef SelT selection_type;
    typedef ValT value_type;
    
    std::unordered_map<ValT,SelT> map;
    size_t map_size;

    // call at parameterization    
    void reserve(size_t expected_elements, double scaling_factor = 2.0)
    {
        map_size = size_t(double(expected_elements) * scaling_factor);
        clean();
    }

    // call at start of process
    void clean()
    {
        index.clear();
        index.reserve(map_size);
    }    
    
    // call during phase 2
    template<typename ST, typename VT>
    void insert(ST&& s1, VT&& v1)
    {
        map.emplace(std::forward<VT>(v1), std::forward<ST>(s1));
    }
    
    // call at end of phase 2
    void optimize()
    {
    }

    // call during phase 3
    template<typename ST, typename VT, typename F>
    void match(const ST& s2, const VT& v2, F& f)
    {
        auto it = map.find(v2);
        if (it != map.end())
            f(it->second, s2);
    }
};


class subISDT_stern_dumer
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    // API member function
    ~subISDT_stern_dumer() final
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
    
    subISDT_stern_dumer()
        : config(stern_dumer_config_default), stats("Stern/Dumer")
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

        // copy initialization parameters
        H12T.reset(_H12T);
        S.reset(_S);
        columns = _H2Tcolumns;
        callback = _callback;
        ptr = _ptr;
        wmax = w;

        // copy parameters from current config
        p = config.p;
        // set attack parameters
        p1 = p/2; p2 = p - p1;
        rows = H12T.rows();
        rows1 = rows/2; rows2 = rows - rows1;

        words = (columns+63)/64;

        // check configuration
        if (p < 2)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support p < 2");
        if (columns < 6)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support ell < 6");
        if (words > 1)
            throw std::runtime_error("subISDT_stern_dumer::initialize(): Stern/Dumer does not support l > 64");

        firstwordmask = detail::lastwordmask(columns);
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
                firstwords[i] = *H12T.word_ptr(i);
            curpath[0] = *S.word_ptr();
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
    
    size_t p, p1, p2, rows, rows1, rows2;
    
    size_t cp;
    
    
    stern_dumer_config_t config;
    decoding_statistics stats;
    cpucycle_statistic cpu_prepareloop, cpu_loopnext, cpu_callback;
};

template<size_t _bit_alignment = 64>
using ISD_stern_dumer = ISD_generic<subISDT_stern_dumer,_bit_alignment>;

vec solve_SD_stern_dumer(const cmat_view& H, const cvec_view& S, unsigned int w);
vec solve_SD_stern_dumer(const syndrome_decoding_problem& SD)
{
    return solve_SD_stern_dumer(SD.H, SD.S, SD.w);
}

vec solve_SD_stern_dumer(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap);
vec solve_SD_stern_dumer(const syndrome_decoding_problem& SD, const configmap_t& configmap)
{
    return solve_SD_stern_dumer(SD.H, SD.S, SD.w, configmap);
}



MCCL_END_NAMESPACE

#endif
