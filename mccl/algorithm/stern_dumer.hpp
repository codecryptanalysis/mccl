#ifndef MCCL_ALGORITHM_STERN_DUMER_HPP
#define MCCL_ALGORITHM_STERN_DUMER_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/tools/unordered_multimap.hpp>
#include <mccl/tools/bitfield.hpp>
#include <mccl/tools/enumerate.hpp>

#include <unordered_map>

MCCL_BEGIN_NAMESPACE

struct stern_dumer_config_t
{
    const std::string modulename = "stern_dumer";
    const std::string description = "Stern/Dumer configuration";
    const std::string manualstring = 
        "Stern/Dumer:\n"
        "\tParameters: p\n"
        "\tAlgorithm:\n"
        "\t\tPartition columns of H2 into two sets.\n\t\tCompare p/2-columns sums from both sides.\n\t\tReturn pairs that sum up to S2.\n"
        ;

    unsigned int p = 4;

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 4, "subISDT parameter p");
    }
};

// global default. modifiable.
// at construction of subISDT_stern_dumer the current global default values will be loaded
extern stern_dumer_config_t stern_dumer_config_default;



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
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support l < 6 (since we use bitfield)");
        if (words > 1)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support l > 64 (yet)");
        if ( p > 8)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support p > 8 (yet)");
        if (rows1 >= 65535 || rows2 >= 65535)
            throw std::runtime_error("subISDT_stern_dumer::initialize: Stern/Dumer does not support rows1 or rows2 >= 65535");

        firstwordmask = detail::lastwordmask(columns);
        padmask = ~firstwordmask;
        
        bitfield.resize(columns);

        // TODO: compute a reasonable reserve size
        // hashmap.reserve(...);
    }

    // API member function
    void solve() final
    {
        stats.cnt_solve.inc();
        prepare_loop();
        while (loop_next())
            ;
    }
    
    // API member function
    void prepare_loop() final
    {
        stats.cnt_prepare_loop.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_prepareloop);
        
        firstwords.resize(rows);
        for (unsigned i = 0; i < rows; ++i)
            firstwords[i] = (*H12T.word_ptr(i)) & firstwordmask;
        Sval = (*S.word_ptr()) & firstwordmask;
        
        bitfield.clear();
        hashmap.clear();
    }

    // API member function
    bool loop_next() final
    {
        stats.cnt_loop_next.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);

//        std::cout << "1" << std::flush;
        // stage 1: store left-table in bitfield
        enumerate.enumerate_val(firstwords.data()+0, firstwords.data()+rows1, p1,
            [this](uint64_t val)
            { 
                bitfield.stage1(val); 
            });
//        std::cout << "2" << std::flush;
        // stage 2: compare right-table with bitfield: store matches
        enumerate.enumerate(firstwords.data()+rows1, firstwords.data()+rows, p2,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                val ^= Sval;
                if (bitfield.stage2(val))
                {
//                    std::cout << "a" << std::flush;
                    hashmap.emplace(val, pack_indices(idxbegin,idxend) );
                }
            });
//        std::cout << "3" << std::flush;
        // stage 3: retrieve matches from left-table and process
        enumerate.enumerate(firstwords.data()+0, firstwords.data()+rows1, p1,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                if (bitfield.stage3(val))
                {
//                    std::cout << "b" << std::flush;
                    uint32_t idx[16];
                    uint32_t* it = idx+0;
                    for (auto it2 = idxbegin; it2 != idxend; ++it2,++it)
                        *it = *it2;
                    auto range = hashmap.equal_range(val);
                    for (auto valit = range.first; valit != range.second; ++valit)
                    {
//                        std::cout << "c" << std::flush;
                        
                        uint64_t packed_indices = valit->second;
                        auto it2 = unpack_indices(packed_indices, it);

                        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
                        if (!(*callback)(ptr, idx+0, it2, 0))
                            return false;
                    }
                }
                return true;
            });
        return false;
    }
    
    static uint64_t pack_indices(const uint32_t* begin, const uint32_t* end)
    {
        uint64_t x = ~uint64_t(0);
        for (; begin != end; ++begin)
        {
            x <<= 16;
            x |= uint64_t(*begin);
        }
        return x;
    }
    
    static uint32_t* unpack_indices(uint64_t x, uint32_t* first)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            uint16_t y = uint16_t(x & 0xFFFF);
            if (y != 0xFFFF)
            {
                *first = y;
                ++first;
                x >>= 16;
            }
        }
        return first;
    }

    decoding_statistics get_stats() const { return stats; };

private:
    callback_t callback;
    void* ptr;
    cmat_view H12T;
    cvec_view S;
    size_t columns, words;
    unsigned int wmax;
    
    staged_bitfield<false,false> bitfield;
    std::unordered_multimap<uint64_t, uint64_t> hashmap;
    
    enumerate_t<uint32_t> enumerate;

    std::vector<uint64_t> firstwords;
    uint64_t firstwordmask, padmask, Sval;
    
    size_t p, p1, p2, rows, rows1, rows2;
    
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
