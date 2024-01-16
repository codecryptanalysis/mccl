#ifndef MCCL_ALGORITHM_SIEVING_HPP
#define MCCL_ALGORITHM_SIEVING_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/tools/enumerate.hpp>

MCCL_BEGIN_NAMESPACE

struct sieving_config_t
{
    const std::string modulename = "sieving";
    const std::string description = "Sieving configuration";
    const std::string manualstring =
        "Sieving:\n"
        "\tParameters: p\n"
        "\tAlgorithm:\n"
        "\t\tReturns all sets of at most p column indices of H2 that sum up to S2\n"
        ;

    unsigned int p = 3; // SE: potentially to change.

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 3, "subISDT parameter p"); // SE: potentially to change.
    }
};

// global default. modifiable.
// at construction of subISDT_sieving the current global default values will be loaded
extern sieving_config_t sieving_config_default;

class subISDT_sieving
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;

    // API member function
    ~subISDT_sieving() final
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

    subISDT_sieving()
        : config(sieving_config_default), stats("Sieving")
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
            throw std::runtime_error("subISDT_sieving::initialize: sieving does not support p = 0");

        H12T.reset(_H12T);
        S.reset(_S);
        columns = _H2Tcolumns;
        callback = _callback;
        ptr = _ptr;
        wmax = w;

        rows = H12T.rows();
        words = (columns + 63) / 64;

        if (words > 1)
            throw std::runtime_error("subISDT_sieving::initialize(): sieving does not support l > 64");

        // SD: Potentially add check configuration.
        firstwordmask = detail::lastwordmask(columns);
        padmask = ~firstwordmask;
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
        // SE: Potentially change as in Stern/Dumer.
        if (words > 0)
        {
            for (unsigned i = 0; i < rows; ++i)
                firstwords[i] = *H12T.word_ptr(i);
            Sval = (*S.word_ptr()) & firstwordmask;
        }
    }

    // API member function
    bool loop_next() final // SE: To be changed.
    {
        stats.cnt_loop_next.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);

        if (words == 0) // Solving the sub-instance.
        {
            enumerate.enumerate(firstwords.data() + 0, firstwords.data() + rows, p,
                [this](uint32_t* begin, uint32_t* end, uint64_t)
                {
                    MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
                    return (*callback)(ptr, begin, end, 0);
                });
        }
        else // Checking if a solution to the sub-instance yields a solution to the original problem.
        {
            enumerate.enumerate(firstwords.data() + 0, firstwords.data() + rows, p,
                [this](uint32_t* begin, uint32_t* end, uint64_t val)
                {
                    if ((val & firstwordmask) == Sval)
                    {
                        unsigned int w = hammingweight(val & padmask);
                        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
                        return (*callback)(ptr, begin, end, w);
                    }
                    return true;
                });
        }
        return false;
    }

    decoding_statistics get_stats() const { return stats; };

private:
    callback_t callback;
    void* ptr;
    cmat_view H12T;
    cvec_view S;
    size_t columns, words;
    unsigned int wmax;

    std::vector<uint64_t> firstwords;
    uint64_t firstwordmask, padmask, Sval;

    enumerate_t<uint32_t> enumerate;

    size_t p, rows;

    sieving_config_t config;
    decoding_statistics stats;
    cpucycle_statistic cpu_prepareloop, cpu_loopnext, cpu_callback;
};

template<size_t _bit_alignment = 64>
using ISD_sieving = ISD_generic<subISDT_sieving,_bit_alignment>;

vec solve_SD_sieving(const cmat_view& H, const cvec_view& S, unsigned int w);
static inline vec solve_SD_sieving(const syndrome_decoding_problem& SD)
{
    return solve_SD_sieving(SD.H, SD.S, SD.w);
}

vec solve_SD_sieving(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap);
static inline vec solve_SD_sieving(const syndrome_decoding_problem& SD, const configmap_t& configmap)
{
    return solve_SD_sieving(SD.H, SD.S, SD.w, configmap);
}

MCCL_END_NAMESPACE

#endif