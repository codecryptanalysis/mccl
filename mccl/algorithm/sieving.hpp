#ifndef MCCL_ALGORITHM_SIEVING_HPP
#define MCCL_ALGORITHM_SIEVING_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/tools/enumerate.hpp>
#include <unordered_set>

MCCL_BEGIN_NAMESPACE

typedef std::array<uint32_t, 4> indexarray_t;
typedef std::pair<indexarray_t, uint64_t> element_t;
typedef std::unordered_set<element_t> database;

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

    size_t p = 3, alpha = 1, N = 100; // SE: Potentially change.
    std::string alg = "GJN";

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 3, "subISDT parameter p");
        c(alpha, "alpha", 1, "subISDT parameter alpha");
        c(N, "N", 100, "subISDT parameter N");
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
        if (columns == 0)
            throw std::runtime_error("subISDT_sieving::initialize: sieving does not support l = 0");
        callback = _callback;
        ptr = _ptr;
        wmax = w;

        rows = H12T.rows();
        words = (columns + 63) / 64;
        N = config.N;
        alpha = config.alpha;

        if (words > 1)
            throw std::runtime_error("subISDT_sieving::initialize(): sieving does not support l > 64");

        // SE: Potentially add check configuration.
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
        for (unsigned i = 0; i < rows; ++i)
                firstwords[i] = *H12T.word_ptr(i); // SE: Why we don't apply mask to this part, too?
        Sval = (*S.word_ptr()) & firstwordmask;
    }

    // sampling N random vectors of weight w
    void sample_vec(size_t element_weight, size_t output_length, database output)
    {
        output.clear();

        uint64_t rnd_val;
        element_t element;
        mccl_base_random_generator rnd = mccl_base_random_generator();
        while (output.size() < output_length)
        {
            element.second = 0;
            for (unsigned k = 0; k < element_weight; ++k)
            {
                element.first[k] = rnd() % rows;
                for (unsigned i = 0; i < k; ++i)
                {
                    while (element.first[i] == element.first[k])
                    {
                        element.first[k] = rnd() % rows;
                        i = 0;
                    }
                }
                element.second = firstwords[element.first[k]];
            }

            output.insert(element);
        }
        

        //element_t element;
        //for (size_t i = 0; i < output_length; ++i) // to modify
        //{
        //    element = 0;
        //    while (hammingweight(element) < element_weight)
        //    {
        //        rnd_val = rnd() % columns;
        //        //((element >> rnd_val) & 1);
        //        element |= uint64_t(1) << rnd_val;
        //    }
        //    output.push_back(element);
        //}
    }

    // API member function
    bool loop_next() final
    {
        stats.cnt_loop_next.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);

        // sampling N random vectors of weight p
        std::unordered_set<uint64_t> listini;
        sample_vec(p, N, listini);

        // sieving part
        std::unordered_set<uint64_t> listout;
        for (unsigned int i = 0; i < rows; ++i)
        {
            uint64_t Si = (Sval >> i) & 1;

            // check if any of the previously sampled e satisfy the first ith constraint
            for (const auto& val : listini)
            {
                if((val & firstwordmask & 1) == Si)
                    listout.insert(val);
            }

            // bucketing
            std::vector<uint64_t> centers;
            sample_centers(centers, alg);

            std::vector< std::vector<uint64_t> > output;
            bucketing(listini, centers, output);

            // check if any of the summed vectors from NNS satisfy the first i constraints
            for (auto& bucket : output)
            {
                for (auto& x : bucket)
                {
                    for (auto& y : bucket)
                    {
                        if (hammingweight(x & y) == (p - alpha) &&
                            (hammingweight(firstwords[i] & (x + y) & firstwordmask) & 1) == Si) // SE: modify the check so that only val is checked against & 1
                            listout.insert(x + y);
                    }
                }
            }

            listini.swap(listout);
            listout.clear();
        }

        for (auto& val : listini)
        {
            if ((val & firstwordmask) == Sval)
            {
                unsigned int w = hammingweight(val & padmask);
                MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
                // SE: to check???
                return (*callback)(ptr, element.first + 0, element.first + element_weight, w);
            }
            return true;
        }

        return false;
    }

    // bucketing routine
    void bucketing(const std::unordered_set<uint64_t>& listin,
            const std::vector<uint64_t>& centers,
            std::vector< std::vector<uint64_t> >& output)
    {
        output.resize(centers.size());
        for (auto& x : output)
            x.clear();

        for (const auto& x : listin)
        {
            for (unsigned i = 0; i < centers.size(); ++i)
            {
                if (hammingweight(x & centers[i]) == alpha)
                    output[i].push_back(x);
            }
        }
    }

    // calculate binomial coefficient ("n choose k")
    size_t binomial_coeff(size_t n, size_t k)
    {
        if (k > n)
            return 0;
        if (k == 0 || k == n)
            return 1;

        return binomial_coeff(n - 1, k - 1)
            + binomial_coeff(n - 1, k);
    }

    // centers sampling routine
    void sample_centers(std::vector<uint64_t>& centers, std::string alg)
    {
        if (alg.compare("GJN"))
        {
            size_t num = binomial_coeff(columns, p / 2);
            enumerate_vec(p / 2, num, centers);
        }
        //else if (alg.compare("Hash"))
        //{
        //    // 1. sample a code, namely a parity-check matrix of size r x n (i.e. r x columns)
        //    r = columns / 2;
        //    // sample matrix
        //    std::vector<uint64_t> code; // SE: How ??? (to fill in)

        //    // 2. enumerate all vectors of weight v
        //    v = alpha;
        //    size_t num = binomial_coeff(columns, v);
        //    enumerate_vec(v, num, centers);
        //    for (size_t i = 0; i < centers.size(); ++i)
        //    {
        //        for (size_t j = 0; j < r; ++j)
        //        {
        //            if ((hammingweight(centers[i] & code[j]) & 1) != 0)
        //            {
        //                centers.erase(centers.begin() + i);
        //                continue;
        //            }

        //        }
        //    }

        //}
        //else if (alg.compare("RPC"))
        //{

        //}
        else
        {
            throw std::runtime_error("subISDT_sieving::sample_centers: sieving does not support algorithms other than GJN.");
        }
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

    size_t p, rows, N, alpha, v, r;
    std::string alg;

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