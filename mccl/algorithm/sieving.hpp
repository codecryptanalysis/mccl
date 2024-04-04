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

typedef std::array<uint32_t, 2> indexarray_t_center;
typedef std::pair<indexarray_t_center, uint64_t> center_t;

extern size_t loop_it;

// custom hash function for element_t
struct element_hash_t
{
    std::size_t operator()(const element_t& e) const noexcept
    {
#if 1
	// we assume that the XOR of the H21T firstwords acts random and for practical purposes acts as identifier for the selection of row indices in e.first
	return e.second;
#else
	// compute a hash value from the indices in e.first
	std::size_t h = e.first[0];
	for (unsigned i = 1; i < e.first.size(); ++i)
		h ^= e.first[i] + 0x9e3779b9 + (seed<<6) + (seed>>2);
	return h;
#endif	    
    }
};
typedef std::unordered_set<element_t, element_hash_t> database;

// intersect element x and y and returns the size of intersection 
size_t intersection_elements(const element_t&, const element_t&, size_t);
size_t intersection_elements(const element_t&, const center_t&, size_t, size_t);

// combine element x and y into element dest: 
// - assume x and y have element_weight indices
// - returns true if intersection of x and y equals p - alpha
// - dest contains the indices from x and y that occur exactly once (essentially x XOR y)
bool combine_elements(const element_t&, const element_t&, element_t&, size_t);

// sampling N random vectors of weight w:
// INVARIANT1: element.second = xor_{i=0}^{elementweight-1} firstwords[element.first[i]];
// INVARIANT2: element.first[0, ..., elementweight - 1] is a sorted array with values in[0, ..., rows - 1]
void sample_vec(size_t, size_t, size_t, const std::vector<uint64_t>&, mccl_base_random_generator, database&);

// calculate binomial coefficient ("n choose k")
size_t binomial_coeff(size_t, size_t);

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

    size_t p = 4, alpha = 2, N = 400; // SE: Potentially change.
    std::string alg = "GJN";

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 4, "subISDT parameter p");
        c(alpha, "alpha", 2, "subISDT parameter alpha");
        c(N, "N", 800, "subISDT parameter N");
        c(alg, "alg", "GJN", "subISDT algorithm");
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

        p = config.p;
        if (p == 0)
            throw std::runtime_error("subISDT_sieving::initialize: sieving does not support p = 0");


        // copy initialization parameters
        H12T.reset(_H12T);
        S.reset(_S);
        columns = _H2Tcolumns;
        callback = _callback;
        ptr = _ptr;
        wmax = w;
        rows = H12T.rows();
        words = (columns + 63) / 64;

        // copy parameters from current config
        N = config.N;
        alpha = config.alpha;
        alg = config.alg;


        // checks
        if (columns == 0)
            throw std::runtime_error("subISDT_sieving::initialize: sieving does not support l = 0");
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
                firstwords[i] = *H12T.word_ptr(i);
        Sval = (*S.word_ptr()) & firstwordmask;
    }
    
    // API member function
    bool loop_next() final
    {
        // std::cout << "Loop iteration: " << loop_it << std::endl;

        stats.cnt_loop_next.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);

        // sampling N random vectors of weight p
        database listini;
        sample_vec(p, rows, N, firstwords, rnd, listini);

        // sampling centers
        std::vector<center_t> centers;
        sample_centers(centers);

        // sieving part
        database listout;
        std::vector<std::vector<element_t>> buckets;
        for (unsigned int i = 0; i < columns; ++i)
        {
            listout.clear();
            uint64_t Si_mask = (uint64_t(2) << i) - 1; //SE: why this?
            uint64_t Si = Sval & Si_mask;

            // check if any of the previously sampled e satisfy the first ith constraint
            for (const auto& element : listini)
            {
                if ((element.second & Si_mask) == Si || (element.second & Si_mask) == 0)
                    listout.insert(element);
            }

#if 1
            // near neighbor search
            bucketing(listini, centers, buckets);
            checking(buckets, Si, Si_mask, listout);
#else
            element_t element_xy = *listini.begin(); //SE: Why this?
            for (const auto& element_x : listini)
            {
                for (const auto& element_y : listini)
                {
                    if (combine_elements(element_x, element_y, element_xy, p))
                    {
                        if ((element_xy.second & Si_mask) == Si || (element_xy.second & Si_mask) == 0)
                        {
                            listout.insert(element_xy);
                        }
                    }
                }
            }
#endif           
            // std::cout << "listout size: " << listout.size() << ", \t" << std::flush;
#if 0 
            listini.clear();
            size_t good = 0;
            uint64_t goodmask = (uint64_t(2) << i) - 1;
            for (database::iterator it = listout.begin(); it != listout.end(); ++it)
            {
                if (((it->second ^ Sval) & goodmask) == 0 || ((it->second) & goodmask) == 0)
                    ++good;
                listini.insert(*it);
                if (listini.size() == N)
                    break;
            }
            listout.clear();
#else           
            listini.swap(listout);
            resample(listini, N);
            listout.clear();
#endif
            // std::cout << "listin size: " << listini.size() << " " << std::endl << std::flush;
        }

        loop_it++;

        size_t good = 0;
        for (const auto& element : listini)
        {
            if ((element.second & firstwordmask) == Sval)
            {
                unsigned int wH1part = hammingweight(element.second & padmask);
                MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
                {
                    ++good;
                    const uint32_t* beginptr = &element.first[0];
                    if (!(*callback)(ptr, beginptr, beginptr + p, 0))
                        return false;
                }
            }
        }
        std::cout << good << " " << std::endl;
        return false;
    }

    // centers sampling routine
    void sample_centers(std::vector<center_t>& centers)
    {
        centers.clear();
        center_t center;
        if (alg.compare("GJN") == 0)
        {
            for (unsigned i = 0; i < rows - 1; i += 1)
            {
                center.first[0] = i;
                for (unsigned j = i + 1; j < rows; j += 1)
                {
                    center.first[1] = j;
                    center.second = firstwords[i] ^ firstwords[j];
                    centers.push_back(center);
                }    
            }
        }
        else
        {
            throw std::runtime_error("subISDT_sieving::sample_centers: sieving does not support algorithms other than GJN.");
        }
    }

    // routine for determining valid centers
    void find_valid_centers(const element_t& element, const std::vector<center_t>& centers, std::vector<size_t>& valid_centers)
    {
        valid_centers.clear();
        if (alg.compare("GJN") == 0)
        {
            for (unsigned i = 0; i < centers.size(); ++i)
            {
                if(intersection_elements(element, centers[i], p, alpha) == alpha)
                    //valid_centers.push_back(centers[i]);
                    valid_centers.push_back(i);
            }
        }
        else
        {
            throw std::runtime_error("subISDT_sieving::sample_centers: sieving does not support algorithms other than GJN.");
        }
    }

    // bucketing routine
    void bucketing(const database& listin, const std::vector<center_t>& centers, std::vector<std::vector<element_t>>& buckets)
    {
        buckets.resize(centers.size());
        for (auto& b : buckets)
            b.clear();

        std::vector<size_t> valid_centers;
        for (const auto& element : listin)
        {
            find_valid_centers(element, centers, valid_centers);
            for (const auto& vc : valid_centers)
            {
                buckets[vc].push_back(element);
            }
        }
    }

    // checking routine
    void checking(const std::vector<std::vector<element_t>>& buckets, uint64_t Si, uint64_t Si_mask, database& listout)
    {
        element_t element_new;
        for (const auto& bucket : buckets)
        {
            if (bucket.size() == 0)
                continue;
            for (size_t j = 0; j < bucket.size() - 1; ++j)
            {
                for (size_t k = j + 1; k < bucket.size(); ++k)
                {
                    if (combine_elements(bucket[j], bucket[k], element_new, p))
                    {
                        if (listout.count(element_new) > 0)
                            continue;
                        if ((element_new.second & Si_mask) == Si || (element_new.second & Si_mask) == 0)
                            listout.insert(element_new);
                    }
                }
            }
        }
    }

    // resampling
    void resample(database& listout, size_t N)
    {
        if (listout.size() < N)
            return;

        while (listout.size() > N)
        {
            size_t random_ind = rnd() % listout.size();
            auto random_it = std::next(listout.begin(), random_ind);
            listout.erase(random_it);
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

	mccl_base_random_generator rnd;
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
