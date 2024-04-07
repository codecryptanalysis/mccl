#ifndef MCCL_ALGORITHM_MMT_HPP
#define MCCL_ALGORITHM_MMT_HPP

#include <mccl/config/config.hpp>
#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/tools/unordered_multimap.hpp>
#include <mccl/tools/bitfield.hpp>
#include <mccl/tools/enumerate.hpp>

#include <unordered_map>
#include <type_traits>

MCCL_BEGIN_NAMESPACE

struct mmt_config_t
{
    const std::string modulename = "mmt";
    const std::string description = "mmt configuration";
    const std::string manualstring = 
        "MMT:\n"
        "\tParameters: p, l1\n"
        "\tAlgorithm:\n"
        "\t\tPartition columns of H2 into two sets.\n"
	"\t\tCompare p/2-columns sums from both sides.\n"
	"\t\tReturn pairs that sum up to S2.\n"
        ;

    unsigned int p = 4;
    unsigned int l1 = 6;
	unsigned int bucketsize = 10;

    template<typename Container>
    void process(Container& c)
    {
        c(p, "p", 4, "subISDT parameter p");
        c(l1, "l1", 6, "subISDT parameter l1");

		// TODO one can compute this directly. 
        c(bucketsize, "bucketsize", 10, "subISDT parameter bucketsize");
    }
};

// global default. modifiable.
// at construction of subISDT_mmt the current global default values will be loaded
extern mmt_config_t mmt_config_default;

template<
        typename keyType,
        typename valueType>
class SimpleHashMap {
    using data_type          = valueType;
    using index_type         = size_t;

public:
    typedef keyType 	T;

    typedef size_t 		LoadType;
    typedef size_t 		IndexType;

    // size per bucket
    const size_t bucketsize;

    // number of buckets
    const size_t nrbuckets;

    // total number of elements in the HM
    const size_t total_size = bucketsize * nrbuckets;

    using load_type = uint16_t;

	SimpleHashMap() = delete;
    /// constructor. Zero initializing everything
    SimpleHashMap(const size_t bucketsize, const size_t nrbuckets) noexcept :
		bucketsize(bucketsize), nrbuckets(nrbuckets){
    	__internal_hashmap_array.resize(total_size);
    	__internal_load_array.resize(nrbuckets);

	}

    /// \param e key element (hashed down = index within the internal array)
    /// \param value element to insert
    /// \param tid (ignored) can be anything
    void insert(const keyType &e,
                          const valueType value,
                          const uint32_t tid) noexcept {
        (void)tid;
        insert(e, value);
    }

    /// \param e element to insert
    /// \return nothing
    void insert(const keyType &e, const valueType value) noexcept {
        // hash down the element to the index
        const size_t index = e;
        size_t load = __internal_load_array[index];

        // early exit, if it's already full
        if (load == bucketsize) {
           return ;
        }


        // just some debugging checks
        __internal_load_array[index] += 1;

        /// NOTE: this store never needs to be atomic, as the position was
        /// computed atomically.
        if constexpr (std::is_array<data_type>::value) {
            memcpy(__internal_hashmap_array[index*bucketsize + load], value, sizeof(data_type));
        } else {
            __internal_hashmap_array[index*bucketsize + load] = value;
        }

    }

    /// \param e Element to hash down.
    /// \return the position within the internal const_array of `e`
    inline index_type find(const keyType &e) const noexcept {
        const index_type index = e;
        // return the index instead of the actual element, to
        // reduce the size of the returned element.
        return index*nrbuckets;
    }

    inline index_type find(const keyType &e, index_type &__load) const noexcept {
        const index_type index = e;
        __load = __internal_load_array[index];
        // return the index instead of the actual element, to
        // reduce the size of the returned element.
        return index*nrbuckets;
    }

    /// NOTE: can be called with only a single thread
    /// overwrites the internal data const_array
    /// with zero initialized elements.
    void clear() noexcept {
        memset(__internal_load_array.data(), 0, nrbuckets*sizeof(load_type));
    }

    // internal const_array
    alignas(1024) std::vector<data_type> __internal_hashmap_array;
    alignas(1024) std::vector<load_type> __internal_load_array;
};



class subISDT_mmt
    final : public subISDT_API
{
public:
    using subISDT_API::callback_t;
    using HMType = SimpleHashMap<uint64_t, 
		  std::pair<uint32_t, uint32_t>>;

    // API member function
    ~subISDT_mmt() final
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

		delete hashmap;
    }
    
    subISDT_mmt()
        : config(mmt_config_default), stats("MMT")
    {}

    void load_config(const configmap_t& configmap) final
    {
        mccl::load_config(config, configmap);
    }
    void save_config(configmap_t& configmap) final
    {
        mccl::save_config(config, configmap);
    }

    // API member function
    void initialize(const cmat_view& _H12T,
                    size_t _H2Tcolumns,
                    const cvec_view& _S,
                    unsigned int w,
                    callback_t _callback,
                    void* _ptr) final {
        if (stats.cnt_initialize._counter != 0) {
            stats.refresh();
        }
        stats.cnt_initialize.inc();

        // copy initialization parameters
        H12T.reset(_H12T);
        S.reset(_S);
        columns = _H2Tcolumns;
        callback = _callback;
        ptr = _ptr;

        // copy parameters from current config
        p = config.p;
        // set attack parameters
        p1 = p/4;
        l1 = config.l1;
        rows = H12T.rows();
        rows1 = rows/2; rows2 = rows - rows1;

        words = (columns+63)/64;

        // check configuration
        if (p % 4)
            throw std::runtime_error("subISDT_mmt::initialize: MMT does not support p % 4 != 0");
        if (columns < 6)
            throw std::runtime_error("subISDT_mmt::initialize: MMT does not support l < 6 (since we use bitfield)");
        if (words > 1)
            throw std::runtime_error("subISDT_mmt::initialize: MMT does not support l > 64 (yet)");
        if ( p1 > 3)
            throw std::runtime_error("subISDT_mmt::initialize: MMT does not support p > 3 (yet)");
        if (rows1 >= 65535 || rows2 >= 65535)
            throw std::runtime_error("subISDT_mmt::initialize: MMT does not support rows1 or rows2 >= 65535");
        if (l1 >= columns)
            throw std::runtime_error("subISDT_mmt::initialize: MMT does not support l1 >= l");

        firstwordmask = detail::lastwordmask(columns);
        l1mask = detail::lastwordmask(l1);
        helpermask = detail::lastwordmask(16*p1);
		
		hashmap_bucketsize = config.bucketsize;
        hashmap = new HMType{hashmap_bucketsize, 1u << l1};

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
        // TODO rand sucks
        iTl = rand() & l1mask;
        iTr = (Sval ^ iTl);
        
        hashmap->clear();
        Ihashmap.clear();
    }

    // API member function
    bool loop_next() final
    {
        stats.cnt_loop_next.inc();
        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_loopnext);

        // fill the first hashmap
        enumerate.enumerate(firstwords.data()+0, firstwords.data()+rows2, p1,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                hashmap->insert(val&l1mask,
                                std::pair<uint32_t, uint32_t>(val, pack_indices(idxbegin, idxend)));
            });

        // fill the intermediate list
        enumerate.enumerate(firstwords.data()+rows2, firstwords.data()+rows, p1,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                val ^= iTl;
                const uint64_t val2 = val & l1mask;

                auto it = idx;
                for (auto it2 = idxbegin; it2 != idxend; ++it2,++it) {
                    *it = *it2 + rows2;
                }
                const uint64_t tmp = pack_indices(idx, it) << (p1*16);

                const size_t hashmap_offset = val2*hashmap_bucketsize;
                const size_t left_load = hashmap->__internal_load_array[val2];
                for (auto iter = hashmap->__internal_hashmap_array.begin() + hashmap_offset;
                     iter != hashmap->__internal_hashmap_array.begin() + hashmap_offset + left_load;
                     iter++) {

                    const uint64_t val3 = val ^ iter->first;
                    const uint64_t tmp2 = tmp ^ (iter->second & helpermask);
                    Ihashmap.emplace(val3 >> l1, tmp2);
                }
            });

        // find collisions on the right side of the tree
        enumerate.enumerate(firstwords.data()+rows2, firstwords.data()+rows, p1,
            [this](const uint32_t* idxbegin, const uint32_t* idxend, uint64_t val)
            {
                val ^= iTr;
                const uint64_t val2 = val & l1mask;

                uint32_t* it = idx;
                for (auto it2 = idxbegin; it2 != idxend; ++it2,++it) {
                    *it = *it2 + rows2;
                }

                const size_t hashmap_offset = val2*hashmap_bucketsize;
                const size_t left_load = hashmap->__internal_load_array[val2];
                for (auto iter = hashmap->__internal_hashmap_array.begin() + hashmap_offset;
                     iter != hashmap->__internal_hashmap_array.begin() + hashmap_offset + left_load;
                     iter++) {

                    uint64_t val3 = val^iter->first;
                    val3 >>= l1;
                    auto *it2 = unpack_indices(iter->second, it, 1);

                    auto range = Ihashmap.equal_range(val3);
                    for (auto valit = range.first; valit != range.second; ++valit){
                        uint64_t packed_indices = valit->second;
                        auto *it3 = unpack_indices(packed_indices, it2, 2);

                        MCCL_CPUCYCLE_STATISTIC_BLOCK(cpu_callback);
                        if (!(*callback)(ptr, idx+0, it3, 0)) {
                            return false;
                        }
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

    uint32_t* unpack_indices(uint64_t x, uint32_t* first, const uint32_t nr)
    {
        for (size_t i = 0; i < nr; ++i)
        {
            uint32_t y = uint32_t(x & 0xFFFF);
            if (y == 0xFFFF)
                break;
            *first = y;
            ++first;
            x >>= 16;
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

    enumerate_t<uint32_t> enumerate;

    std::vector<uint64_t> firstwords;
    uint64_t firstwordmask, l1mask, helpermask, Sval, iTl, iTr;

    uint32_t idx[16] = {0};

    size_t p, l1, p1, rows, rows1, rows2;
    
    mmt_config_t config;
    decoding_statistics stats;
    cpucycle_statistic cpu_prepareloop, cpu_loopnext, cpu_callback;

    // std::unordered_multimap<uint64_t, uint64_t, StupidHasher, StupidCMP> hashmap;
    HMType *hashmap;
    std::unordered_multimap<uint64_t, uint64_t> Ihashmap;

    size_t hashmap_bucketsize;
};



template<size_t _bit_alignment = 64>
using ISD_mmt = ISD_generic<subISDT_mmt,_bit_alignment>;

vec solve_SD_mmt(const cmat_view& H, const cvec_view& S, unsigned int w);
static inline vec solve_SD_mmt(const syndrome_decoding_problem& SD)
{
    return solve_SD_mmt(SD.H, SD.S, SD.w);
}

vec solve_SD_mmt(const cmat_view& H, const cvec_view& S, unsigned int w, const configmap_t& configmap);
static inline vec solve_SD_mmt(const syndrome_decoding_problem& SD, const configmap_t& configmap)
{
    return solve_SD_mmt(SD.H, SD.S, SD.w, configmap);
}



MCCL_END_NAMESPACE

#endif
