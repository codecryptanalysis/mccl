#ifndef MCCL_TOOLS_HASH_TABLE_HPP
#define MCCL_TOOLS_HASH_TABLE_HPP

#include <mccl/config/config.hpp>

#include <mccl/tools/aligned_vector.hpp>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>

MCCL_BEGIN_NAMESPACE

namespace detail
{

// hash table utility: use prime number p of buckets and use bucket = (hash % p)
/*
   hash primes for fast modular arithmetic avoiding heavy division instruction:

    div operation (n/p) is simple fast computation of 1 mul, 1 shift       : div(n) = n/p = (n * _muldiv) >> (64+_shift)
    mod operation (n%p) is simple fast computation of 2 mul, 1 shift, 1 sub: mod(n) = n%p = n - div(n) *p

   sufficiently many primes allow such a fast computation using _muldiv and _shift
*/

class hash_prime
{
public:
// use __int128 without throwing a compiler warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    typedef unsigned __int128 uint128_t;
#pragma GCC diagnostic pop

    hash_prime(uint64_t prime = 0, uint64_t muldiv = 0, unsigned shift = 0)
        : _prime(prime), _muldiv(muldiv), _shift(shift)
    {
        _check();
    }
    
    hash_prime(const hash_prime& ) = default;
    hash_prime(hash_prime&& ) = default;
    hash_prime& operator= (const hash_prime&) = default;
    hash_prime& operator= (hash_prime&&) = default;

    const uint64_t& prime() const { return _prime; }

    // returns n / prime, using 1 multiplication, 1 shift
    inline uint64_t div(uint64_t n) const
    {
        // get the top 64-bit of the 128-bit multiplication n * _muldiv
        uint64_t div = (uint128_t(n) * _muldiv) >> 64;
        // shift right by _shift
        div >>= _shift;
        return div;
    }
    // returns n % prime, using 2 multiplications, 1 shift, 1 subtraction
    inline uint64_t mod(uint64_t n) const
    {
        // n % p = n - div*p
        return n - div(n) * _prime;
    }
    
private:
    uint64_t _prime;
    uint64_t _muldiv;
    unsigned _shift;

    // verifies parameters are 0, or are correct to compute mod prime
    void _check();
};


// 3820 hash_prime values <= 2^63 of varying bitsizes have been precomputed
// and may obtained through the following functions:

// obtain smallest internal hash_prime with prime > n
hash_prime get_hash_prime_gt(uint64_t n);
// obtain smallest internal hash_prime with prime >= n
hash_prime get_hash_prime_ge(uint64_t n);
// obtain largest internal hash_prime with prime < n
hash_prime get_hash_prime_lt(uint64_t n);
// obtain largest internal hash_prime with prime <= n
hash_prime get_hash_prime_le(uint64_t n);

// create your own hash_prime for given p
// by computing a suitable muldiv and shift value for the given p
// (it does not do a prime check of p, though)
// if dothrow == true then throws when it fails, otherwise it returns hash_prime(0,0,0)
hash_prime create_hash_prime(uint64_t p, bool dothrow = true);


// hash function that returns uint64_t hash
uint64_t hash(uint64_t x) { return x; }
// overloads for custom types may use hash_combine:
uint64_t hash_combine(uint64_t x, uint64_t y) { return 4611686018427388039ULL * x + 268435459ULL * y + 2147483659; }


/*
   packed cacheline bucket:
    try to pack the maximum number of elements in a cacheline
    expects size of Key/Value to be a integer multiple of alignment
    orders Key array and Value array by larger alignment first, break tie by larger size
*/
template<typename Key, typename Value, size_t CachelineSize = 64>
struct alignas(CachelineSize) cacheline_bucket_key_first_t
{
    static_assert((sizeof(Key)   % alignof(Key))   == 0, "sizeof(Key) must be multiple of alignof(Key)");
    static_assert((sizeof(Value) % alignof(Value)) == 0, "sizeof(Value) must be multiple of alignof(Value)");
    static_assert(alignof(Key) >= alignof(Value), "cacheline_key_first_t: Key type alignment must not be smaller than of Value type");
    
    typedef uint8_t bucket_size_t;

    static constexpr size_t cacheline_size = CachelineSize;
    static constexpr size_t bucket_size = (cacheline_size - sizeof(bucket_size_t)) / (sizeof(Key)+sizeof(Value));
    static_assert(size_t(~bucket_size_t(0)) >= bucket_size, "cacheline_bucket_key_first_t: unexpected bucket size");

    Key keys[bucket_size];
    Value values[bucket_size];
    bucket_size_t size;
};

template<typename Key, typename Value, size_t CachelineSize = 64>
struct alignas(CachelineSize) cacheline_bucket_value_first_t
{
    static_assert((sizeof(Key)   % alignof(Key))   == 0, "sizeof(Key) must be multiple of alignof(Key)");
    static_assert((sizeof(Value) % alignof(Value)) == 0, "sizeof(Value) must be multiple of alignof(Value)");
    static_assert(alignof(Value) >= alignof(Key), "cacheline_value_first_t: Value type alignment must not be smaller than of Key type");

    typedef uint8_t bucket_size_t;

    static constexpr size_t cacheline_size = CachelineSize;
    static constexpr size_t bucket_size = (cacheline_size - sizeof(bucket_size_t)) / (sizeof(Key)+sizeof(Value));
    static_assert(size_t(~bucket_size_t(0)) >= bucket_size, "cacheline_bucket_value_first_t: unexpected bucket size");

    Value values[bucket_size];
    Key keys[bucket_size];
    bucket_size_t size;
};

template<typename Key, typename Value, size_t CachelineSize = 64>
using cacheline_bucket_t = typename std::conditional< 
    (alignof(Value)>alignof(Key)) || (alignof(Value)==alignof(Key) && sizeof(Value)>sizeof(Key)),
    cacheline_bucket_value_first_t <Key,Value,CachelineSize>,
    cacheline_bucket_key_first_t   <Key,Value,CachelineSize>
    >::type;

} // namespace detail



struct default_unordered_multimap_traits
{
    // architectural cacheline size
    static constexpr size_t cacheline_size = 64;
    // max_load_factor determines max capacity at which either automatic rehash or insert loss occurs
    static constexpr float default_max_load_factor = 0.9f;
    // scale determines default reserve size for a certain number of elements
    static constexpr float default_scale_factor = 1.5f;

    // automatic grow parameters
    static constexpr bool is_auto_grow = false;
    static constexpr float default_grow_factor = 1.4f;

    // default queue size for insert & match operations
    static constexpr size_t default_insert_batch_size = 128;
    static constexpr size_t default_match_batch_size = 128;
};





/** cacheline hash table for ISD **/
// simple unordered_multimap using cacheline buckets
// - buckets are aligned to cachelines and contain as many elements as possible in a cacheline
// - buckets overflow in subsequent buckets
// usage:
//  first reserve() / clear() to setup / clear hashmap for the target number of elements
//  then fill with insert(k,v) with elements (multimap => duplicate keys will be stored)
//  then lookup keys with elements: match(k, f)
//  will search starting at bucket(k) to find elements with that key
//   - for each matching key (may be more than 1) with value v it will call f(v)
//   - at the first empty position it aborts the search
template<typename Key = uint64_t, typename Value = uint64_t, typename Traits = default_unordered_multimap_traits>
class cacheline_unordered_multimap
{
public:
    typedef Key key_type;
    typedef Value value_type;
    typedef Traits traits;
    typedef detail::hash_prime hash_prime;

    static constexpr bool is_auto_grow = traits::is_auto_grow;
    static constexpr size_t cacheline_size = traits::cacheline_size;
    
    static constexpr float default_scale_factor = traits::default_scale_factor;
    static constexpr float default_max_load_factor = traits::default_max_load_factor;
    static constexpr float default_grow_factor = traits::default_grow_factor;

    typedef detail::cacheline_bucket_t<key_type,value_type,cacheline_size> bucket_t;
    static constexpr size_t bucket_size = bucket_t::bucket_size;

    static_assert( std::is_trivial<Key>::value, "Key type is not trivial");
    static_assert( std::is_trivial<Value>::value, "Value type is not trivial");
    static_assert(bucket_size > 0, "bucket_size must be non-zero");
    static_assert(sizeof(bucket_t) == cacheline_size, "sizeof(bucket_t) != cacheline_size");
    static_assert(alignof(bucket_t) == cacheline_size, "alignof(bucket_t) != cacheline_size");



    cacheline_unordered_multimap(float __max_load_factor = default_max_load_factor, float __grow_factor = default_grow_factor)
        : _max_load_factor(__max_load_factor), _grow_factor(__grow_factor), _size(0), _max_size(0), _reserved_size(1)
    {
    }
    
    cacheline_unordered_multimap(const cacheline_unordered_multimap& ) = default;
    cacheline_unordered_multimap(      cacheline_unordered_multimap&&) = default;
    cacheline_unordered_multimap& operator= (const cacheline_unordered_multimap& ) = default;
    cacheline_unordered_multimap& operator= (      cacheline_unordered_multimap&&) = default;

    // read configuration
    bool   empty()             const { return _size == 0; }
    size_t size()              const { return _size; }
    size_t capacity()          const { return _max_size; }
    size_t bucket_count()      const { return _reserved_size; }
    float  load_factor()       const { return float(_size) / float(_reserved_size); }
    float  max_load_factor()   const { return _max_load_factor; }
    float  grow_factor()       const { return _grow_factor; }

    // set new max_load_factor
    void max_load_factor(float ml)
    {
        // set max load
        _max_load_factor = ml;
        // recalculate _max_size
        _max_size = float(_hp.prime()) * float(bucket_size) * _max_load_factor;
        // changing _max_size may trigger a rehash (if autogrow is enabled)
        if (is_auto_grow && _size > _max_size)
            rehash( _grow_factor * std::max<float>( float(_reserved_size) , float(_size)/_max_load_factor ) );
    }
    
    // set new grow factor
    void grow_factor(float gf)
    {
        // prevent too low grow factor
        _grow_factor = std::max<float>(gf, 1.01);
    }

    // reserve to store a given number of elements
    void reserve(size_t elements, double scale = default_scale_factor)
    {
        // lower bound scale by 1/max_load_factor
        scale = std::max<float>(scale, 1.0f/_max_load_factor);
        // compute target number of buckets
        size_t buckets = double(elements) * scale / double(bucket_size);
        // resize, and rehash contents when necessary
        rehash(buckets);
    }
    
    // reserve a given number of buckets, multimap must be empty
    void _reserve(size_t buckets)
    {
        if (!empty())
            throw std::runtime_error("cacheline_unordered_multimap::_reserve(): multimap not empty");
        // find fast prime >= buckets
        _hp = detail::get_hash_prime_ge(buckets);
        // resize map and set _max_size
        _reserved_size = _hp.prime() * bucket_size;
        _max_size = float(_reserved_size) * _max_load_factor;
        _map.resize(_hp.prime()+1);
        // check proper alignment
        uintptr_t check_align = reinterpret_cast<uintptr_t>(&_map[0]);
        if (0 != (check_align & (cacheline_size-1)))
            throw std::runtime_error("cacheline_hash_table::reserve(): aligned allocation failed");
        // clear map
        clear();
    }

    // resize multimap to a different number of buckets
    bool rehash(size_t buckets)
    {
        // if empty then perform simpler _reserve directly
        if (empty())
        {
            _reserve(buckets);
            return true;
        }
        // cannot shrink number of buckets if this violates max_load_factor
        if (float(buckets)*_max_load_factor <= float(_size))
            return false;
        // create temporary multimap with same parameters
        cacheline_unordered_multimap tmp(_max_load_factor, _grow_factor);
        // reserve target number of buckets
        tmp._reserve(buckets);
        // and insert all elements
        for (auto& B : _map)
        {
            for (size_t i = 0; i < B.size; ++i)
                tmp.insert(B.keys[i], B.values[i]);
        }
        // move assign data from temporary multimap
        *this = std::move(tmp);
        return true;
    }

    // clear multimap
    void clear()
    {
        _size = 0;
        if (!_map.empty())
            memset(&_map[0], 0, sizeof(bucket_t)*_hp.prime());
    }

    // compute uint64_t hash of key
    uint64_t hash(const key_type& k) const
    {
        return mccl::detail::hash(k);
    }
    
    // compute bucket from key via hash
    uint64_t bucket(const key_type& k) const
    {
        return _hp.mod( this->hash(k) );
    }

    // prefetch target bucket to speed up operations
    inline void prefetch(const key_type& k)
    {
        uint64_t h = bucket(k);
        __builtin_prefetch(&_map[h].size,1,0);
    }
    
    bool insert(const key_type& k, const value_type& v)
    {
        if (size() >= capacity())
        {
            if (!is_auto_grow)
                return false;
            rehash( float(_reserved_size) * _grow_factor );
        }
        // already increase size, since we always insert the element
        ++_size;
        // compute initial bucket index to start search
        uint64_t b = bucket(k);
        while (true)
        {
            // obtain bucket
            auto& B = _map[b];
            // if bucket is already full then we move to next bucket
            if (__builtin_expect(B.size == bucket_size, 0))
            {
                if (++b == _hp.prime())
                    b = 0;
                continue;
            }
            // insert element in first non-full bucket encountered
            auto j = B.size ++;
            B.keys[j] = k;
            B.values[j] = v;
            return true;
        }
    }
    
    template<typename F>
    void match(const key_type& k, F&& f) const
    {
        // compute intial bucket index to start search
        uint64_t b = bucket(k);
        while (true)
        {
            const auto& B = _map[b];
            if (B.size < bucket_size)
            {
                for (size_t j = 0; j < B.size; ++j)
                    if (B.keys[j] == k)
                        f(B.values[j]);
                return;
            }
            else
            {
                __builtin_prefetch(&_map[b+1].size,0,0);
                for (size_t j = 0; j < bucket_size; ++j)
                    if (B.keys[j] == k)
                        f(B.values[j]);
            }
            // increase b mod p
            if (++b == _hp.prime())
                b = 0;
        }
    }
    
private:
    float _max_load_factor, _grow_factor;
    size_t _size, _max_size, _reserved_size;
    hash_prime _hp;

    aligned_vector<bucket_t> _map;
};


/** batched cacheline hash table **/
// very similar to the above simple unordered_multimap using cacheline buckets
// however:
// - has a queue for inserts and matches:
//   - when an operation enters the queue, the target bucket is prefetched
//   - when the queue is full then the entire queue is processed
// - besides queue_insert / queue_match we also have:
//   - finalize_insert / finalize_match: process all elements in the queue and clear the queue
// - !! do not forget to call finalize_insert / finalize_match at the end of the insert / match phase

template<typename Key = uint64_t, typename Value = uint64_t, typename traits = default_unordered_multimap_traits>
class batch_unordered_multimap
{
public:
    typedef Key key_type;
    typedef Value value_type;
    typedef detail::hash_prime hash_prime;

    static constexpr bool is_auto_grow = traits::is_auto_grow;
    
    static constexpr size_t cacheline_size = traits::cacheline_size;
    static constexpr size_t default_insert_batch_size = traits::default_insert_batch_size;
    static constexpr size_t default_match_batch_size = traits::default_match_batch_size;
    
    static constexpr float default_scale_factor = traits::default_scale_factor;
    static constexpr float default_max_load_factor = traits::default_max_load_factor;
    static constexpr float default_grow_factor = traits::default_grow_factor;

    typedef detail::cacheline_bucket_t<key_type,value_type,cacheline_size> bucket_t;
    static const size_t bucket_size = bucket_t::bucket_size;

    static_assert( std::is_trivial<Key>::value, "Key type is not trivial");
    static_assert( std::is_trivial<Value>::value, "Value type is not trivial");
    static_assert(bucket_size > 0, "bucket_size must be non-zero");
    static_assert(sizeof(bucket_t) == cacheline_size, "sizeof(bucket_t) != cacheline_size");
    static_assert(alignof(bucket_t) == cacheline_size, "alignof(bucket_t) != cacheline_size");


    batch_unordered_multimap(
        float __max_load_factor = default_max_load_factor,
        float __grow_factor = default_grow_factor,
        size_t __insert_batch_size = default_insert_batch_size,
        size_t __match_batch_size = default_match_batch_size
        )
        : _max_load_factor(__max_load_factor), _grow_factor(__grow_factor)
        , _size(0), _max_size(0), _reserved_size(1)
        , _insert_batch_size(__insert_batch_size), _match_batch_size(__match_batch_size)
    {
        _insert_queue.resize(_insert_batch_size);
        _insert_queue_count = 0;
        _match_queue.resize(_match_batch_size);
        _match_queue_count = 0;
    }
    
    batch_unordered_multimap(const batch_unordered_multimap& ) = default;
    batch_unordered_multimap(      batch_unordered_multimap&&) = default;
    batch_unordered_multimap& operator=(const batch_unordered_multimap& ) = default;
    batch_unordered_multimap& operator=(      batch_unordered_multimap&&) = default;

    // read configuration
    bool   empty()             const { return _size == 0; }
    size_t size()              const { return _size; }
    size_t capacity()          const { return _max_size; }
    size_t bucket_count()      const { return _reserved_size; }
    float  load_factor()       const { return float(_size) / float(_reserved_size); }
    float  max_load_factor()   const { return _max_load_factor; }
    float  grow_factor()       const { return _grow_factor; }
    size_t insert_batch_size() const { return _insert_batch_size; }
    size_t match_batch_size()  const { return _match_batch_size; }
    
    // set new max_load_factor
    void max_load_factor(float ml)
    {
        // set max load
        _max_load_factor = ml;
        // recalculate _max_size
        _max_size = float(_hp.prime()) * float(bucket_size) * _max_load_factor;
        // changing _max_size may trigger a rehash (if autogrow is enabled)
        if (is_auto_grow && _size > _max_size)
            rehash( _grow_factor * std::max<float>( float(_reserved_size), float(_size)/_max_load_factor ));
    }
    
    void grow_factor(float gf)
    {
        _grow_factor = std::max<float>( gf, 1.01 );
    }
    
    // set new insert_batch_size
    void insert_batch_size(size_t ibs)
    {
        while (ibs < _insert_queue_count)
            process_insert_queue();
        _insert_batch_size = ibs;
        _insert_queue.resize(_insert_batch_size);
    }

    // set new match batch size
    void match_batch_size(size_t mbs)
    {
        if (mbs < _match_queue_count)
            throw std::runtime_error("batch_unordered_multimap::match_batch_size(): cannot shrink smaller than current match queue size");
        _match_batch_size = mbs;
        _match_queue.resize(_match_batch_size);
    }

    // reserve to store a given number of elements
    void reserve(size_t elements, double scale = default_scale_factor)
    {
        // lower bound scale by 1/_max_load_factor
        scale = std::max<float>( scale, 1.0f/_max_load_factor );
        // compute target number of buckets
        size_t buckets = float(elements) * scale / float(bucket_size);
        // reserve buckets if empty, otherwise rehash everything
        if (!empty())
            rehash(buckets);
        else
            _reserve(buckets);
    }
    
    // reserve a given number of buckets, multimap must be empty
    void _reserve(size_t buckets)
    {
        if (!empty())
            throw std::runtime_error("batch_unordered_multimap::_reserve(): multimap not empty");
        // find fast prime p >= buckets
        _hp = detail::get_hash_prime_ge(buckets);
        // resize map and set _max_size based on p
        _reserved_size = _hp.prime() * bucket_size;
        _max_size = float(_reserved_size) * _max_load_factor;
        _map.resize(_hp.prime()+1);
        // check proper alignment
        uintptr_t check_align = reinterpret_cast<uintptr_t>(&_map[0]);
        if (0 != (check_align & (cacheline_size-1)))
            throw std::runtime_error("batch_unordered_multimap::reserve(): aligned allocation failed");
        // clear map
        clear();
    }

    // resize multimap to a different number of buckets
    bool rehash(size_t buckets)
    {
        if (float(buckets)*_max_load_factor <= float(_size))
            return false;
        // clear insert queue
        finalize_insert();
        // create temporary multimap with same parameters
        batch_unordered_multimap tmp(_max_load_factor, _grow_factor, _insert_batch_size, _match_batch_size);
        // reserve given number of buckets
        tmp._reserve(buckets);
        // move match queue to tmp to ensure it remains intact at the end
        std::swap(_match_queue, tmp._match_queue);
        std::swap(_match_queue_count, tmp._match_queue_count);
        // insert all elements
        for (auto& B : _map)
        {
            for (size_t i = 0; i < B.size; ++i)
                tmp.queue_insert(B.keys[i], B.values[i]);
        }
        tmp.finalize_insert();
        // move assign data from tmp
        *this = std::move(tmp);
        return true;
    }

    // clear multimap and insert/match queues
    void clear()
    {
        _size = 0;
        // clearing entire bucket to 0 is quicker than only zeroing bucket_t.size
        if (!_map.empty())
            memset(&_map[0], 0, sizeof(bucket_t)*_map.size());
        // clear insert and match queues
        _insert_queue.resize(_insert_batch_size);
        _insert_queue_count = 0;
        _match_queue.resize(_match_batch_size);
        _match_queue_count = 0;
    }

    // compute uint64_t hash of key
    uint64_t hash(const key_type& k) const
    {
        return mccl::detail::hash(k);
    }

    // compute bucket from key via hash
    uint64_t bucket(const key_type& k) const
    {
        return _hp.mod( this->hash(k) );
    }


    void insert(const key_type& k, const value_type& v)
    {
        queue_insert(k,v);
    }
    
    bool queue_insert(const key_type& k, const value_type& v)
    {
        if (size() >= capacity())
        {
            if (!is_auto_grow)
                return false;
            rehash( float(_reserved_size) * _grow_factor );
        }
        // already increase size, since we always insert the element
        ++_size;
        // compute bucket
        uint64_t b = this->bucket(k);
        // prefetch target bucket
        __builtin_prefetch(&_map[b].size,1,0);
        // insert in queue
        auto& item = _insert_queue[_insert_queue_count++];
        item.key = k;
        item.value = v;
        item.bucket = b;
        // if our insert queue is big enough then process it
        if (__builtin_expect(_insert_queue_count == _insert_batch_size, 0))
            process_insert_queue();
        return true;
    }
    
    bool process_insert_queue()
    {
        do
        {
            // remember queue end, and reset _insert_queue_count
            size_t e = _insert_queue_count;
            _insert_queue_count = 0;
            for (size_t i = 0; i < e; ++i)
            {
                // process item
                auto& item = _insert_queue[i];
                uint64_t b = item.bucket;
                auto& B = _map[b];
                // check if cacheline bucket is already full
                if (__builtin_expect(B.size == bucket_size,0))
                {
                    // if full then requeue item and prefetch next cacheline bucket
                    if (++b == _hp.prime())
                        b = 0;
                    __builtin_prefetch(&_map[b].size,1,0);
                    _insert_queue[_insert_queue_count] = item;
                    _insert_queue[_insert_queue_count].bucket = b;
                    ++_insert_queue_count;
                    continue;
                }
                // if not full then insert item in bucket B
                auto j = B.size ++;
                B.keys[j] = item.key;
                B.values[j] = item.value;
            }
        // ensure that queue has actually decreased by at least 1
        } while (_insert_queue_count == _insert_batch_size);
        return _insert_queue_count == 0;
    }

    void finalize_insert()
    {
        while (!process_insert_queue())
            ;
    }


    template<typename FF>
    void queue_match(const key_type& k, uintptr_t aux_data, FF&& f)
    {
        // compute bucket
        uint64_t b = this->bucket(k);
        // prefetch target bucket
        __builtin_prefetch(&_map[b].size,0,0);
        // insert in queue
        auto& item = _match_queue[_match_queue_count++];
        item.key = k;
        item.aux_data = aux_data;
        item.bucket = b;
        // if our insert queue is big enough then process it
        if (__builtin_expect(_match_queue_count == _match_batch_size, 0))
            process_match_queue(f);
    }
    
    template<typename FF>
    bool process_match_queue(FF&& f)
    {
        do
        {
            // remember queue end, and reset _insert_queue_count
            size_t e = _match_queue_count;
            _match_queue_count = 0;
            for (size_t i = 0; i < e; ++i)
            {
                // process item
                auto& item = _match_queue[i];
                uint64_t b = item.bucket;
                auto& B = _map[b];
                if (__builtin_expect(B.size < bucket_size,1))
                {
                    for (size_t i = 0; i < B.size; ++i)
                    {
                        if (B.keys[i] == item.key)
                            f(item.aux_data, item.key, B.values[i]);
                    }
                } else
                {
                    for (size_t i = 0; i < bucket_size; ++i)
                    {
                        if (B.keys[i] == item.key)
                            f(item.aux_data, item.key, B.values[i]);
                    }
                    if (++b == _hp.prime())
                        b = 0;
                    __builtin_prefetch(&_map[b].size,0,0);
                    _match_queue[_match_queue_count] = item;
                    _match_queue[_match_queue_count].bucket = b;
                    ++_match_queue_count;
                }
            }
        // ensure that queue has actually decreased
        } while (_match_queue_count == _match_batch_size);
        return _match_queue_count == 0;
    }

    template<typename FF>
    void finalize_match(FF&& f)
    {
        while (!process_match_queue(f))
            ;
    }
    
private:
    float _max_load_factor, _grow_factor;
    size_t _size, _max_size, _reserved_size;
    size_t _insert_batch_size, _match_batch_size;
    hash_prime _hp;
    
    aligned_vector<bucket_t> _map;

    struct insert_item_t {
        key_type key;
        value_type value;
        uint64_t bucket;
    };
    aligned_vector<insert_item_t> _insert_queue;
    size_t _insert_queue_count;
    
    struct match_item_t {
        key_type key;
        uintptr_t aux_data;
        uint64_t bucket;
    };
    aligned_vector<match_item_t> _match_queue;
    size_t _match_queue_count;
};

MCCL_END_NAMESPACE

#endif
