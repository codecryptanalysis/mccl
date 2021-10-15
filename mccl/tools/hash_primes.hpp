#ifndef MCCL_TOOLS_HASH_TABLE_HPP
#define MCCL_TOOLS_HASH_TABLE_HPP

#include <mccl/config/config.hpp>

#include <mccl/tools/aligned_vector.hpp>

#include <vector>
#include <cstdlib>
#include <memory>

MCCL_BEGIN_NAMESPACE

namespace detail
{

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


// 406 hash_prime values <= 2^50 of varying bitsizes have been precomputed
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

} // namespace detail



/** very simple hash table **/
// each bucket stores a single element but overflows in subsequent buckets by a maximum of MaxLoop
// usage:
//  first reserve() / clear() to setup / clear hashmap for the target number of elements
//  then fill with insert(k,v) with elements
//   - duplicate keys will be stored
//   - the hash table will only check MaxLoop items to find an empty position
//   - for MaxLoop == 1 this means a fixed position with at most 1 key value per bucket
//  then lookup keys with elements: match(k, f)
//  will search MaxLoop positions starting at bucket(k) to find elements with that key
//   - for each matching key (may be more than 1) with value v it will call f(v)
//   - at the first empty position it aborts the search
template<typename Key = uint64_t, typename Value = uint64_t, size_t MaxLoop = 16>
class simple_hash_table
{
public:
    typedef Key key_type;
    typedef Value value_type;
    // we don't use std::pair<key_type,value_type> to suppress a compiler warning when using memset
    struct pair_type
    {
        key_type first;
        value_type second;
    };
    typedef detail::hash_prime hash_prime;
    
    simple_hash_table()
    {
        memset(&_empty_key, 0xFF, sizeof(key_type));
    }
    
    void reserve(size_t n, double scale = 2.0f)
    {
        size_t s = double(n) * scale;
        _hp = detail::get_hash_prime_ge(s);
        _map.resize(_hp.prime());
        clear();
    }
    
    void clear()
    {
        memset(&_map[0], 0xFF, sizeof(pair_type)*_hp.prime());
    }
    
    uint64_t hash(const key_type& k) const
    {
        return mccl::detail::hash(k);
    }
    uint64_t bucket(const key_type& k) const
    {
        return _hp.mod( this->hash(k) );
    }
    
    bool insert(const key_type& k, const value_type& v)
    {
        // compute initial bucket index to start search
        uint64_t h = bucket(k);
        // check at most MaxLoop positions
        for (size_t i = 0; i < MaxLoop; ++i)
        {
            // store at the first empty position found
            if (_map[h].first == _empty_key)
            {
                _map[h].first = k;
                _map[h].second = v;
                return true;
            }
            // increase h mod p
            if (++h >= _hp.prime())
                h -= _hp.prime();
        }
        return false;
    }
    
    template<typename F>
    void match(const key_type& k, F&& f) const
    {
        // compute intial bucket index to start search
        uint64_t h = bucket(k);
        for (size_t i = 0; i < MaxLoop; ++i)
        {
            // for every match we call f with the value
            if (_map[h].first == k)
                f(_map[h].second);
            // we can stop at the first empty position found
            else if (_map[h].first == _empty_key)
                return;
            // increase h mod p
            if (++h >= _hp.prime())
                h -= _hp.prime();
        }        
    }
    
private:
    key_type _empty_key;
    hash_prime _hp;
    std::vector< pair_type > _map;
};





/** cacheline hash table for ISD **/
// similar to the very simple hash table above
// except buckets are aligned to cachelines and contain as many elements as possible in a cacheline
// buckets overflow in subsequent buckets, by a maximum of ceil(MaxLoop/BucketSize)
// usage:
//  first reserve() / clear() to setup / clear hashmap for the target number of elements
//  then fill with insert(k,v) with elements
//   - duplicate keys will be stored
//   - the hash table will only check MaxLoop items to find an empty position
//  then lookup keys with elements: match(k, f)
//  will search MaxLoop positions starting at bucket(k) to find elements with that key
//   - for each matching key (may be more than 1) with value v it will call f(v)
//   - at the first empty position it aborts the search
template<typename Key = uint64_t, typename Value = uint64_t, size_t MaxLoop = 16>
class cacheline_hash_table
{
public:
    typedef Key key_type;
    typedef Value value_type;
    typedef std::pair<Key,Value> pair_type;
    typedef detail::hash_prime hash_prime;

    static const size_t cacheline_size = 64;
    static const size_t bucket_size = cacheline_size / sizeof(pair_type);
    static const size_t max_bucket_loop = (MaxLoop + bucket_size - 1) / bucket_size;

    static_assert(bucket_size > 0, "bucket_size must be non-zero");

    struct alignas(cacheline_size) bucket_t
    {
        pair_type a[bucket_size];

              pair_type& operator[](size_t i)       { return a[i]; }
        const pair_type& operator[](size_t i) const { return a[i]; }
    };
    static_assert(sizeof(bucket_t) == cacheline_size, "sizeof(bucket_t) != cacheline_size");

    cacheline_hash_table()
    {
        memset(&_empty_key, 0xFF, sizeof(key_type));
        _size = 0;
        _max_size = 0;
    }

    void reserve(size_t n, double scale = 2.0f, double max_fill_ratio = 0.9f)
    {
        if (scale < 1.0f / max_fill_ratio)
            scale = 1.0f / max_fill_ratio;
        size_t s = double(n) * scale / double(bucket_size);
        _hp = detail::get_hash_prime_ge(s);
        _max_size = double(_hp.prime()) * bucket_size * max_fill_ratio;

        _map.resize(_hp.prime()+1);
        
        // check proper alignment
        uintptr_t check_align = reinterpret_cast<uintptr_t>(&_map[0]);
        if (0 != (check_align & (cacheline_size-1)))
            throw std::runtime_error("cacheline_hash_table::reserve(): aligned allocation failed");
        clear();
    }

    void clear()
    {
        memset(&_map[0], 0xFF, sizeof(bucket_t)*_hp.prime());
        _size = 0;
    }

    size_t capacity() const { return _max_size; }
    size_t size() const { return _size; }

    uint64_t hash(const key_type& k) const
    {
        return mccl::detail::hash(k);
    }
    uint64_t bucket(const key_type& k) const
    {
        return _hp.mod( this->hash(k) );
    }
    
    bool insert(const key_type& k, const value_type& v)
    {
        if (size() >= capacity())
            return false;
        // compute initial bucket index to start search
        uint64_t h = bucket(k);
        // check at most MaxLoop positions
        for (size_t i = 0; i < max_bucket_loop; ++i)
        {
            auto& B = _map[h];

            // only look at bucket if it isn't full already
            if (B[bucket_size-1].first == _empty_key)
            {
                for (size_t j = 0; j < bucket_size; ++j)
                {
                    // store at the first empty position found
                    if (B[j].first == _empty_key)
                    {
                        B[j].first = k;
                        B[j].second = v;
                        ++_size;
                        return true;
                    }
                }
            }
            // increase h mod p
            if (++h >= _hp.prime())
                h -= _hp.prime();
        }
        return false;
    }
    
    template<typename F>
    void match(const key_type& k, F&& f) const
    {
        // compute intial bucket index to start search
        uint64_t h = bucket(k);
        for (size_t i = 0; i < max_bucket_loop; ++i)
        {
            const auto& B = _map[h];
            for (size_t j = 0; j < bucket_size; ++j)
            {
                // for every match we call f with the value
                if (B[j].first == k)
                    f(B[j].second);
                // we can stop at the first empty position found
                else if (B[j].first == _empty_key)
                    return;
            }
            // increase h mod p
            if (++h >= _hp.prime())
                h -= _hp.prime();
        }
    }
    
private:
    size_t _size, _max_size;
    key_type _empty_key;
    hash_prime _hp;

    aligned_vector<bucket_t> _map;
    // C++11 std::vector doesn't guarantee proper cacheline alignment
//    bucket_t* _map;
//    void* _map2;
};

MCCL_END_NAMESPACE

#endif
