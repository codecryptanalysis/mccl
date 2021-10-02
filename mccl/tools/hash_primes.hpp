#ifndef MCCL_TOOLS_HASH_PRIMES_HPP
#define MCCL_TOOLS_HASH_PRIMES_HPP

#include <mccl/config/config.hpp>

MCCL_BEGIN_NAMESPACE

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

    // returns n % prime, using 2 multiplications, 1 shift, 1 subtraction
    inline uint64_t mod(uint64_t n)
    {
        // get the top 64-bit of the 128-bit multiplication n * _muldiv
        uint64_t div = (uint128_t(n) * _muldiv) >> 64;
        // shift right by _shift
        div >>= _shift;
        // now div==n/p, so n%p = n - div*p
        return n - div * _prime;
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

MCCL_END_NAMESPACE

#endif
