#ifndef MCCL_CORE_RANDOM_HPP
#define MCCL_CORE_RANDOM_HPP

#include <mccl/config/config.hpp>

#include <random>
#include <array>

MCCL_BEGIN_NAMESPACE

struct mccl_base_random_generator
{
    mccl_base_random_generator()
    {
        seed();
    }
    mccl_base_random_generator(uint64_t s)
    {
        seed(s);
    }
    void _reseed()
    {
        std::seed_seq _seed(seedarray.begin(), seedarray.end());
        rnd.seed(_seed);
    }
    void seed()
    {
        std::random_device rnddev;
        for (unsigned i = 0; i < seedarray.size(); ++i)
            seedarray[i] = rnddev();
        _reseed();
    }
    void seed(uint64_t s)
    {
        seedarray[0] = uint32_t(s);
        seedarray[1] = uint32_t(s>>32);
        _reseed();
    }
    uint64_t get_seed() const
    {
        return uint64_t(seedarray[0]) | (uint64_t(seedarray[1])<<32);
    }
    void operator()(uint64_t& word)
    {
        word = rnd();
    }
    uint64_t operator()()
    {
        return rnd();
    }
    std::array<uint32_t,2> seedarray;
    std::mt19937_64 rnd;
};

MCCL_END_NAMESPACE

#endif
