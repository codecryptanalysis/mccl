// config
#include <mccl/config/config.hpp>
#include <mccl/tools/hash_primes.hpp>
#include <mccl/contrib/memory_usage.hpp>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <utility>
#include <random>
#include <chrono>

#include "test_utils.hpp"

using namespace mccl;

typedef std::conditional<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>::type bench_clock_t;

std::vector<uint64_t> test_numbers;

int test_prime(uint64_t p)
{
    detail::hash_prime hp = detail::get_hash_prime_ge(p);
    
    int status = 0;
    status |= (hp.prime()  != p   );
    status |= (hp.mod(1)   != 1   );
    status |= (hp.mod(p-1) != p-1 );
    status |= (hp.mod(p)   != 0   );
    for (auto n : test_numbers)
        status |= (hp.mod(n) != (n%p));
        
    return status;
}

template<typename HT>
struct ht_wrapper
{
    HT _map;
    void reserve(size_t n, double sf) { _map.reserve(n); }
    void insert(uint64_t k, uint64_t v) { _map.insert({k,v}); }
    
    template<typename F>
    void match(uint64_t k, F&& f)
    {
        auto it = _map.find(k);
        if (it != _map.end())
            f(it->second);
    }
};

struct counter_t
{
    uint64_t c;
    counter_t(): c(0) {}
    void operator()(uint64_t v) { c += v; }
};

template<typename HT>
int test_hash_table(const std::string& HT_name, double sf)
{
    size_t before_mem = getCurrentRSS();
    
    HT h;
    h.reserve(test_numbers.size(), sf);

    auto start1 = bench_clock_t::now();
    for (auto& n : test_numbers)
        h.insert(n, 1);
    auto end1 = bench_clock_t::now();
    counter_t c;
    for (auto& n : test_numbers)
        h.match(n, c);
    auto end2 = bench_clock_t::now();
    double ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count();
    double ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2-end1).count();
    size_t used_mem = getCurrentRSS();
    std::cout << HT_name << ": " << ms1 << " \t" << ms2 << " \t" << c.c << " \t" << double(test_numbers.size())*double(c.c) * (ms1+ms2) * sf << " \t mem: " << ((used_mem - before_mem)>>20) << "MB (" << (before_mem>>20) << "MB->" << (used_mem>>20) << "MB)" << std::endl;
    return 0;
}

int main(int, char**)
{
    // generate samples to test on
    std::random_device rd;
    std::mt19937_64 mt(rd());
    for (size_t i = 0; i < (1<<20); ++i)
        test_numbers.push_back( mt() );
    
    // retrieve all hash primes
    uint64_t smallest_hash_prime = detail::get_hash_prime_ge(0).prime();
    uint64_t largest_hash_prime  = detail::get_hash_prime_lt(~uint64_t(0)).prime();
    
    std::vector<uint64_t> hash_primes;
    uint64_t p = smallest_hash_prime;
//    std::cout << "Collecting hash primes: ";
    while (true)
    {
//        std::cout << p << " " << std::flush;
        hash_primes.push_back(p);
        // if reached end then break, otherwise it'll throw on the next get
        if (p == largest_hash_prime)
            break;
        // get next hash prime
        p = detail::get_hash_prime_gt(p).prime();
    }
    std::cout << std::endl << "Found " << hash_primes.size() << " hash primes in collection." << std::endl;
    
    // test all hash primes
    int status = 0;
    for (auto p : hash_primes)
        status |= test_prime(p);

#if 0
    std::cout << "\n====== Benchmark simple_hash_table vs cacheline_hash_table" << std::endl;
    test_numbers.clear();

    for (size_t b = 26; b <= 26; ++b)
    for (double sf = 1.1; sf <= 2.05; sf += 0.1)
    {
        while (test_numbers.size() < (1ULL<<b))
            test_numbers.push_back( mt() );

        std::cout << "=== Collection size: " << test_numbers.size() << " reserve scale factor: sf=" << sf << std::endl;
        std::cout << "   HASH TABLE        : ST ms \tL ms \tstored   \trating      \tmemory usage" << std::endl;
        test_hash_table< simple_hash_table<uint64_t,uint64_t> >             ("simple_hash_table    ", sf);
        test_hash_table< cacheline_hash_table<uint64_t,uint64_t> >          ("cacheline_hash_table ", sf);

        // only test std::unordered_map once as last
        // parameter sf is actually unused for std::unordered_map
        if (sf >= 1.95)
            test_hash_table< ht_wrapper<std::unordered_map<uint64_t,uint64_t>> >("unordered_map        ", sf);
    }
#endif

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
