// config
#include <mccl/config/config.hpp>
#include <mccl/tools/unordered_multimap.hpp>
#include <mccl/contrib/memory_usage.hpp>
#include <mccl/contrib/program_options.hpp>

/* just place parallel-hashmap dir and robin_hood.h in mccl/contrib
   and uncomment these lines as well as those in main() to also benchmark these unordered_maps
*/
//#include <mccl/contrib/parallel-hashmap/parallel_hashmap/phmap.h>
//#include <mccl/contrib/robin_hood.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <set>
#include <utility>
#include <random>
#include <chrono>

#include "test_utils.hpp"

using namespace mccl;
namespace po = program_options;

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
    size_t _dummy;

    void reserve(size_t n, double) { _map.reserve(n); }

    size_t size() const { return _map.size(); }

    void clear() {}

    void prefetch(uint64_t k) { _dummy += _map.bucket_size( _map.bucket(k) ); }

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
    void operator()(uintptr_t, uint64_t, uint64_t v) { c += v; }
};

template<typename HT>
 __attribute__ ((noinline)) int test_hash_table(const std::string& HT_name, double sf)
{
    size_t before_mem = getCurrentRSS();
    HT h;
    h.reserve(test_numbers.size(), sf);
    auto start = bench_clock_t::now();
    h.clear();
    auto end0 = bench_clock_t::now();
    for (auto& n : test_numbers)
        h.insert(n, 1);
    auto end1 = bench_clock_t::now();
    counter_t c;
    for (auto& n : test_numbers)
        h.match(n, c);
    auto end2 = bench_clock_t::now();
    double ms0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0-start).count();
    double ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1-end0).count();
    double ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2-end1).count();
    size_t used_mem = getCurrentRSS();
    size_t hs = h.size();
    std::cout << std::setw(30) << HT_name << " : "
        << std::setw(8) << ms0 << " | "
        << std::setw(9) << ms1 << " | "
        << std::setw(8) << ms2 << " | "
        << std::setw(11) << hs << " | "
        << std::setw(11) << c.c << " | "
        << std::setw(11) << double(test_numbers.size())*double(c.c) * (ms0+ms1+ms2) * sf << " | "
        << ((used_mem - before_mem)>>20) << "MB (" << (before_mem>>20) << "MB->" << (used_mem>>20) << "MB)"
        << std::endl;
    return 0;
}

template<typename HT>
 __attribute__ ((noinline)) int test_hash_table_prefetch(const std::string& HT_name, double sf)
{
    size_t before_mem = getCurrentRSS();
    HT h;
    h.reserve(test_numbers.size(), sf);
    auto start = bench_clock_t::now();
    h.clear();
    auto end0 = bench_clock_t::now();
    for (size_t i = 0; i+127 < test_numbers.size(); i += 128)
    {
        for (size_t j = 0; j < 128; ++j)
            h.prefetch( test_numbers[i+j] );
        for (size_t j = 0; j < 128; ++j)
            h.insert( test_numbers[i+j], 1 );
    }
    auto end1 = bench_clock_t::now();
    counter_t c;
    for (size_t i = 0; i+127 < test_numbers.size(); i += 128)
    {
        for (size_t j = 0; j < 128; ++j)
            h.prefetch( test_numbers[i+j] );
        for (size_t j = 0; j < 128; ++j)
            h.match( test_numbers[i+j], c );
    }
    auto end2 = bench_clock_t::now();
    double ms0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0-start).count();
    double ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1-end0).count();
    double ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2-end1).count();
    size_t used_mem = getCurrentRSS();
    size_t hs = h.size();
    std::cout << std::setw(30) << HT_name << " : "
        << std::setw(8) << ms0 << " | "
        << std::setw(9) << ms1 << " | "
        << std::setw(8) << ms2 << " | "
        << std::setw(11) << hs << " | "
        << std::setw(11) << c.c << " | "
        << std::setw(11) << double(test_numbers.size())*double(c.c) * (ms0+ms1+ms2) * sf << " | "
        << ((used_mem - before_mem)>>20) << "MB (" << (before_mem>>20) << "MB->" << (used_mem>>20) << "MB)"
        << std::endl;
    return 0;
}

template<typename HT>
 __attribute__ ((noinline)) int test_queued_hash_table(const std::string& HT_name, double sf)
{
    size_t before_mem = getCurrentRSS();
    
    HT h;
    h.reserve(test_numbers.size(), sf);

    auto start = bench_clock_t::now();
    h.clear();
    h.reserve(test_numbers.size(), sf);
    auto end0 = bench_clock_t::now();
    for (auto& n : test_numbers)
        h.queue_insert(n, 1);
    while (!h.process_insert_queue())
    {}
    auto end1 = bench_clock_t::now();
    counter_t c;
    for (auto& n : test_numbers)
        h.queue_match(n, 0, c);
    while (!h.process_match_queue(c))
    {}
    auto end2 = bench_clock_t::now();

    double ms0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0-start).count();
    double ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1-end0).count();
    double ms2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2-end1).count();
    size_t used_mem = getCurrentRSS();
    size_t hs = h.size();
    std::cout << std::setw(30) << HT_name << " : "
        << std::setw(8) << ms0 << " | "
        << std::setw(9) << ms1 << " | "
        << std::setw(8) << ms2 << " | "
        << std::setw(11) << hs << " | "
        << std::setw(11) << c.c << " | "
        << std::setw(11) << double(test_numbers.size())*double(c.c) * (ms0+ms1+ms2) * sf << " | "
        << ((used_mem - before_mem)>>20) << "MB (" << (before_mem>>20) << "MB->" << (used_mem>>20) << "MB)"
        << std::endl;
    return 0;
}

int main(int argc, char** argv)
{
    po::options_description allopts;
    allopts.add_options()
        ("bench,b", "Benchmark unordered_multimaps")
        ("help,h",  "Show options")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, allopts), vm);
    po::notify(vm);
    if (vm.count("help"))
    {
        std::cout << allopts << std::endl;
        return 0;
    }
    
    // generate samples to test on
    std::random_device rd;
    std::mt19937_64 mt(rd());
    for (size_t i = 0; i < (1<<16); ++i)
        test_numbers.push_back( mt() );
    
    // retrieve all hash primes
    uint64_t smallest_hash_prime = detail::get_hash_prime_ge(0).prime();
    uint64_t largest_hash_prime  = detail::get_hash_prime_lt(~uint64_t(0)).prime();
    
    std::vector<uint64_t> hash_primes;
    uint64_t p = smallest_hash_prime;
    while (true)
    {
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

    if (vm.count("bench"))
    {
        std::cout << "\n====== Benchmark simple_hash_table vs cacheline_hash_table" << std::endl;
        test_numbers.clear();

        for (size_t b = 26; b <= 26; ++b)
        {
            for (double sf = 1.1; sf <= 2.05; sf += 0.1)
            {
                while (test_numbers.size() < (1ULL<<b))
                    test_numbers.push_back( mt() );

                std::cout << "=== Collection size: " << test_numbers.size() << " reserve scale factor: sf=" << sf << std::endl;
                std::cout << std::setw(30) << "HASH TABLE" << " : "
                    << std::setw(8) << "clear ms" << " | "
                    << std::setw(9) << "insert ms" << " | "
                    << std::setw(8) << "match ms" << " | "
                    << std::setw(11) << "stored" << " | "
                    << std::setw(11) << "retrieved" << " | "
                    << std::setw(11) << "rating" << " | "
                    << std::setw(15) << "memory usage"
                    << std::endl;

                test_hash_table< cacheline_unordered_multimap<uint64_t,uint8_t> >          ("cacheline_unordered_multimap", sf);

                test_hash_table_prefetch< cacheline_unordered_multimap<uint64_t,uint8_t> > ("cacheline_unordered_multimap", sf);

                test_queued_hash_table< batch_unordered_multimap<uint64_t,uint8_t> >       ("batch_unordered_multimap", sf);
        
                // only test other unordered_map once as last
                // parameter sf is actually unused for these
                if (sf >= 1.95)
                {
                    test_hash_table< ht_wrapper<std::unordered_multimap<uint64_t,uint8_t>> >          ("unordered_multimap", sf);
                    test_hash_table_prefetch< ht_wrapper<std::unordered_multimap<uint64_t,uint8_t>> > ("unordered_multimap", sf);
//                    test_hash_table< ht_wrapper<phmap::flat_hash_map<uint64_t,uint8_t>> >             ("phmap::flat_hash_map", sf);
//                    test_hash_table< ht_wrapper<robin_hood::unordered_flat_map<uint64_t,uint8_t>> >   ("rh unordered_flat_map", sf);
                }
            }
        }
    }

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
