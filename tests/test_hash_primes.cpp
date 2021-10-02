// config
#include <mccl/config/config.hpp>

#include <mccl/tools/hash_primes.hpp>

#include <iostream>
#include <vector>
#include <set>
#include <utility>
#include <random>

#include "test_utils.hpp"

using namespace mccl;

std::vector<uint64_t> test_numbers;

int test_prime(uint64_t p)
{
    hash_prime hp = get_hash_prime_ge(p);
    
    int status = 0;
    status |= (hp.prime()  != p   );
    status |= (hp.mod(1)   != 1   );
    status |= (hp.mod(p-1) != p-1 );
    status |= (hp.mod(p)   != 0   );
    for (auto n : test_numbers)
        status |= (hp.mod(n) != (n%p));
        
    return status;
}

int main(int, char**)
{
    // generate samples to test on
    std::random_device rd;
    std::mt19937_64 mt(rd());
    for (size_t i = 0; i < (1<<20); ++i)
        test_numbers.push_back( mt() );
    
    // retrieve all hash primes
    uint64_t smallest_hash_prime = get_hash_prime_ge(0).prime();
    uint64_t largest_hash_prime  = get_hash_prime_lt(~uint64_t(0)).prime();
    
    std::vector<uint64_t> hash_primes;
    uint64_t p = smallest_hash_prime;
    std::cout << "Collecting hash primes: ";
    while (true)
    {
        std::cout << p << " " << std::flush;
        hash_primes.push_back(p);
        // if reached end then break, otherwise it'll throw on the next get
        if (p == largest_hash_prime)
            break;
        // get next hash prime
        p = get_hash_prime_gt(p).prime();
    }
    std::cout << std::endl << "Found " << hash_primes.size() << " hash primes in collection." << std::endl;
    
    // test all hash primes
    int status = 0;
    for (auto p : hash_primes)
        status |= test_prime(p);
    
    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
