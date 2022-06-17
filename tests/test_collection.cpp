// config
#include <mccl/config/config.hpp>
#include <mccl/core/collection.hpp>
#include <mccl/contrib/memory_usage.hpp>
#include <mccl/contrib/program_options.hpp>

#include <cstring>
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

int test_allocator()
{
    std::vector<void*> pages;

    for (std::size_t s = 1<<20; s <= 256<<20; s<<=1)
    {
        for (std::size_t a = 64; a <= 4096; a <<= 1)
        {
            std::size_t basemem = getCurrentRSS();
            std::cout << s << " " << a << ": " << basemem << std::endl;

            void* p1 = detail::mccl_aligned_alloc(s, a);
            std::cout << "\t p 1: " << uintptr_t(p1) << std::endl;
            memset(p1, 1, s);
            std::size_t afteralloc1 = getCurrentRSS();

            void* p2 = detail::mccl_aligned_alloc(s, a);
            std::cout << "\t p 2: " << uintptr_t(p2) << " " << intptr_t(p2)-intptr_t(p1) << std::endl;
            memset(p2, 2, s);
            std::size_t afteralloc2 = getCurrentRSS();

            detail::mccl_aligned_free(p1);
            std::size_t afterfree2 = getCurrentRSS();

            detail::mccl_aligned_free(p2);
            std::size_t afterfree1 = getCurrentRSS();

            std::cout << "\t alloc 1: " << afteralloc1 << " (+=" << afteralloc1-basemem << ")" << std::endl;
            std::cout << "\t alloc 2: " << afteralloc2 << " (+=" << afteralloc2-afteralloc1 << ")" << std::endl;
            std::cout << "\t free  2: " << afterfree2 << " (-=" << afteralloc2-afterfree2 << ")" << std::endl;
            std::cout << "\t free  1: " << afterfree1 << " (-=" << afterfree2-afterfree1 << ")" << std::endl;
        }
    }
    return 0;        
}

int main(int argc, char** argv)
{
    po::options_description allopts;
    allopts.add_options()
        ("bench,b", "Benchmark allocator")
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

    int status = 0;
    status |= test_allocator();

    if (vm.count("bench"))
    {
        std::cout << "\n====== Benchmark " << std::endl;
    }

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
