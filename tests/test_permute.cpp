#include <mccl/config/config.hpp>

#include <mccl/tools/parser.hpp>
#include <mccl/core/matrix_permute.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <utility>

using namespace mccl;

std::size_t binomial(std::size_t k, std::size_t N)
{
    if (k > N) return 0;
    std::size_t r = 1;
    if (k > N-k)
        k = N-k;
    for (unsigned i = 0; i < k; ++i)
    {
        r *= (N-i);
        r /= (i+1);
    }
    return r;
}

int test_enum(unsigned totalrows, unsigned minsumsize, unsigned maxsumsize)
{
//    std::cout << totalrows << " " << minsumsize << " " << maxsumsize << std::endl;
    matrix_t<uint64_t> m(totalrows, totalrows);
    for (unsigned i = 0; i < totalrows; ++i)
        m.bitset(i,i);
        
    std::vector<std::size_t> counts(maxsumsize+1, 0);
    
    matrix_enumeraterows_t<uint64_t> rowenum(m, maxsumsize, minsumsize);
    do {
        rowenum.compute();
//        std::cout << rowenum.result() << std::endl;
        
        // check result        
        unsigned s = 0;
        for (unsigned c = 0; c < totalrows; ++c)
        {
            if (rowenum.result()[c])
            {
                if (s < rowenum.selectionsize() && c == rowenum.selection()[s])
                    ++s;
                else
                    return 1;
            }
        }
        if (s != rowenum.selectionsize())
            return 1;
        // count results
        ++counts[rowenum.selectionsize()];
    } while (rowenum.next());
    
    // compute pascal triangle
    for (unsigned j = minsumsize; j <= maxsumsize; ++j)
    {
        if (counts[j] != binomial(j,totalrows))
            return 1;
//        std::cout << " " << j << " " << totalrows << " " << counts[j] << " " << binomial(j,totalrows) << std::endl;
    }
    
    return 0;
}

int main(int, char**)
{
    int status = 0;

    Parser<uint64_t> parse;
    status |= !parse.load_file("./tests/data/Goppa_197.txt");

    auto H = parse.get_H();
    auto S = parse.get_S();
    size_t n = parse.get_n();
    size_t k = parse.get_k();
    size_t w = parse.get_w();

    std::vector<size_t> rowweights(n-k);
    for( size_t r = 0; r < n-k; r++)
        rowweights[r] = hammingweight(H[r]);
    auto total_hw = hammingweight(H);

    matrix_permute_t<uint64_t> permutator(H);
    permutator.random_permute(0,n,n);

    status |= (total_hw != hammingweight(H));
    for( size_t r = 0; r < n-k; r++)
        status |= (rowweights[r] != hammingweight(H[r]));

    status |= test_enum(63,1,1);
    status |= test_enum(63,2,2);
    status |= test_enum(63,3,3);
    status |= test_enum(63,1,3);

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
