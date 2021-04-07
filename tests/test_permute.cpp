#include <mccl/config/config.hpp>

#include <mccl/tools/parser.hpp>
#include <mccl/core/matrix_permute.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <utility>

using namespace mccl;

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

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
