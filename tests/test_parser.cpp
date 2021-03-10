#include <mccl/tools/parser.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <utility>

//using namespace mccl;

int main(int, char**)
{
    int status = 0;

    Parser<uint64_t> parse;
    parse.load_file("data/Goppa_1101.txt");


    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
