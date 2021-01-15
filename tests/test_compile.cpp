#include <mccl/config/config.hpp>
#include <mccl/core/matrix.hpp>
#include <mccl/core/birthday.hpp>

#include <mccl/contrib/string_algo.hpp>
#include <mccl/contrib/thread_pool.hpp>
#include <mccl/contrib/parallel_algorithms.hpp>
#include <mccl/contrib/json.hpp>
#include <mccl/contrib/program_options.hpp>

#include <iostream>

#include "test_utils.h"

int main(int, char**)
{
    int status = 0;
    
    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
