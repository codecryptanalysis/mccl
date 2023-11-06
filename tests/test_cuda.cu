// config
#include <mccl/config/config.hpp>

// core header files
#include <mccl/core/matrix.hpp>
#include <mccl/core/matrix_m4ri.hpp>

// algorithm header files
#include <mccl/algorithm/decoding.hpp>

// tools header files
//#include <mccl/tools/parser.hpp>

// contrib header files
#include <mccl/contrib/string_algo.hpp>
#include <mccl/contrib/thread_pool.hpp>
#include <mccl/contrib/parallel_algorithms.hpp>
#include <mccl/contrib/json.hpp>
#include <mccl/contrib/program_options.hpp>

#include <iostream>
#include <vector>
#include <set>
#include <utility>

#include <cuda.h>

#include "test_utils.hpp"

int main(int, char**)
{
    int status = 0;
    int deviceCount = 0;
    if (cudaSuccess != cudaGetDeviceCount(&deviceCount))
    {
	LOG_CERR("CUDA: failed cudaGetDeviceCount");
	status = 1;
    }
    if (deviceCount == 0)
        LOG_CERR("There is no device supporting CUDA");
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, dev))
	{
		LOG_CERR("CUDA: failed cudaGetDeviceProperties");
		status = 1;
		continue;
	}
        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                LOG_CERR("There is no device supporting CUDA.");
            }
            else if (deviceCount == 1)
            {
                LOG_CERR("There is 1 device supporting CUDA.");
            }
            else
            {
                LOG_CERR("There are " << deviceCount << " devices supporting CUDA");
            }
        }
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Major revision number:                         %d\n",
               deviceProp.major);
        printf("  Minor revision number:                         %d\n",
               deviceProp.minor);
        printf("  Total amount of global memory:                 %u bytes\n",
               deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
        printf("  Number of multiprocessors:                     %d\n",
               deviceProp.multiProcessorCount);
        printf("  Number of cores:                               %d\n",
               8 * deviceProp.multiProcessorCount);
#endif
        printf("  Total amount of constant memory:               %u bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %u bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %u bytes\n",
               deviceProp.memPitch);
        printf("  Texture alignment:                             %u bytes\n",
               deviceProp.textureAlignment);
        printf("  Clock rate:                                    %.2f GHz\n",
               deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
        printf("  Concurrent copy and execution:                 %s\n",
               deviceProp.deviceOverlap ? "Yes" : "No");
#endif
    }
    if (status)
    {
	LOG_CERR("Tests failed.");
    }
    else
    {
	LOG_CERR("All tests passed.");
    }
    return status;
}
