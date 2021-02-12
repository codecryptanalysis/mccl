#include <mccl/config/config.hpp>

#include <mccl/core/matrix_detail.hpp>
#include <mccl/core/matrix.hpp>
#include <mccl/core/matrix_m4ri.hpp>

#include <iostream>
#include <vector>
#include <set>
#include <utility>

#include "test_utils.hpp"

using namespace mccl;

typedef mccl::matrix_ref_t<uint64_t> matrixref;
typedef mccl::matrix_ptr_t<uint64_t> matrixptr;
typedef mccl::vector_ref_t<uint64_t> vectorref;
typedef mccl::vector_ptr_t<uint64_t> vectorptr;
typedef mccl::matrix_t<uint64_t> mat_t;
typedef mccl::vector_t<uint64_t> vec_t;
typedef mccl::detail::matrix_base_ref_t<uint64_t> base_t;

int test_bool(bool val, const std::string& errmsg = "error")
{
    if (val)
       return 0;
    LOG_CERR(errmsg);
    return -1;
}

void compile_test()
{
    // don't call
    mat_t mat(16,16);
    vec_t vec(16);
    vectorref vref(vec);
    std::cout << mat << vec;

    m4ri_transpose(mat, mat);
    m4ri_add(mat, mat, mat);
    m4ri_mul_naive(mat, mat, mat);
    m4ri_addmul_naive(mat, mat, mat);
    m4ri_gauss_delayed(mat);
    m4ri_echelonize_naive(mat);
    m4ri_invert_naive(mat, mat, mat);

    m4ri_echelonize(mat);
    m4ri_echelonize_pluq(mat);
    m4ri_echelonize_m4ri(mat);
    m4ri_mul(mat, mat, mat);
    m4ri_addmul(mat, mat, mat);
}

int test_m4ri(size_t r = 512, size_t c = 512)
{
    int status = 0;
    
    mat_t m1(r,c);
    mat_t m2(r,c);

    fillrandom(m1);

    std::cout << r << std::endl;
    m4ri_transpose(m2, m1);
    for (size_t i = 0; i < m1.rows() && status == 0; ++i)
        for (size_t j = 0; j < m1.columns() && status == 0; ++j)
            status |= test_bool(m1(i,j) == m2(j,i), "transpose failed");

    return status;
}



int main(int, char**)
{
    int status = 0;

    for (size_t i = 1; i <= 4*64; ++i)
        status |= test_m4ri(i,i);

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
