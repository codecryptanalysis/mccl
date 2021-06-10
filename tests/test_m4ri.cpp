#include <mccl/config/config.hpp>

#include <mccl/core/matrix.hpp>
#include <mccl/core/matrix_m4ri.hpp>

#include <iostream>
#include <vector>
#include <set>
#include <utility>

#include "test_utils.hpp"

using namespace mccl;

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
    mat m(16,16);
    vec v(16);
    vec_view vref(v);
    std::cout << m << v;

    m4ri_transpose(m, m);
    m4ri_add(m, m, m);
    m4ri_mul_naive(m, m, m);
    m4ri_addmul_naive(m, m, m);
    m4ri_gauss_delayed(m);
    m4ri_echelonize_naive(m);
    m4ri_invert_naive(m, m, m);

    m4ri_echelonize(m);
    m4ri_echelonize_pluq(m);
    m4ri_echelonize_m4ri(m);
    m4ri_mul(m, m, m);
    m4ri_addmul(m, m, m);
}

int test_m4ri(size_t r = 512, size_t c = 512)
{
    int status = 0;
    
    mat m1(r,c);
    mat m2(r,c);

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
