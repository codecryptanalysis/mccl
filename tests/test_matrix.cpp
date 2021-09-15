#include <mccl/config/config.hpp>

#include <mccl/core/matrix.hpp>
#include <mccl/core/matrix_algorithms.hpp>

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

int test_transpose(size_t r = 512, size_t c = 512)
{
    mat m1(r , c+1);

    fillrandom(m1);
    mat m2 = m_transpose(m1);

    int status = 0;
    for (size_t i = 0; i < m1.rows() && status == 0; ++i)
        for (size_t j = 0; j < m1.columns() && status == 0; ++j)
            status |= test_bool(m1(i,j) == m2(j,i), "transpose failed");
    return status;
}
    
int test_swapcolumns(size_t r, size_t c)
{
    mat m1(r, c);
    
    fillrandom(m1);

    mat m3;
    mat m4;
    
    int status = 0;
    for (size_t i = 0; i < 64 && i < m1.columns(); i += 3)
        for (size_t j = 0; j < 128 && j  < m1.columns(); j += 5)
        {
            m3 = m_copy(m1);
            m4 = m_copy(m1);
            m4.swapcolumns(i, j);
            for (size_t r = 0; r < m3.rows(); ++r)
            {
                bool col1 = m3(r, i), col2 = m3(r, j);
                m3.setbit(r, i, col2);
                m3.setbit(r, j, col1);
            }
            status |= test_bool(m3.isequal(m4), "swap columns failed");
        }
    return status;
}

int test_matrixref(size_t r = 512, size_t c = 512)
{
    if (r%64 != 0) return 0;
    if (c%64 != 0) return 0;
    LOG_CERR(r << "x" << c);
    mat matrix(2*r, 2*c);
    auto blockptr = matrix.blockptr();
    mat_view mref1(matrix);
    mat_view mrefUL(matrix.submatrix(0, r, 0, c));
    mat_view mrefUR(matrix.submatrix(0, r, c, c));
    mat_view mrefLL(matrix.submatrix(r, r, 0, c));
    mat_view mrefLR(matrix.submatrix(r, r, c, c));
    mrefUL.setbit(1,2);
    mrefUR.setbit(3,4);
    mrefLL.setbit(5,6);
    mrefLR.setbit(7,8);

    int status = 0;
    status |= test_bool(mref1(1,2));
    status |= test_bool(mref1(3,4+c));
    status |= test_bool(mref1(5+r,6));
    status |= test_bool(mref1(7+r,8+c));
    status |= test_bool(mref1.hw() == 4);
    status |= test_bool(1 == hammingweight(mrefUL));
    status |= test_bool(1 == hammingweight(mrefUR));
    status |= test_bool(1 == hammingweight(mrefLL));
    status |= test_bool(1 == hammingweight(mrefLR));

    fillrandom(matrix);
    status |= test_bool(hammingweight(matrix) == hammingweight(mrefUL)+hammingweight(mrefUR)+hammingweight(mrefLL)+hammingweight(mrefLR));
    double f = double(hammingweight(matrix))/double(4*r*c);
    status |= test_bool(f >= 0.45, "too low weight for random");
    status |= test_bool(f <= 0.55, "too high weight for random");
    
    if (r == c)
    {
        mrefUL.transpose(mrefUR);
        for (size_t i = 0; i < mrefUL.rows() && status == 0; ++i)
            for (size_t j = 0; j < mrefUL.columns() && status == 0; ++j)
                status |= test_bool(mrefUL(i,j) == mrefUR(j,i), "transpose failed");
    }
    return status;
}

int main(int, char**)
{
    int status = 0;

    for (size_t i = 4; i <= 512; ++i)
        status |= test_matrixref(i,i);
        
    for (size_t i = 1; i <= 4*64; ++i)
    {
        status |= test_transpose(i,i);
        status |= test_transpose(i,i+32);
        status |= test_transpose(i,i+64);
        status |= test_transpose(i,i+128);
    }

    status |= test_swapcolumns(1024, 256);
    
    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
