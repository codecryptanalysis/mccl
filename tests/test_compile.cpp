#include <mccl/config/config.hpp>

#include <mccl/core/matrix_detail.hpp>
#include <mccl/core/matrix.hpp>

#include <mccl/contrib/string_algo.hpp>
#include <mccl/contrib/thread_pool.hpp>
#include <mccl/contrib/parallel_algorithms.hpp>
#include <mccl/contrib/json.hpp>
#include <mccl/contrib/program_options.hpp>

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


int test_matrixref()
{
    mat_t matrix(1024, 1024);
    matrixref mref1(matrix);
    matrixref mrefUL(matrix.submatrix(0, 512, 0, 512));
    matrixref mrefUR(matrix.submatrix(0, 512, 512, 512));
    matrixref mrefLL(matrix.submatrix(512, 512, 0, 512));
    matrixref mrefLR(matrix.submatrix(512, 512, 512, 512));
    mrefUL.bitset(1,2);
    mrefUR.bitset(3,4);
    mrefLL.bitset(5,6);
    mrefLR.bitset(7,8);

    int status = 0;
    status |= test_bool(mref1(1,2));
    status |= test_bool(mref1(3,4+512));
    status |= test_bool(mref1(5+512,6));
    status |= test_bool(mref1(7+512,8+512));
    status |= test_bool(mref1.hammingweight() == 4);
    status |= test_bool(1 == hammingweight(mrefUL));
    status |= test_bool(1 == hammingweight(mrefUR));
    status |= test_bool(1 == hammingweight(mrefLL));
    status |= test_bool(1 == hammingweight(mrefLR));
    return status;
}




template<typename T>
int test_constructor(const std::string& Tname, std::vector<bool> truthtable)
{
    int status = 0;
    status |= test_bool(truthtable[ 0] == std::is_constructible<T,             matrixref&>::value, "test constructor `" + Tname + "(            matrixref&)` failed: " + std::to_string(truthtable[0]));
    status |= test_bool(truthtable[ 1] == std::is_constructible<T, const       matrixref&>::value, "test constructor `" + Tname + "(const       matrixref&)` failed: " + std::to_string(truthtable[1]));
    status |= test_bool(truthtable[ 2] == std::is_constructible<T,             matrixref*>::value, "test constructor `" + Tname + "(            matrixref*)` failed: " + std::to_string(truthtable[2]));
    status |= test_bool(truthtable[ 3] == std::is_constructible<T, const       matrixref*>::value, "test constructor `" + Tname + "(const       matrixref*)` failed: " + std::to_string(truthtable[3]));

    status |= test_bool(truthtable[ 4] == std::is_constructible<T,             matrixptr&>::value, "test constructor `" + Tname + "(            matrixptr&)` failed: " + std::to_string(truthtable[4]));
    status |= test_bool(truthtable[ 5] == std::is_constructible<T, const       matrixptr&>::value, "test constructor `" + Tname + "(const       matrixptr&)` failed: " + std::to_string(truthtable[5]));
    status |= test_bool(truthtable[ 6] == std::is_constructible<T,             matrixptr*>::value, "test constructor `" + Tname + "(            matrixptr*)` failed: " + std::to_string(truthtable[6]));
    status |= test_bool(truthtable[ 7] == std::is_constructible<T, const       matrixptr*>::value, "test constructor `" + Tname + "(const       matrixptr*)` failed: " + std::to_string(truthtable[7]));

    status |= test_bool(truthtable[ 8] == std::is_constructible<T,             vectorref&>::value, "test constructor `" + Tname + "(            vectorref&)` failed: " + std::to_string(truthtable[8]));
    status |= test_bool(truthtable[ 9] == std::is_constructible<T, const       vectorref&>::value, "test constructor `" + Tname + "(const       vectorref&)` failed: " + std::to_string(truthtable[9]));
    status |= test_bool(truthtable[10] == std::is_constructible<T,             vectorref*>::value, "test constructor `" + Tname + "(            vectorref*)` failed: " + std::to_string(truthtable[10]));
    status |= test_bool(truthtable[11] == std::is_constructible<T, const       vectorref*>::value, "test constructor `" + Tname + "(const       vectorref*)` failed: " + std::to_string(truthtable[11]));

    status |= test_bool(truthtable[12] == std::is_constructible<T,             vectorptr&>::value, "test constructor `" + Tname + "(            vectorptr&)` failed: " + std::to_string(truthtable[12]));
    status |= test_bool(truthtable[13] == std::is_constructible<T, const       vectorptr&>::value, "test constructor `" + Tname + "(const       vectorptr&)` failed: " + std::to_string(truthtable[13]));
    status |= test_bool(truthtable[14] == std::is_constructible<T,             vectorptr*>::value, "test constructor `" + Tname + "(            vectorptr*)` failed: " + std::to_string(truthtable[14]));
    status |= test_bool(truthtable[15] == std::is_constructible<T, const       vectorptr*>::value, "test constructor `" + Tname + "(const       vectorptr*)` failed: " + std::to_string(truthtable[15]));

    status |= test_bool(truthtable[16] == std::is_constructible<T,        base_t&>::value, "test constructor `" + Tname + "(       base_t&)` failed: " + std::to_string(truthtable[16]));
    status |= test_bool(truthtable[17] == std::is_constructible<T, const  base_t&>::value, "test constructor `" + Tname + "(const  base_t&)` failed: " + std::to_string(truthtable[17]));
    status |= test_bool(truthtable[18] == std::is_constructible<T,        base_t*>::value, "test constructor `" + Tname + "(       base_t*)` failed: " + std::to_string(truthtable[18]));
    status |= test_bool(truthtable[19] == std::is_constructible<T, const  base_t*>::value, "test constructor `" + Tname + "(const  base_t*)` failed: " + std::to_string(truthtable[19]));

    if (status == 0)
        return 0;
    return -1;
}

template<typename T>
int test_assign(const std::string& Tname, std::vector<bool> truthtable)
{
    int status = 0;
    status |= test_bool(truthtable[ 0] == std::is_assignable<T,             matrixref&>::value, "test assign `" + Tname + "(            matrixref&)` failed: " + std::to_string(truthtable[0]));
    status |= test_bool(truthtable[ 1] == std::is_assignable<T, const       matrixref&>::value, "test assign `" + Tname + "(const       matrixref&)` failed: " + std::to_string(truthtable[1]));
    status |= test_bool(truthtable[ 2] == std::is_assignable<T,             matrixref*>::value, "test assign `" + Tname + "(            matrixref*)` failed: " + std::to_string(truthtable[2]));
    status |= test_bool(truthtable[ 3] == std::is_assignable<T, const       matrixref*>::value, "test assign `" + Tname + "(const       matrixref*)` failed: " + std::to_string(truthtable[3]));

    status |= test_bool(truthtable[ 4] == std::is_assignable<T,             matrixptr&>::value, "test assign `" + Tname + "(            matrixptr&)` failed: " + std::to_string(truthtable[4]));
    status |= test_bool(truthtable[ 5] == std::is_assignable<T, const       matrixptr&>::value, "test assign `" + Tname + "(const       matrixptr&)` failed: " + std::to_string(truthtable[5]));
    status |= test_bool(truthtable[ 6] == std::is_assignable<T,             matrixptr*>::value, "test assign `" + Tname + "(            matrixptr*)` failed: " + std::to_string(truthtable[6]));
    status |= test_bool(truthtable[ 7] == std::is_assignable<T, const       matrixptr*>::value, "test assign `" + Tname + "(const       matrixptr*)` failed: " + std::to_string(truthtable[7]));

    status |= test_bool(truthtable[ 8] == std::is_assignable<T,             vectorref&>::value, "test assign `" + Tname + "(            vectorref&)` failed: " + std::to_string(truthtable[8]));
    status |= test_bool(truthtable[ 9] == std::is_assignable<T, const       vectorref&>::value, "test assign `" + Tname + "(const       vectorref&)` failed: " + std::to_string(truthtable[9]));
    status |= test_bool(truthtable[10] == std::is_assignable<T,             vectorref*>::value, "test assign `" + Tname + "(            vectorref*)` failed: " + std::to_string(truthtable[10]));
    status |= test_bool(truthtable[11] == std::is_assignable<T, const       vectorref*>::value, "test assign `" + Tname + "(const       vectorref*)` failed: " + std::to_string(truthtable[11]));

    status |= test_bool(truthtable[12] == std::is_assignable<T,             vectorptr&>::value, "test assign `" + Tname + "(            vectorptr&)` failed: " + std::to_string(truthtable[12]));
    status |= test_bool(truthtable[13] == std::is_assignable<T, const       vectorptr&>::value, "test assign `" + Tname + "(const       vectorptr&)` failed: " + std::to_string(truthtable[13]));
    status |= test_bool(truthtable[14] == std::is_assignable<T,             vectorptr*>::value, "test assign `" + Tname + "(            vectorptr*)` failed: " + std::to_string(truthtable[14]));
    status |= test_bool(truthtable[15] == std::is_assignable<T, const       vectorptr*>::value, "test assign `" + Tname + "(const       vectorptr*)` failed: " + std::to_string(truthtable[15]));

    status |= test_bool(truthtable[16] == std::is_assignable<T,        base_t&>::value, "test assign `" + Tname + "(       base_t&)` failed: " + std::to_string(truthtable[16]));
    status |= test_bool(truthtable[17] == std::is_assignable<T, const  base_t&>::value, "test assign `" + Tname + "(const  base_t&)` failed: " + std::to_string(truthtable[17]));
    status |= test_bool(truthtable[18] == std::is_assignable<T,        base_t*>::value, "test assign `" + Tname + "(       base_t*)` failed: " + std::to_string(truthtable[18]));
    status |= test_bool(truthtable[19] == std::is_assignable<T, const  base_t*>::value, "test assign `" + Tname + "(const  base_t*)` failed: " + std::to_string(truthtable[19]));
    if (status == 0)
        return 0;
    return -1;
}

int main(int, char**)
{
    int status = 0;

    status |= test_constructor<      matrixref>("      matrixref", { 1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0, 1,1,0,0 });
    status |= test_assign     <      matrixref>("      matrixref", { 1,1,0,0, 0,0,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0 });
    status |= test_constructor<      vectorref>("      vectorref", { 0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0, 1,1,0,0 });
    status |= test_assign     <      vectorref>("      vectorref", { 0,0,0,0, 0,0,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0 });

    status |= test_constructor<      matrixptr>("      matrixptr", { 0,0,0,0, 1,1,0,0, 0,0,0,0, 1,1,0,0, 1,1,0,0 });
    status |= test_assign     <      matrixptr>("      matrixptr", { 0,0,0,0, 1,1,0,0, 0,0,0,0, 1,1,0,0, 0,0,0,0 });
    status |= test_constructor<      vectorptr>("      vectorptr", { 0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,0,0, 1,1,0,0 });
    status |= test_assign     <      vectorptr>("      vectorptr", { 0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,0,0, 0,0,0,0 });

    if (status == 0)
    {
        LOG_CERR("All tests passed.");
        return 0;
    }
    return -1;
}
