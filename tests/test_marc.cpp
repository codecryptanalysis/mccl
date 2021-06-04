// config
#include <mccl/config/config.hpp>

// core header files
#include <mccl/core/matrix_base.hpp>
#include <mccl/core/matrix_ops.hpp>
#include <mccl/core/matrix2.hpp>

#include "test_utils.hpp"

using namespace mccl;

int main(int, char**)
{
/*
    m_ptr m1, m2;
    cm_ptr cm1, cm2;
    v_ptr v1, v2;
    cv_ptr cv1, cv2;
    m1 = m2;
    cm1 = m2;
    v1 = v2;
    cv1 = v2;
    v1 = m2;
    cv1 = m2;
    cv1 = cm2;
 */   
    mat m(5,5);
     vec_view vv, vv2;
//    vec v(5);
    vv = m[0];
    vv2 = m[1];
    
    m[2] = v_and(vv,vv2);

    return 0;
}
