#ifndef MCCL_CORE_MATRIX_M4RI_HPP
#define MCCL_CORE_MATRIX_M4RI_HPP

#include <mccl/core/matrix.hpp>
#include <stdlib.h>

MCCL_BEGIN_NAMESPACE

// m4ri only operates with matrices using uint64_t words
typedef matrix_ref_t<uint64_t> m4ri_ref_t;
// m4ri uses its own data structures, this is a pointer to a m4ri handle
typedef void* m4ri_handle_t;

m4ri_handle_t create_m4ri_handle(m4ri_ref_t& m);
inline void free_m4ri_handle(m4ri_handle_t h) { free(h); }


void transpose(m4ri_handle_t dst, const m4ri_handle_t src);

MCCL_END_NAMESPACE

#endif
