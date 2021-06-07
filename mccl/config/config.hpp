#ifndef MCCL_CONFIG_CONFIG_H
#define MCCL_CONFIG_CONFIG_H

#include <mccl/config/config.h>
#include <cstdint>

#ifndef MCCL_NAMESPACE
#define MCCL_NAMESPACE mccl
#endif

#ifndef MCCL_BEGIN_NAMESPACE
#define MCCL_BEGIN_NAMESPACE namespace MCCL_NAMESPACE {
#endif

#ifndef MCCL_END_NAMESPACE
#define MCCL_END_NAMESPACE }
#endif

#define MCCL_VECTOR_ASSUME_EQUAL_DIMENSIONS
#define MCCL_VECTOR_ASSUME_NONEMPTY

MCCL_BEGIN_NAMESPACE

using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;
using std::int_least8_t;
using std::uint_least8_t;
using std::int_least16_t;
using std::uint_least16_t;
using std::int_least32_t;
using std::uint_least32_t;
using std::int_least64_t;
using std::uint_least64_t;
using std::int_fast8_t;
using std::uint_fast8_t;
using std::int_fast16_t;
using std::uint_fast16_t;
using std::int_fast32_t;
using std::uint_fast32_t;
using std::int_fast64_t;
using std::uint_fast64_t;
using std::intptr_t;
using std::uintptr_t;
using std::size_t;

MCCL_END_NAMESPACE

#endif
