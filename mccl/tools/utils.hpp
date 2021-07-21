#ifndef MCCL_TOOLS_UTILS_HPP
#define MCCL_TOOLS_UTILS_HPP

#include <mccl/config/config.hpp>

#include <math.h>

MCCL_BEGIN_NAMESPACE

namespace detail
{

  template<typename Int = size_t>
  Int binomial(size_t N, size_t k)
  {
    Int r = 0;
    if (k > N)
      return r;
    if (k > N-k)
      k = N-k;
    r = 1;
    for (size_t i = 0; i < k; ++i)
    {
      r *= Int(N-i);
      r /= Int(i+1);
    }
    return r;
  }

}

inline size_t d_gilbert_varshamov(size_t n, size_t k)
{
  size_t d = 0;
  bigint_t aux = bigint_t(1) << (n-k);
  bigint_t b = 1;
  while(aux >= 0) {
    aux -= b;
    d++;
    b *= (n-d+1);
    b /= d;
  }
  return d;
}


inline int get_cryptographic_w(size_t n, size_t k)
{
  return ceil( 1.05 * double(d_gilbert_varshamov(n, k)) );
}

MCCL_END_NAMESPACE

#endif
