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

  template<typename Int = size_t>
  Int gcd(Int x, Int y)
  {
    // base case
    if (x == 0)
      return y;
    if (y == 0)
      return x;
    // remove common factors of 2: k
    unsigned k = 0;
    for (; ((x|y)&1) == 0; ++k)
    {
      x >>= 1;
      y >>= 1;
    }
    // remove extra factors of 2
    while ((x&1) == 0)
      x >>= 1;
    // loop invariant: x odd
    do {
      // make y also odd
      while ((y&1) == 0)
        y >>= 1;
      // ensure y >= x
      if (x > y)
        std::swap(x,y);
      // reduce y with x
      y -= x;
    } while (y != 0);
    // return x and restore common factor 2**k
    return x << k;
  }

  template<typename Int = size_t>
  Int lcm(Int x, Int y)
  {
    Int g = gcd(x,y);
    x /= g;
    x *= y;
    return x;
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
    b *= (unsigned long)(n-d+1);
    b /= (unsigned long)(d);
  }
  return d;
}


inline int get_cryptographic_w(size_t n, size_t k)
{
  return ceil( 1.05 * double(d_gilbert_varshamov(n, k)) );
}

MCCL_END_NAMESPACE

#endif
