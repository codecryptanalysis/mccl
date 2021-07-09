#ifndef MCCL_TOOLS_UTILS_HPP
#define MCCL_TOOLS_UTILS_HPP

#include <mccl/config/config.hpp>
#include <mccl/contrib/BigInt.hpp>

#include <math.h>

MCCL_BEGIN_NAMESPACE

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

size_t d_Gilbert_Varshamov(size_t n, size_t k)
{
  BigInt minballsize = pow(BigInt(2), n-k);
  BigInt ballsize = 1;
  size_t d = 1;
  while (ballsize < minballsize)
  {
    ballsize += binomial<BigInt>(n, d);
    ++d;
  }
  return d;
}


int get_cryptographic_w(size_t n, size_t k)
{
  return ceil( 1.05 * double(d_Gilbert_Varshamov(n, k)) );
}

MCCL_END_NAMESPACE

#endif
