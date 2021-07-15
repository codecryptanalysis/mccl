#ifndef MCCL_TOOLS_STATISTICS_HPP
#define MCCL_TOOLS_STATISTICS_HPP

#include <mccl/config/config.hpp>

#include <chrono>
#include <algorithm>
#include <stdexcept>

#ifdef MCCL_HAVE_CPU_COUNTERS
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

MCCL_BEGIN_NAMESPACE

template<typename number_t>
struct number_statistic
{
  std::vector<number_t> samples;
  void add(number_t n)
  {
    samples.push_back(n);
  }
  void clear()
  {
    samples.clear();
  }
  void reserve(size_t n)
  {
    samples.reserve(n);
  }
  size_t size() const
  {
    return samples.size();
  }
  double total()
  {
    return double(std::accumulate(samples.begin(), samples.end(), number_t(0)));
  }
  double mean()
  {
    if (size() == 0)
      throw std::runtime_error("number_statistic::mean(): no samples!");
    return total()/double(size());
  }
  double _mid(size_t b, size_t e)
  {
    if (e-b == 0)
      throw std::runtime_error("number_statistic::_mid(): no samples!");
    if (e-b == 1)
      return samples[b];
    if (!std::is_sorted(samples.begin(), samples.end()))
      std::sort(samples.begin(), samples.end());
    if ((e-b)%2 == 0)
      return double(samples[b + (e-b)/2 - 1] + samples[b + (e-b)/2])/double(2.0);
    return samples[b + (e-b)/2];
  }
  double median()
  {
    return _mid(0, size());
  }
  double Q1()
  {
    if (size() == 1)
      return samples[0];
    return _mid(0, size()/2);
  }
  double Q3()
  {
    return _mid(size()/2, size());
  }
};

struct time_statistic
  : public number_statistic<double>
{
  typedef std::chrono::high_resolution_clock clock_t;
  typedef typename clock_t::time_point time_point;
  typedef std::chrono::duration<double> duration_t;
  
  void start()
  {
    _start = clock_t::now();
  }
  void stop()
  {
    time_point _end = clock_t::now();
    this->add( duration_t(_end - _start).count() );
  }
  time_point _start;
};

struct cpucycle_statistic
  : public number_statistic<uint64_t>
{
#ifdef MCCL_HAVE_CPU_COUNTERS
  static inline uint64_t clock() { return __rdtsc(); }
#else
  static inline uint64_t clock() { return 0; }
#endif
  void start()
  {
    _start = clock();
  }
  void stop()
  {
    uint64_t _end = clock();
    this->add( _end - _start );
  }
  uint64_t _start;
};


MCCL_END_NAMESPACE

#endif
