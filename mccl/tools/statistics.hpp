#ifndef MCCL_TOOLS_STATISTICS_HPP
#define MCCL_TOOLS_STATISTICS_HPP

#include <mccl/config/config.hpp>

#include <chrono>
#include <algorithm>
#include <stdexcept>

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
  double median()
  {
    if (size() == 0)
      throw std::runtime_error("number_statistic::median(): no samples!");
    std::sort(samples.begin(), samples.end());
    if (samples.size()%2 == 0)
    {
      return double(samples[size()/2 - 1] + samples[size()/2])/double(2.0);
    }
    return samples[samples.size()];
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


MCCL_END_NAMESPACE

#endif

