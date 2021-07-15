#ifndef MCCL_TOOLS_STATISTICS_HPP
#define MCCL_TOOLS_STATISTICS_HPP

#include <mccl/config/config.hpp>

#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iomanip>

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
  double elapsed_time()
  {
    time_point _end = clock_t::now();
    return duration_t(_end - _start).count();
  }
  time_point _start;
};

template<typename T>
struct cpucycle_helper {
  T& stat;
  cpucycle_helper(T& _stat): stat(_stat)
  {
    stat.start();
  }
  ~cpucycle_helper()
  {
    stat.stop();
  }
};
struct cpucycle_statistic
  : public number_statistic<uint64_t>
{
#ifdef MCCL_HAVE_CPU_COUNTERS
  static inline uint64_t clock() { return __rdtsc(); }
#else
  static inline uint64_t clock() { return 0; }
#endif
  cpucycle_statistic()
    : _total(0)
  {}
  void start()
  {
    _start = clock();
  }
  void stop()
  {
    uint64_t _end = clock();
    _total += (_end - _start);
  }
  void refresh()
  {
    this->add(_total);
    _total = 0;
  }
  uint64_t _start, _total;
};
#ifdef MCCL_HAVE_CPU_COUNTERS
#define MCCL_CPUCYCLE_STATISTIC_BLOCK(s) cpucycle_helper<cpucycle_statistic> _mccl_cpucycle_guard(s);
#else
#define MCCL_CPUCYCLE_STATISTIC_BLOCK(s)
#endif


struct counter_statistic
  : public number_statistic<uint64_t>
{
  
  inline void inc(uint64_t val=1)
  {
    _counter+=val;
  }
  inline void dec(uint64_t val=1)
  {
    _counter-=val;
  }
  void refresh()
  {
    this->add( _counter );
    _counter = 0;
  }
  void print(std::string name, std::ostream& o = std::cerr)
  {
    o << std::setw(15) << name << ":";
    o << std::setw(15) << this->total() << ",";
    o << std::setw(15) << this->mean() << ",";
    o << std::setw(15) << this->median() << std::endl; 
  }
  uint64_t _counter = 0;
};

class decoding_statistics {
public:
  std::string name;
  decoding_statistics(std::string _name) : name(_name) {}

  // counters
  counter_statistic cnt_initialize;
  counter_statistic cnt_callback;
  counter_statistic cnt_prepare_loop;
  counter_statistic cnt_loop_next;
  counter_statistic cnt_solve;
  counter_statistic cnt_check_solution;
  // time

  // cpucycle

  // refresh counters
  void refresh() {
    cnt_initialize.refresh();
    cnt_callback.refresh();
    cnt_prepare_loop.refresh();
    cnt_loop_next.refresh();
    cnt_solve.refresh();
    cnt_check_solution.refresh();
  }

  // print
  void print(std::ostream& o = std::cerr) {
    if(cnt_solve.size()==0) {
      o << "No statistics " << name << std::endl;
      return;
    }
    o << "Decoder: " << name << std::endl;
    o << std::setw(15+17) << "total count," << std::setw(16) << "mean count," << std::setw(16) << "median count," << std::endl;
    cnt_initialize.print("Initialize", o);
    cnt_callback.print("Callback", o);
    cnt_prepare_loop.print("Prepare loop", o);
    cnt_loop_next.print("Loop next", o);
    cnt_solve.print("Solve", o);
    cnt_check_solution.print("Check solution", o);
    o << std::endl;
  }
};


MCCL_END_NAMESPACE

#endif
