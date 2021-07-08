#include <mccl/config/config.hpp>
#include <mccl/tools/parser.hpp>
#include <mccl/algorithm/prange.hpp>
#include <mccl/algorithm/LB.hpp>

#include <mccl/contrib/program_options.hpp>

#include <iostream>
#include <unistd.h>
#include <chrono>
#include <algorithm>

namespace po = program_options;

using namespace mccl;

bool quiet = false;

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

// we should probably move this function in the tools
std::size_t binomial(std::size_t k, std::size_t N)
{
    if (k > N) return 0;
    std::size_t r = 1;
    if (k > N-k)
        k = N-k;
    for (unsigned i = 0; i < k; ++i)
    {
        r *= (N-i);
        r /= (i+1);
    }
    return r;
}

template<typename subISD_t = ISD_API_exhaustive_sparse_t>
int run_subISD(mat_view &H, vec_view &S, size_t w) {
  subISD_t subISD;
  ISD_single_generic<subISD_t> ISD_single(subISD);
  ISD_single.initialize(H, S, w);
  ISD_single.solve();
  if (!quiet)
  {
    std::cout << "Solution found:" << '\n';
    std::cout << ISD_single.get_solution() << '\n';
  }
  return ISD_single.get_cnt();
}

template<typename subISDT_t = ISD_API_exhaustive_transposed_sparserange_t>
int run_subISDT(mat_view& H, vec_view& S, size_t w, size_t l, size_t p, size_t u)
{
  subISDT_t subISDT;
  subISDT.configure(p);
  ISD_single_generic_transposed<subISDT_t> ISD_single(subISDT);
  ISD_single.configure(l, u);
  ISD_single.initialize(H, S, w);
  ISD_single.solve();
  if (!quiet)
  {
    std::cout << "Solution found:\n";
    std::cout << ISD_single.get_solution() << std::endl;
  }
  return ISD_single.get_cnt();
}

int main(int argc, char *argv[])
{
try
{
    std::string filepath, algo;
    size_t trials = 1;
    size_t n = 0, k = 0, w = 0, l = 0, u = 1, p = 3;
    
    po::options_description allopts, cmdopts("Command options"), opts("Extra options");
    // These are the core commands, you need at least one of these
    cmdopts.add_options()
      ("help,h", "Show options")
      ("file,f", po::value<std::string>(&filepath), "Specify input file")
      ("gen,g", "Generate random ISD instances")
      ;
    // these are other configuration options
    opts.add_options()
      ("algo,a", po::value<std::string>(&algo)->default_value("P"), "Specify algorithm: P, LB, TP, TLB")
      ("trials,t", po::value<size_t>(&trials)->default_value(1), "Number of ISD trials")
      ("genunique", "Generate unique decoding instance")
      ("genrandom", "Generate random decoding instance")
      ("n", po::value<size_t>(&n), "Code length")
      ("k", po::value<size_t>(&k), "Code dimension")
      ("w", po::value<size_t>(&w), "Error weight")
      ("l", po::value<size_t>(&l)->default_value(0), "H2 rows")
      ("p", po::value<size_t>(&p)->default_value(3), "subISD parameter p")
      ("u", po::value<size_t>(&u)->default_value(1), "I column swaps per iteration")
      ("q", po::bool_switch(&quiet), "Quiet: supress most output")
      ;
    allopts.add(cmdopts).add(opts);
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, allopts, false, true), vm);
    po::notify(vm);
    
    if (vm.count("help") || vm.count("file")+vm.count("gen")==0)
    {
      std::cout << cmdopts << opts;
      return 0;
    }
    if (algo != "P" && algo != "LB" && algo != "TP" && algo != "TLB")
    {
      std::cout << "Unknown algorithm: " << algo << std::endl;
      return 1;
    }
    if (vm.positional.size() > 0)
      n = vm.positional[0].as<size_t>();
    if (vm.positional.size() > 1)
      k = vm.positional[1].as<size_t>();
    if (vm.positional.size() > 2)
      w = vm.positional[2].as<size_t>();
    if (vm.positional.size() > 3)
      l = vm.positional[3].as<size_t>();
    if (vm.positional.size() > 4)
      u = vm.positional[4].as<size_t>();
    if (vm.positional.size() > 5)
    {
      std::cout << "Unknown option: " << vm.positional[5].as<std::string>() << std::endl;
      return 1;
    }

    Parser parse;
    if (filepath != "")
    {
      std::cout << "Parsing instance " << filepath << '\n';
      bool b = parse.load_file(filepath);
      if (!b) {
        std::cout << "Parsing instance failed" << '\n';
        return 1;
      }
      n = parse.get_n();
      k = parse.get_k();
      w = parse.get_w();
    } else
    {
      if (n == 0 || k >= n || w >= n)
      {
        std::cout << "Bad input parameters: n=" << n << ", k=" << k << ", w=" << w << std::endl;
        return 1;
      }
      parse.random_SD(n, k, w);
    }
    if (l > n-k)
      l = n-k;
    if (u < 1)
      u = 1;

    std::cout << "n=" << n << ", k=" << k << ", w=" << w << " | algo=" << algo << ", l=" << l << ", p=" << p << ", u=" << u << " | trials=" << trials << std::endl;

    mat_view H = parse.get_H();
    vec_view S = parse.get_S();
    number_statistic<size_t> cnt_stat;
    number_statistic<double> time_stat;
    auto time_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < trials; ++i)
    {
      if(i > 0 && vm.count("gen")) {
        parse.regenerate();
        H.reset(parse.get_H());
        S.reset(parse.get_S());
      }
      auto trial_start = std::chrono::high_resolution_clock::now();
      if (algo=="P")
        cnt_stat.add( run_subISD<subISD_prange>(H,S,w) );
      else if (algo=="LB")
        cnt_stat.add( run_subISD<subISD_LB>(H,S,w) );
      else if (algo=="TP")
        cnt_stat.add( run_subISDT<subISDT_prange>(H,S,w,l,p,u) );
      else if (algo=="TLB")
        cnt_stat.add( run_subISDT<subISDT_LB>(H,S,w,l,p,u) );
      auto trial_end = std::chrono::high_resolution_clock::now();
      time_stat.add( std::chrono::duration<double>(trial_end - trial_start).count() );
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(time_end - time_start).count();
    
    double mean_cnt = cnt_stat.mean(), median_cnt = cnt_stat.median();
    double inv_mean_cnt = double(1.0) / mean_cnt, inv_median_cnt = double(1.0) / median_cnt;
    std::cout << "Time: total=" << total_time << "s (total mean=" << total_time/double(trials) << ") mean=" << time_stat.mean() << "s median=" << time_stat.median() << "s" << std::endl;
    std::cout << "Number of iterations: mean=" << mean_cnt << " median=" << median_cnt << std::endl;
    std::cout << "Inverse of iterations: invmean=" << inv_mean_cnt << " invmedian=" << inv_median_cnt << std::endl;
    std::cout << "Mean iteration time: " << total_time / cnt_stat.total() << "s" << std::endl;
    
    return 0;
} catch (std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
} catch (...) {
    std::cerr << "Caught unknown exception!" << std::endl;
    return 1;
}
}
