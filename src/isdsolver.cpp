#include <mccl/config/config.hpp>
#include <mccl/algorithm/prange.hpp>
#include <mccl/algorithm/LB.hpp>
#include <mccl/tools/parser.hpp>
#include <mccl/tools/statistics.hpp>
#include <mccl/tools/utils.hpp>

#include <mccl/contrib/program_options.hpp>

#include <iostream>
#include <unistd.h>

namespace po = program_options;

using namespace mccl;

bool quiet = false;

template<typename subISD_t = ISD_API_exhaustive_sparse_t>
int run_subISD(mat_view &H, vec_view &S, size_t w)
{
  subISD_t subISD;
  ISD_single_generic<subISD_t> ISD_single(subISD);
  ISD_single.initialize(H, S, w);
  ISD_single.solve();
  if (!quiet)
    std::cout << "Solution found:\n" << ISD_single.get_solution() << std::endl;
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
    std::cout << "Solution found:\n" << ISD_single.get_solution() << std::endl;
  return ISD_single.get_cnt();
}

int main(int argc, char *argv[])
{
try
{
    std::string filepath, algo;
    size_t trials = 1;
    size_t l = 0, p = 3;
    int n = 0, k = -1, w = -1, u = -1;
    
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
      ("n", po::value<int>(&n), "Code length")
      ("k", po::value<int>(&k)->default_value(-1), "Code dimension ( -1 = auto with rate 1/2 )")
      ("w", po::value<int>(&w)->default_value(-1), "Error weight ( -1 = 1.05*d_GV )")
      ("l", po::value<size_t>(&l)->default_value(0), "H2 rows")
      ("p", po::value<size_t>(&p)->default_value(3), "subISD parameter p")
      ("updaterows,u", po::value<int>(&u)->default_value(-1), "Echelon column swaps per iteration ( -1 = full )")
      ("quiet,q", po::bool_switch(&quiet), "Quiet: supress most output")
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
      k = vm.positional[1].as<int>();
    if (vm.positional.size() > 2)
      w = vm.positional[2].as<int>();
    if (vm.positional.size() > 3)
      l = vm.positional[3].as<size_t>();
    if (vm.positional.size() > 4)
      u = vm.positional[4].as<int>();
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
      // automatic choice of k is using rate 1/2
      if (k == -1)
        k = n / 2;
      // automatic choice of w is based on GV-bound: w = ceil(1.05 * d_GV)
      if (w == -1)
        w = get_cryptographic_w(n, k);
      if (n <= 0 || k >= n || w >= n)
      {
        std::cout << "Bad input parameters: n=" << n << ", k=" << k << ", w=" << w << std::endl;
        return 1;
      }
      parse.random_SD(n, k, w);
    }
    if (l > size_t(n-k))
      l = size_t(n-k);
    if (u <= 0)
      u = -1;

    std::cout << "n=" << n << ", k=" << k << ", w=" << w << " | algo=" << algo << ", l=" << l << ", p=" << p << ", u=" << u << " | trials=" << trials << std::endl;

    mat_view H = parse.get_H();
    vec_view S = parse.get_S();

    number_statistic<size_t> cnt_stat;
    time_statistic time_trial_stat, time_total_stat;

    time_total_stat.start();
    for (size_t i = 0; i < trials; ++i)
    {
      if(i > 0 && vm.count("gen"))
      {
        parse.regenerate();
        H.reset(parse.get_H());
        S.reset(parse.get_S());
      }
      time_trial_stat.start();
      if (algo=="P")
        cnt_stat.add( run_subISD<subISD_prange>(H,S,w) );
      else if (algo=="LB")
        cnt_stat.add( run_subISD<subISD_LB>(H,S,w) );
      else if (algo=="TP")
        cnt_stat.add( run_subISDT<subISDT_prange>(H,S,w,l,p,u) );
      else if (algo=="TLB")
        cnt_stat.add( run_subISDT<subISDT_LB>(H,S,w,l,p,u) );
      time_trial_stat.stop();
    }
    time_total_stat.stop();
    
    double mean_cnt = cnt_stat.mean(), median_cnt = cnt_stat.median();
    double inv_mean_cnt = double(1.0) / mean_cnt, inv_median_cnt = double(1.0) / median_cnt;
    std::cout << "Time                 :   total=" << time_total_stat.total() << "s (total mean=" << time_total_stat.total()/double(trials) << ") mean=" << time_trial_stat.mean() << "s median=" << time_trial_stat.median() << "s" << std::endl;
    std::cout << "Number of iterations :    mean=" << mean_cnt << " median=" << median_cnt << std::endl;
    std::cout << "Inverse of iterations: invmean=" << inv_mean_cnt << " invmedian=" << inv_median_cnt << std::endl;
    std::cout << "Time per iteration   :    mean=" << time_total_stat.total() / cnt_stat.total() << "s" << std::endl;
    
    return 0;
}
catch (std::exception& e)
{
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
}
catch (...)
{
    std::cerr << "Caught unknown exception!" << std::endl;
    return 1;
}
}
