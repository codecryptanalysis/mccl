#include <mccl/config/config.hpp>

#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/algorithm/prange.hpp>
#include <mccl/algorithm/lee_brickell.hpp>

#include <mccl/tools/parser.hpp>
#include <mccl/tools/statistics.hpp>
#include <mccl/tools/utils.hpp>

#include <mccl/contrib/program_options.hpp>

#include <iostream>
#include <unistd.h>

namespace po = program_options;

using namespace mccl;

bool quiet = false;

template<typename subISDT_t>
int run_subISDT(mat_view& H, vec_view& S, size_t w)
{
  subISDT_t subISDT;
  ISD_generic<subISDT_t> ISD_single(subISDT);
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
    size_t trials;
    int n = 0, k, w, u;
    size_t l, p;
    int updatetype;
    uint64_t genseed;
    
    po::options_description allopts, cmdopts("Command options"), opts("Extra options");
    // These are the core commands, you need at least one of these
    cmdopts.add_options()
      ("help,h", "Show options")
      ("file,f", po::value<std::string>(&filepath), "Specify input file")
      ("generate,g", "Generate random ISD instances")
      ;
    // these are other configuration options
    opts.add_options()
      ("algo,a", po::value<std::string>(&algo)->default_value("P"), "Specify algorithm: P, LB")
      ("trials,t", po::value<size_t>(&trials)->default_value(1), "Number of ISD trials")
      ("genunique", "Generate unique decoding instance")
      ("genrandom", "Generate random decoding instance")
      ("genseed", po::value<uint64_t>(&genseed), "Set instance generator random generator seed")
      ("n", po::value<int>(&n), "Code length")
      ("k", po::value<int>(&k)->default_value(-1), "Code dimension ( -1 = auto with rate 1/2 )")
      ("w", po::value<int>(&w)->default_value(-1), "Error weight ( -1 = 1.05*d_GV )")
      ("l", po::value<size_t>(&l)->default_value(0), "H2 rows")
      ("p", po::value<size_t>(&p)->default_value(3), "subISD parameter p")
      ("updaterows,u", po::value<int>(&u)->default_value(-1), "Echelon column swaps per iteration ( -1 = full )")
      ("updatetype", po::value<int>(&updatetype)->default_value(10), "Echelon/ISD column update type: 1, 2, 3, 4, 12, 13, 14, 10")
      ("quiet,q", po::bool_switch(&quiet), "Quiet: supress most output")
      ;
    // TODO: automatically generate from *_config_default: ISD_generic_config_default, lee_brickell_config_default, etc...
    allopts.add(cmdopts).add(opts);
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, allopts, false, true), vm);
    po::notify(vm);
    
    if (vm.count("help") || vm.count("file")+vm.count("generate")==0)
    {
      std::cout << cmdopts << opts;
      return 0;
    }
    if (algo != "P" && algo != "LB")
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
    configmap_t configmap;

    Parser parse;
    if (vm.count("genseed"))
      parse.seed(genseed);
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

    // write configmap for ISD algorithms
    configmap["l"] = detail::to_string(l);
    configmap["u"] = detail::to_string(u);
    configmap["updatetype"] = detail::to_string(updatetype);
    configmap["p"] = detail::to_string(p);

    // update default configurations accordingly
    // note: prange has no configuration
    load_config(ISD_generic_config_default, configmap);
    load_config(lee_brickell_config_default, configmap);

    // TODO: output ISD_generic / algo specific configuration
    std::cout << "n=" << n << ", k=" << k << ", w=" << w << " | algo=" << algo << ", l=" << l << ", p=" << p << ", u=" << u << " | trials=" << trials;
    if (vm.count("generate"))
      std::cout << ", genseed=" << parse.get_seed();
    std::cout << std::endl;

    mat_view H = parse.get_H();
    vec_view S = parse.get_S();

    number_statistic<size_t> cnt_stat;
    time_statistic time_trial_stat, time_total_stat;

    time_total_stat.start();
    for (size_t i = 0; i < trials; ++i)
    {
      if(i > 0 && vm.count("generate"))
      {
        parse.regenerate();
        H.reset(parse.get_H());
        S.reset(parse.get_S());
      }
      time_trial_stat.start();
      if (algo=="P")
        cnt_stat.add( run_subISDT<subISDT_prange>(H,S,w) );
      else if (algo=="LB")
        cnt_stat.add( run_subISDT<subISDT_lee_brickell>(H,S,w) );
      time_trial_stat.stop();
    }
    time_total_stat.stop();
    
    double mean_cnt = cnt_stat.mean(), median_cnt = cnt_stat.median();
    double inv_mean_cnt = double(1.0) / mean_cnt, inv_median_cnt = double(1.0) / median_cnt;
    std::cout << "Time                 :   total=" << time_total_stat.total() << "s (total mean=" << time_total_stat.total()/double(trials) << ") mean=" << time_trial_stat.mean() << "s median=" << time_trial_stat.median() << "s" << std::endl;
    std::cout << "Number of iterations :    mean=" << mean_cnt << " median=" << median_cnt << " Q1,Q3=" << cnt_stat.Q1() << "," << cnt_stat.Q3() << std::endl;
    std::cout << "Inverse of iterations: invmean=" << inv_mean_cnt << " invmedian=" << inv_median_cnt << std::endl;
    std::cout << "Time per iteration   :    mean=" << time_trial_stat.mean()/mean_cnt << "s" << std::endl;
    
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
