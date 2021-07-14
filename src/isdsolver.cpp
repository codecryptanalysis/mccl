#include <mccl/config/config.hpp>

#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/algorithm/prange.hpp>
#include <mccl/algorithm/lee_brickell.hpp>

#include <mccl/tools/parser.hpp>
#include <mccl/tools/statistics.hpp>
#include <mccl/tools/utils.hpp>

#include <mccl/contrib/program_options.hpp>
#include <mccl/contrib/string_algo.hpp>

#include <iostream>
#include <unistd.h>
#include <cstdlib>

namespace po = program_options;
namespace sa = string_algo;

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


struct add_program_options_helper
{
    po::options_description* opts;
    template<typename T, typename T2>
    void operator()(T&, const std::string& valname, const T2&, const std::string& description)
    {
      // first check if option already exists (some subISDs have a common parameter name like 'p')
      for (auto& option: opts->_options)
      {
        if (valname.size() == 1 && option->shortopt == valname)
          return;
        if (option->longopt == valname)
          return;
      }
      // otherwise add option
      opts->add_options()
        (valname, po::value<T>(), description)
        ;
    }
};

template<typename Configuration>
void add_program_options(po::options_description& opts, Configuration& conf)
{
  add_program_options_helper helper;
  helper.opts = &opts;
  conf.process(helper);
}

struct show_program_options_helper
{
    po::options_description* opts;
    template<typename T, typename T2>
    void operator()(T&, const std::string& valname, const T2& defaultval, const std::string& description)
    {
      // first check if option already exists (some subISDs have a common parameter name like 'p')
      for (auto& option: opts->_options)
      {
        if (valname.size() == 1 && option->shortopt == valname)
          return;
        if (option->longopt == valname)
          return;
      }
      // otherwise add option
      opts->add_options()
        (valname, po::value<T>()->default_value(defaultval), description)
        ;
    }
};

template<typename Configuration>
void show_program_options(Configuration& conf, unsigned line_length)
{
  po::options_description opts(conf.description, line_length, line_length/2);
  show_program_options_helper helper;
  helper.opts = &opts;
  conf.process(helper);
  std::cout << opts;
}

template<typename Configuration>
void show_manual(Configuration& conf)
{
  std::string manualstr = conf.manualstring;
  sa::replace_all(manualstr, std::string("\t"), std::string("  "));
  std::cout << "\n" << manualstr << "\n\n";
}

int main(int argc, char *argv[])
{
try
{
    std::string filepath, algo;
    size_t trials;
    int n = 0, k, w;
    uint64_t genseed;
    
    // maximum width to print program options
    unsigned line_length = 78;
    
    po::options_description
      allopts,
      cmdopts("Command options", line_length, line_length/2),
      auxopts("Extra options", line_length, line_length/2),
      genopts("Generator options", line_length, line_length/2),
      isdopts("ISD options")
      ;
      
    // These are the core commands, you need at least one of these
    cmdopts.add_options()
      ("help,h", "Show options")
      ("manual", "Show manual")
      ("file,f", po::value<std::string>(&filepath), "Specify input file")
      ("generate,g", "Generate random ISD instances")
      ;
    // these are other configuration options
    auxopts.add_options()
      ("algo,a", po::value<std::string>(&algo)->default_value("P"), "Specify algorithm: P, LB")
      ("trials,t", po::value<size_t>(&trials)->default_value(1), "Number of ISD trials")
      ("quiet,q", po::bool_switch(&quiet), "Quiet: supress most output")
      ;
    // options for the generator
    genopts.add_options()
      ("genunique", "Generate unique decoding instance")
      ("genrandom", "Generate random decoding instance")
      ("genseed", po::value<uint64_t>(&genseed), "Set instance generator random generator seed")
      ("n", po::value<int>(&n), "Code length")
      ("k", po::value<int>(&k)->default_value(-1), "Code dimension ( -1 = auto with rate 1/2 )")
      ("w", po::value<int>(&w)->default_value(-1), "Error weight ( -1 = 1.05*d_GV )")
      ;

    // collect options & help description
    // NOTE: if there are common options then only the first description AND *type* is used
    // any default values are ignored, so if no value is passed each algorithm can use its own default value
    
    add_program_options(isdopts, ISD_generic_config_default);
    add_program_options(isdopts, lee_brickell_config_default);

    /* parse all command line options */
    allopts.add(cmdopts).add(auxopts).add(genopts).add(isdopts);
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, allopts, false, false), vm);
    po::notify(vm);

    if (vm.positional.size() > 0)
      n = vm.positional[0].as<size_t>();
    if (vm.positional.size() > 1)
      k = vm.positional[1].as<int>();
    if (vm.positional.size() > 2)
      w = vm.positional[2].as<int>();
    if (vm.positional.size() > 3)
    {
      std::cout << "Unknown option: " << vm.positional[3].as<std::string>() << std::endl;
      return 1;
    }

    // store configuration in configmap
    configmap_t configmap;
    for (auto& o : vm)
    {
      if (o.second.empty())
        configmap[o.first];
      else
        configmap[o.first] = o.second.as<std::string>();
    }

    // update default configurations accordingly
    // note: prange has no configuration
    load_config(ISD_generic_config_default, configmap);
    load_config(lee_brickell_config_default, configmap);


    // show help and/or manual if requested or if no command was given
    if (vm.count("help") || vm.count("manual") ||
        vm.count("file")+vm.count("generate")==0
        )
    {
      std::cout << cmdopts << auxopts << genopts;
      
      show_program_options(ISD_generic_config_default, line_length);
      show_program_options(lee_brickell_config_default, line_length);

      if (vm.count("manual"))
      {
        std::cout << "\n\n === ISD solver manual ===\n";
        
        show_manual(ISD_generic_config_default);
        show_manual(lee_brickell_config_default);
        
      }
      return 0;
    }





    if (algo != "P" && algo != "LB")
    {
      std::cout << "Unknown algorithm: " << algo << std::endl;
      return 1;
    }

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



    // TODO: output ISD_generic / algo specific configuration
    std::cout << "n=" << n << ", k=" << k << ", w=" << w << " | algo=" << algo << " | trials=" << trials;
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
