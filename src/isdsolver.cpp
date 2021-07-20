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
#include <memory>

namespace po = program_options;
namespace sa = string_algo;

using namespace mccl;


/* Helper structs and functions for configuration of submodules */

/* Helpers to collect submodule parameters into program options to parse */
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
    // special case for boolean
    void operator()(bool&, const std::string& valname, bool, const std::string& description)
    {
      std::string optname = valname;
      // first check if option already exists (some subISDs have a common parameter name like 'p')
      for (auto& option: opts->_options)
      {
        if (optname.size() == 1 && option->shortopt == optname)
          return;
        if (option->longopt == optname)
          return;
      }
      // otherwise add option
      opts->add_options()
        (optname, po::bool_switch(), description)
        ;
      opts->add_options()
        ("no-"+optname, po::bool_switch(), description)
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

/* Helpers to print submodule parameters as program options */
struct get_program_options_helper
{
    po::options_description* opts;
    template<typename T, typename T2>
    void operator()(T&, const std::string& valname, const T2& defaultval, const std::string& description)
    {
      opts->add_options()
        (valname, po::value<T>()->default_value(defaultval), description)
        ;
    }
    // special case for boolean
    void operator()(bool&, const std::string& valname, bool defaultval, const std::string& description)
    {
      std::string optname = valname;
      // if defaultval = true then first show the "no-<valname>" option
      if (defaultval)
      {
        opts->add_options()
          ("no-"+optname, po::bool_switch(), description)
          ;
        opts->add_options()
          (optname, po::bool_switch(), "   (default)")
          ;
      }
      else
      {
        opts->add_options()
          (optname, po::bool_switch(), description)
          ;
        opts->add_options()
          ("no-"+optname, po::bool_switch(), "   (default)")
          ;
      }
    }
};
template<typename Configuration>
po::options_description get_program_options(Configuration& conf, unsigned line_length)
{
  po::options_description opts(conf.description, line_length, line_length/2);
  get_program_options_helper helper;
  helper.opts = &opts;
  conf.process(helper);
  return opts;
}

/* Helpers to print manual of submodules */
template<typename Configuration>
void show_manual(Configuration& conf)
{
  std::string manualstr = conf.manualstring;
  sa::replace_all(manualstr, std::string("\t"), std::string("  "));
  std::cout << "\n" << manualstr << "\n\n";
}

template<typename Module>
std::string get_configuration_str(Module& m)
{
  configmap_t configmap;
  m.save_config(configmap);
  std::string ret;
  for (auto& pv : configmap)
  {
    if (!ret.empty())
      ret.push_back(' ');
    ret.append(pv.first).append("=").append(pv.second);
  }
  return ret;
}





/* run Trials */

void run_ISD(syndrome_decoding_API& ISD, cmat_view& H, cvec_view& S, size_t w, bool quiet)
{
  ISD.initialize(H, S, w);
  ISD.solve();
  if (!quiet)
    std::cout << "Solution found:\n" << ISD.get_solution() << std::endl;
}

void runtrials_ISD(syndrome_decoding_API& ISD, cmat_view& H, cvec_view& S, size_t w, size_t trials, bool quiet, bool generate, Parser& parse)
{
  // run trials
  time_statistic time_trial_stat, time_total_stat;

  time_total_stat.start();
  for (size_t i = 0; i < trials; ++i)
  {
    if(i > 0 && generate)
    {
      parse.regenerate();
      H.reset(parse.get_H());
      S.reset(parse.get_S());
    }
    time_trial_stat.start();
    run_ISD(ISD, H,S,w, quiet);
    time_trial_stat.stop();
  }
  time_total_stat.stop();

  /* print basic overall statistics */
  double total_time = time_total_stat.total(), avg_time = time_trial_stat.mean();
  double avg_loop_cnt = ISD.get_stats().cnt_loop_next.mean(),
         total_loop_cnt = ISD.get_stats().cnt_loop_next.total();

  std::cout << "=== Basic statistics ===" << std::endl;
  std::cout << "  Time                 : mean= " << std::setw(10) << avg_time     << "s  total= " << std::setw(10) << total_time << "s" << std::endl;
  std::cout << "  Number of iterations : mean= " << std::setw(10) << avg_loop_cnt << "   total= " << std::setw(10) << total_loop_cnt << std::endl;
  std::cout << "  Inverse of iterations: mean= " << std::setw(10) << 1.0/avg_loop_cnt << std::endl;
  std::cout << "  Time per iteration   : mean= " << std::setw(10) << avg_time/avg_loop_cnt << "s" << std::endl;
}



/* run Benchmark */

void benchmark_ISD(syndrome_decoding_API& ISD, cmat_view& H, cvec_view& S, size_t w, size_t min_iterations, double min_total_time)
{
  ISD.initialize(H, S, w);
  ISD.prepare_loop(true);
  
  size_t its = min_iterations, total_its = 0;
  time_statistic total_time;
  total_time.start();
  while (true)
  {
    for (size_t i = 0; i < its; ++i)
      ISD.loop_next();
      
    total_its += its;
    double total_elapsed_time = total_time.elapsed_time();
    
    if (total_elapsed_time >= min_total_time)
      break;
    // if measured time is very small, try at least 1000 x the number of iterations
    if (total_elapsed_time <= 0.0)
    {
      its *= 1000;
      continue;
    }
    its = size_t( double(total_its) * (min_total_time * 1.25) / total_elapsed_time ) - total_its;
  }
  total_time.stop();
  
  std::cout << "Time                 : " << total_time.total() << "s" << std::endl;
  std::cout << "Number of iterations : " << total_its << std::endl;
  std::cout << "Time per iteration   : mean=" << total_time.total()/double(total_its) << "s" << std::endl;
}


/* Main program */

int main(int argc, char *argv[])
{
try
{
    /* Configuration variables */
    std::string filepath, algo;
    size_t trials;
    bool quiet = false;
    bool print_stats = false;
    
    // generator options
    int n = 0, k, w;
    uint64_t genseed;
    
    // benchmark options
    bool benchmark = false;
    size_t min_bench_iterations;
    double min_bench_time;
    
    // maximum width to print program options
    unsigned line_length = 78;
    
    
    /* Define program options */
    po::options_description
      allopts,
      cmdopts("Command options", line_length, line_length/2),
      auxopts("Extra options", line_length, line_length/2),
      genopts("Generator options", line_length, line_length/2),
      benchopts("Benchmark options", line_length, line_length/2),
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
      ("quiet,q", po::bool_switch(&quiet), "Quiet: reduce verbosity of trials")
      ("printstats", po::bool_switch(&print_stats), "Print ISD function call statistics")
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
    benchopts.add_options()
      ("benchmark", po::bool_switch(&benchmark), "Instead of many trials, benchmark ISD iterations on 1 instance")
      ("minbenchits", po::value<size_t>(&min_bench_iterations)->default_value(100), "Minimum number of ISD iterations")
      ("minbenchtime", po::value<double>(&min_bench_time)->default_value(100.0), "Minimal total time (s) for benchmark")
      ;

    /* Collect submodule program options */
    //  if there are common options then only the first description is used
    //  any default values are ignored, so if no value is passed each algorithm can use its own default value

    add_program_options(isdopts, ISD_generic_config_default);
    add_program_options(isdopts, lee_brickell_config_default);

    /* Parse all program options */
    allopts.add(cmdopts).add(auxopts).add(genopts).add(benchopts).add(isdopts);
    po::variables_map vm;
    // TODO: configuration file?
    // parse command line parameters
    po::store(po::parse_command_line(argc, argv, allopts, false, true /*allow positional parameters*/), vm);
    po::notify(vm);

    // process positional parameters if any
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

    // pass configmap to submodules
    // note: prange has no configuration
    load_config(ISD_generic_config_default, configmap);
    load_config(lee_brickell_config_default, configmap);


    /* show help and/or manual if requested or if no command was given */
    if (vm.count("help") || vm.count("manual") ||
        vm.count("file")+vm.count("generate")==0
        )
    {
      po::print_options_description(
        { cmdopts, auxopts, genopts, benchopts,
          get_program_options(ISD_generic_config_default, line_length),
          get_program_options(lee_brickell_config_default, line_length)
        });

      if (vm.count("manual"))
      {
        std::cout << "\n\n === ISD solver manual ===\n";
        
        show_manual(ISD_generic_config_default);
        show_manual(lee_brickell_config_default);
        
      }
      return 0;
    }


    /* Create the corresponding syndrome decoding object */
    std::unique_ptr<syndrome_decoding_API> ISD_ptr;
    std::unique_ptr<subISDT_API> subISD_ptr;
    std::string ISD_conf_str, subISD_conf_str;

    sa::to_upper(algo);
    if (algo == "P" || algo == "PRANGE")
    {
      algo = "Prange";
      auto _subISD = new subISDT_prange();
      auto _ISD = new ISD_generic<subISDT_prange>(*_subISD);
      subISD_ptr.reset( _subISD );
      ISD_ptr.reset( _ISD );
      subISD_conf_str = get_configuration_str(*_subISD);
      ISD_conf_str = get_configuration_str(*_ISD);
    }
    else if (algo == "LB" || algo == "LEEBRICKELL" || algo == "LEE-BRICKELL")
    {
      algo = "Lee-Brickell";
      auto _subISD = new subISDT_lee_brickell();
      auto _ISD = new ISD_generic<subISDT_lee_brickell>(*_subISD);
      subISD_ptr.reset( _subISD );
      ISD_ptr.reset( _ISD );
      subISD_conf_str = get_configuration_str(*_subISD);
      ISD_conf_str = get_configuration_str(*_ISD);
    }
    else
    {
      std::cout << "Unknown algorithm: " << algo << std::endl;
      return 1;
    }


    /* parse or generate instances */
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


    std::cout << "Run settings       : n=" << n << " k=" << k << " w=" << w << " trials=" << trials;
    if (vm.count("generate"))
      std::cout << " genseed=" << parse.get_seed();
    std::cout << std::endl;
    std::cout << " -     ISD generic : " << ISD_conf_str << std::endl;
    std::cout << " - " << std::setw(15) << algo << " : " << subISD_conf_str << std::endl;

    /* run all trials / benchmark */
    cmat_view H = parse.get_H();
    cvec_view S = parse.get_S();

    if (benchmark)
    {
      // run benchmark
      if (min_bench_iterations == 0)
        min_bench_iterations = 1;
      if (min_bench_time <= 1.0)
        min_bench_time = 1.0;
      benchmark_ISD(*ISD_ptr, H,S,w, min_bench_iterations, min_bench_time);
    }
    else
    {
      runtrials_ISD(*ISD_ptr, H,S,w, trials, quiet, vm.count("generate"), parse);
    }

    /* print detailed statistics */
    if (print_stats)
    {
      std::cout << "\n=== Detailed statistics ===" << std::endl;
      ISD_ptr->get_stats().print(std::cout);
      subISD_ptr->get_stats().print(std::cout);
    }
    
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
