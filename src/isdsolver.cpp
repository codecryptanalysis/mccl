#include<iostream>
#include <unistd.h>

#include <mccl/tools/parser.hpp>
#include <mccl/algorithm/decoding.hpp>


using namespace mccl;

void usage() {
  std::cout << "Usage: ./isdsolver -f INSTANCE_PATH -a ALGO " << '\n';
  std::cout << "ALGO should be \"P\" for PRANGE algorithm or \"LB\" for LEE BRICKEL algorithm" << '\n';
}

int main(int argc, char *argv[])
{
    std::string path = "";
    std::string algo = "";

    int option;
    while((option = getopt(argc, argv, ":f:a:")) != -1){
      switch(option){
        case 'f':
          path = optarg;
          break;
        case 'a':
          algo = optarg;
          break;
        case '?':
          printf("Unknown option: %c\n", optopt);
          break;
       }
    }

    if (algo=="P") {
      algo = "PRANGE";
      std::cout << "Using PRANGE algorithm" << '\n';
    }
    else if (algo=="LB") {
      algo = "LEE BRICKEL";
      std::cout << "Using LEE BRICKEL algorithm" << '\n';
    }
    else {
      std::cout << "Unknown algorithm " << algo << '\n';
      usage();
      std::cout << "Using PRANGE algorithm as default" << '\n';
      algo = "PRANGE";
    }

    if(path==""){
      std::cout << "Missing challenge instance" << '\n';
      usage();
      std::cout << "Aborting" << '\n';
      return 1;
    }

    Parser parse;
    std::cout << "Parsing instance " << path << '\n';
    bool b = parse.load_file(path);

    if (!b) {
        std::cout << "Parsing instance failed" << '\n';
        return 1;
    }

    mat_view H = parse.get_H();
    vec_view S = parse.get_S();
    size_t n = parse.get_n();
    size_t k = parse.get_k();
    size_t w = parse.get_w();

    std::cout << "n = " << n << '\n';
    std::cout << "k = " << k << '\n';
    std::cout << "w = " << w << '\n';


    if (algo=="PRANGE"){
      subISD_prange prange;
      ISD_single_generic<subISD_prange> ISD_single_prange(prange);
      ISD_single_prange.initialize(H, S, w);
      try {
          ISD_single_prange.solve();
      }
      catch(Solution& sol) {
        std::cout << "Solution found:" << '\n';
        std::cout << sol.get_solution() << '\n';
      }
    }
    else if (algo=="LEE BRICKEL"){
      subISD_LB subLB;
      ISD_single_generic<subISD_LB> ISD_single_LB(subLB);
      ISD_single_LB.initialize(H, S, w);
      try {
        ISD_single_LB.solve();
      }
      catch(Solution& sol) {
        std::cout << "Solution found:" << '\n';
        std::cout << sol.get_solution() << '\n';
      }
    }
    else {
      std::cout << "Unknown algorithm" << '\n';
      return 1;
    }

    return 0;
}
