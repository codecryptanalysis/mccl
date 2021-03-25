#ifndef MCCL_TOOLS_PARSER_H
#define MCCL_TOOLS_PARSER_H

#include <mccl/contrib/string_algo.hpp>
namespace sa = string_algo;
#include <string>
#include <fstream>
#include <iostream>

#include <mccl/core/matrix_detail.hpp>
#include <mccl/core/matrix.hpp>

enum Marker {
	MARK_NONE, MARK_N, MARK_K, MARK_W, MARK_SEED, MARK_HT, MARK_ST
};

std::vector<bool> string_to_booleans(std::string& str) {
	std::vector<bool> V;
	for( char &c : str) {
		if(c=='0') V.push_back(false);
		if(c=='1') V.push_back(true);
	}
	return V;
}

template<typename data_t>
class Parser {
	public:
		~Parser() { free();}
		bool load_file(std::string filename);
		mccl::matrix_ref_t<data_t> get_H();
		mccl::vector_ref_t<data_t> get_S();
		size_t get_n() { return n; };
		size_t get_k() { return k; };
		size_t get_w() { return w; };
		size_t get_seed() { return seed; };
	private:
		//mccl::matrix_t<data_t> H;
		bool Nset=false, Kset=false, Wset=false, Seedset=false;
		size_t n, k, w, seed;
		std::vector<bool> ST;
		std::vector<std::vector<bool>> HT;
		mccl::matrix_t<data_t>* H_ptr = nullptr;
		mccl::vector_t<data_t>* S_ptr = nullptr;
		void free() {
			if( H_ptr != nullptr ) delete H_ptr;
			if( S_ptr != nullptr ) delete S_ptr;
		};
};

template<typename data_t>
mccl::matrix_ref_t<data_t> Parser<data_t>::get_H(){
	return mccl::matrix_ref_t<data_t>(*H_ptr);
}

template<typename data_t>
mccl::vector_ref_t<data_t> Parser<data_t>::get_S(){
	return mccl::vector_ref_t<data_t>(*S_ptr);
}


template<typename data_t>
bool Parser<data_t>::load_file(std::string filename) {
	std::ifstream f(filename);
	if(!f.good()) {
		std::cerr << "File " << filename << " does not exist" << std::endl;
		return false;
	}

	Nset=false, Kset=false, Wset=false, Seedset=false;
	std::string line;
	std::vector<std::string> HT_string;
	std::string ST_string="";
	Marker active = MARK_NONE;
	size_t linenr=0;
	while(std::getline(f, line)) {
		if(line.size()==0) continue;
		if( line[0]=='#' ) {
			// comment or marker
			sa::trim_left(line, "# ");
			if(line.size()==0) continue;
			if(line[0]=='n' or line[0]=='N') active=MARK_N;
			if(line[0]=='k' or line[0]=='K') active=MARK_K;
			if(line[0]=='w' or line[0]=='W') active=MARK_W;
			if(line[0]=='h' or line[0]=='H') active=MARK_HT;
			if(line[0]=='s' or line[0]=='S') active=MARK_ST;
			if(sa::starts_with(line, "seed")) active=MARK_SEED;
		} else if( line[0] >= '0' and line[0] <= '9' ) {
			switch(active) {
				case MARK_NONE: std::cerr << "Reading numbers while no marker active" << std::endl; return false;
				case MARK_N: n = std::stoi(line); Nset=true; break;
				case MARK_K: k = std::stoi(line); Kset=true; break;
				case MARK_W: w = std::stoi(line); Wset=true; break;
				case MARK_SEED: seed = std::stoi(line); Seedset=true; break;
				case MARK_HT: HT_string.push_back(line); break;
				case MARK_ST: ST_string=line; break;
			}
		} else {
			std::cerr << "Ignoring line: " << linenr << std::endl;
		}
		linenr++;
	}

	// postprocessing
	if(!Nset){ std::cerr << "No length set" << std::endl; return false; }
	if(n<=0) { std::cerr << "Length not positive" << std::endl; return false; } 

	if(!Kset){ std::cerr << "No dim set, defaulting to rate 1/2" << std::endl; k = n/2; }
	if(k<=0 or k >= n) { std::cerr << "Dimension not in correct range" << std::endl; return false; } 

	if(Wset and (w<0 or w>n)) { std::cerr << "Weight not in correct range" << std::endl; return false; }

	if(ST_string.size()>0) {
		ST = string_to_booleans(ST_string);
		if(ST.size() != n-k) { std::cerr << "s doesn't have the correct length" << std::endl; return false; }
	}

	if(HT_string.size() != k) { std::cerr << "H^transpose is missing some rows" << std::endl; return false; }
	HT.clear();
	for(auto &str : HT_string) {
		auto row = string_to_booleans(str);
		if(row.size()!=n-k) { std::cerr << "Row has wrong length." << std::endl; return false; };
		HT.push_back(row);
	}

	free();
	H_ptr = new mccl::matrix_t<data_t>(n-k, n);
	H_ptr->setidentity();
	for( size_t r = 0; r < n-k; r++) {
		for( size_t c = 0; c < k; c++ ) {
			if(HT[c][r])
				H_ptr->bitset(r, n-k+c);
		}
	}

	S_ptr = new mccl::vector_t<data_t>(n-k);
	for( size_t r = 0; r < n-k; r++ ) {
		if(ST[r])
			S_ptr->bitset(r);
	}
	return true;
}

#endif
