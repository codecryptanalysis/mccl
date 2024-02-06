with import <nixpkgs> {};
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "cryptanalysislib";
  src = ./.;

  buildInputs = [ 
    gmp
	libtool 
	autoconf 
	automake 
	autogen 
	gnumake 
	clang_16
	gcc
  ] ++ (lib.optionals pkgs.stdenv.isLinux ([
	flamegraph
	gdb
    linuxKernel.packages.linux_6_5.perf
	pprof
	valgrind
	massif-visualizer
  ]));
}