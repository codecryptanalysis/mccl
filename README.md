# M C C L

Modular Code Cryptanalysis Library

[![C++ CI](https://github.com/cr-marcstevens/mccl/actions/workflows/cpp-ci.yml/badge.svg)](https://github.com/cr-marcstevens/mccl/actions/workflows/cpp-ci.yml)

# Documentation

Please review documentation and help improve it!

https://github.com/cr-marcstevens/mccl/tree/main/doc

# Workshops

See https://github.com/cr-marcstevens/mccl-wiki

# Repository structure

- `README.md`: this readme
- `LICENSE`: the open-source MIT license
- `doc`: documentation
- `mccl`: library source files
  - `algorithm`: decoding algorithms
  - `core`: low-level objects
  - `config`: general C++ and macro definitions
  - `tools`: stuff that makes the library more convenient
  - `contrib`: external header-only libraries that we use
- `src`: program source files
- `tests`: tests source files
- `tools`: additional tools for mccl project
  - `update_contrib.sh`: update contrib files
- `m4`, `configure.ac`, `Makefile.am`: autotools build system files

# Building

```
autoreconf --install
./configure
make
make check
```
