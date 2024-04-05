# M C C L

Modular Code Cryptanalysis Library

[![C++ CI](https://github.com/codecryptanalysis/mccl/actions/workflows/cpp-ci.yml/badge.svg)](https://github.com/codecryptanalysis/mccl/actions/workflows/cpp-ci.yml)

# Documentation

Please review documentation and help improve it!
We're also looking for tutorials by and for people that are new to the project.

https://github.com/codecryptanalysis/mccl/tree/main/doc

# Algorithms implemented

- LeeBrickell
- Prange
- Stern/Dumer
- MMT

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

# Prerequisites

Autoconf:
```
sudo apt install autoconf
```

Libtool library:
```
sudo apt-get install libtool
```

Building tools (GCC and the rest):
```
sudo apt update
sudo apt install build-essential
```

GMP:
```
sudo apt-get install libgmp3-dev
```

# Getting started

```
git clone git@github.com:cr-marcstevens/mccl.git mccl
cd mccl
autoreconf --install
./configure
make
make check
```
