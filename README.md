# mccl
modular code cryptanalysis library

# Documentation

Please review documentation and help improve it!

https://github.com/cr-marcstevens/mccl/tree/main/doc

# Core API

I've made first steps on the matrix API:

https://github.com/cr-marcstevens/mccl/blob/main/mccl/core/matrix.hpp

# Repository structure

- `README.md`: this readme
- `LICENSE`: the open-source MIT license
- `doc`: documentation
- `mccl`: library headers
  - `algorithm`: decoding algorithms
  - `core`: low-level objects
  - `config`: general C++ and macro definitions
  - `tools`: stuff that makes the library more convenient
- `src`: library source files
  - `algorithm`: decoding algorithms
  - `core`: low-level objects
  - `tools`: stuff that makes the library more convenient
- `m4`, `configure.ac`, `Makefile.am`: autotools build system files
