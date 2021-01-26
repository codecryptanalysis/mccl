# Low-level framework design

# Matrix class design

Matrices are quite large and copying submatrices and vectors should be avoided if possible.
Hence, we mainly work through classes that capture the specification of a (sub)matrix located in memory.
Moreover, the libraries function will therefore take any destination matrix reference as input parameter, instead of returning a new matrix.

- The library-internal base class for a (sub)matrix specification is `detail::matrix_base_ref_t<word_t>` in `mccl/core/matrix_detail.hpp`.
  - The type `word_t` can be any unsigned integer type, e.g. uint64_t.
  - Columns must start aligned on `word_t`, i.e., the first column of any row is bit 0 of the first word of any row.
  - Actual columns are potentially followed by scratchcolumns, which may be potentially altered in any operation. They are ignored in comparison etc.
  - Total number of columns must be a multiple of the bits in `word_t`.
  
- The class `matrix_ref_t<word_t>` is a wrapper around the above internal struct with reference semantics:
  - The constructor will initialize the reference
  - Any operations on `matrix_ref_t<word_t>` will act on the contents:
    - operators: `=`, `^=`, `|=`, `&=`, `==`, `!=`, `[](row)`, `()(row,column)`
    - unary operation functions: `m.op_not()` (m=~m), `m.op_not(m2)` (m=~m2)
    - binary operation functions: `m.op_and(m2)` (m=m&m2), `m.op_and(m1,m2)` (m=m1&m2), `op_xor`, `op_or`, etc.
  - has `iterator` and `const_iterator` that iterate over row vectors
  - Exceptions are the `reset*()` member functions which allow to alter the reference.
- the class `matrix_ptr_t<word_t>` is also a wrapper around `matrix_base_ref_t` but acts with pointer semantics
  - dereferencing returns a reference to `matrix_ref_t<word_t>` (which is a reinterpretation cast of `matrix_ptr_t<word_t>`!)
- similar classes `vector_ref_t<word_t>` and `vector_ptr_t<word_t>` that specify a single row vector. They can be automatically converted to their matrix versions. Operations are similar but more specific to vectors:
  - ref operator `[](column)` returns a bit
  - ptr operators ++, --, +=, -=, +, - operate on the underlying word_t pointer in steps of `stride`, i.e., iterates over contiguous row vectors in memory

- The class `matrix_t<word_t>` is derived from `matrix_ref_t<word_t>` and inherets almost all operations and member functions
  - However, it allocates and manages the memory where the matrix is stored.
  - `reset*()` functions are made private.
