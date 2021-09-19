# Matrices & Vectors design

This library is designed for matrices and vectors over F_2, i.e., bit matrices and bit vectors.
All the bits in a row or vector are stored contiguously in memory packed in unsigned 64-bit integers `uint64_t` named words.
Matrices and vectors may have arbitrary number of columns (and rows), however the first column always corresponds to the least significant bit of the first word of each row.
Hence, while submatrices and -vectors are supported, they must start on a column position that is a multiple of 64.
To ensure that modifications of a submatrix do not affect other unintended columns in the larger matrix, the library uses a bitmask for the last word of each row.

## Matrix and vector types

Matrices are quite large and copying submatrices and vectors should be avoided if possible.
Hence, we mainly work through classes that capture the specification of a (sub)matrix/vector located in memory called matrix/vector views.
Moreover, the libraries function will therefore take any destination as input parameter, instead of returning a new matrix/vector.

The following types are the default matrix & vector types:
- `mat`: a matrix object that allocates and maintains memory
- `mat_view`: a view to a non-const (sub)matrix
- `cmat_view`: a view to a const (sub)matrix
- `vec_view_it`: a view to a non-const (sub)vector that can iterate over a (sub)matrix rows
- `cvec_view_it`: a view to a const (sub)vector that can iterate over a (sub)matrix rows
- `vec`: a vector object that allocates and maintains memory
- `vec_view`: a view to a non-const (sub)vector
- `cvec_view`: a view to a const (sub)vector

## View objects

View objects behave like pointers:
- (1) copy/move constructor copies the view, not the contents. Therefore copy/move assignment from matrix/vector objects and views has been deleted, to avoid confusion. 
- (2) const-ness of the view only dictates whether the view can be modified or not, and not that of the underlying contents. Thus we have non-const & const matrix/vector view versions: `mat_view` vs `cmat_view`, `vec_view_it` vs `cvec_view_it` and `vec_view` vs `cvec_view`.
It follows that a const view to a non-const matrix/vector may still be used to modify its contents, 
while a non-const view to a const matrix/vector cannot be used to modify its contents.

Otherwise, view objects behave mostly like a matrix/vector object when it comes to matrix/vector operations.
They can be passed to all functions that take a matrix/vector and have all basic matrix/vector member functions.

## Result objects

Matrices and vectors have member functions to modify their contents, e.g. `m.m_and(m1,m2);` assigns the result of the bit-wise and of `m1` and `m2` to the matrix (view) `m`.
For convenience, this can also be written as `m = m_and(m1,m2);`.
(note: we use `v_and` and `m_and`, since `and`, `xor`, `or` are C++ reserved key words and cannot be used as names for member functions.)

Internally, the expression `m_and(m1,m2)` returns a `matrix_result` object that captures `m1`, `m2` and the intended member function call `m.m_and` (for a yet unknown matrix (view) `m`). When that `matrix_result` is assigned to the matrix (view) `m`, `m` passes itself to the `matrix_result` which then finally calls the intended function.

-----

## Matrix member functions

### Common const matrix member functions
- `columns()`: nr of columns
- `rows()`: nr of rows
- `hw()`: returns the hammingweight
- `word_ptr(r = 0)`: returns a `[const] uint64_t*` to the first word of row `r`
- `row_words()`: nr of words in a row
- `word_stride()`: nr of words to jump to the next row
- `operator()(r,c)`: returns the bit (as `bool`) on row `r` column `c` of `mat`
- `is_equal(m)`: returns a `bool` whether the two matrices have equal contents
- `subvector(r)`: returns a vector view for whole row `r` (see below: has same block_tag as matrix)
- `subvector(r, cols)`: return a vector view for the first `cols` columns of `r` (see below: has block_tag with same bits as matrix and maskedlastblock = true)
- `subvector(r, coloff, cols)`: returns a vector (const) view to row `r`, columns `coloff`,...,`coloff+cols-1` (see below: has default_block_tag)
- `submatrix(rowoff, rows)`: returns a matrix (const) view to rows `rowoff`,...,`rowoff+rows-1` and all columns (see below: has same block_tag as matrix)
- `submatrix(rowoff, rows, cols)`: returns a matrix (const) view to rows `rowoff`,...,`rowoff+rows-1` and columns `0`,...,`coloff+cols-1` (see below: has block_tag with same bits as matrix and maskedlastblock = true) 
- `submatrix(rowoff, rows, coloff, cols)`: returns a matrix (const) view to rows `rowoff`,...,`rowoff+rows-1` and columns `coloff`,...,`coloff+cols-1` (see below: has default_block_tag)
- `operator[](r)`, `operator()(r)`: returns a vector iterator (const) view to row `r` (see below: has same block_tag as matrix)
- `begin()`, `end()`: returns a vector iterator (const) view to row `0` and `rows()` (see below: has same block_tag as matrix)

Note that actual const-ness of the return type may depend on the const-ness of the underlying matrix contents.
I.e., const for `const mat`, `cmat_view`, and non-const for `mat` and `mat_view`.

### Common matrix member functions:
- `m_clear()`, `m_set(b = true)`: clear matrix to 0, set all bits of matrix to `b`
- `set_identity()`: clear matrix and set the diagonal bits
- `m_not()`: invert all bits
- `clearbit(r, c)`, `flipbit(r,c)`, `setbit(r,c)`, `setbit(r,c,b)`: modify bit a row `r` column `c` (clear to 0, flip, set to 1, set to `b`)
- `clearcolumns(c_off, c_cnt)`, `flipcolumns(c_off,c_cnt)`, `setcolumns(c_off,c_cnt)`, `setcolumns(c_off,c_cnt,b)`: modify columns `c_off`,..,`c_off+c_cnt-1`: clear to 0, flip, set to 1, set to `b`
- `swapcolumns(c1, c2)`: swap columns `c1` and `c2`
- `m_copy(m)`: assign contents of `m`
- `m_not(m)`: assign contents of `m` and invert all bits
- `transpose(m)`: assign the transposed contents of `m`
- `m_xor(m)`, `m_and(m)`, `m_or(m)`: assign the xor/and/or of this matrix with `m`
- `m_xor(m1,m2)`, `m_and(m1,m2)`, `m_or(m1,m2)`: assign the xor/and/or of `m1` and `m2`
- `operator^=(m)`, `operator&=(m)`, `operator|=(m)`: assign the xor/and/or of this matrix with `m`

### Additional `mat_view`, `cmat_view` member functions:
- `operator=(matrix_result)`: assign the matrix result
- `reset(mv)`: assign the view `mv`

### Additional `mat` member functions:
- `resize(rows,cols,b=false)`: resizes matrix, set new bits to `b`
- `operator=(m)`: resize, and assign the contents of m
- `operator=(matrix_result)`: resize, and assign the matrix result
- `operator=(mat&& m)`, `swap(mat& m)`: swap underlying contents with another `mat` object

### Global matrix functions:
- `operator==/!=(const matrix& m1, const matrix& m2)`: compares *content* of m1 & m2
- `operator<<(std::ostream& o, m)`: prints `m` to ostream `o`
- `m_transpose(m)`: returns `matrix_result` object to assign the transposed contents of `m`
- `m_copy(m), m_not(m)`: returns `matrix_result` object to assign the contents of `m` (as is / inverting all bits)
- `m_and(m1,m2), m_or(m1,m2), m_xor(m1,m2)`: returns a `matrix_result` to assign the contents of the and/or/xor of `m1` and `m2`
- `operator & (m1,m2), operator | (m1,m2), operator ^ (m1,m2)`: returns a `matrix_result` to assign the contents of the and/or/xor of `m1` and `m2`
- `hammingweight(m)`: returns the hammingweight of `m`

-----

## Vector member functions

### Common const vector member functions
- `columns()`: nr of columns
- `hw()`: returns the hammingweight
- `word_ptr(r = 0)`: returns a `[const] uint64_t*` to the first word
- `row_words()`: nr of words in a row
- `operator[](c), operator()(c)`: returns the bit (as `bool`) on column `c`
- `is_equal(v)`: returns a `bool` whether the two vectors have equal contents
- `subvector()`: returns the whole vector (see below: has same block_tag as vector)
- `subvector(cols)`: returns a vector (const) view to columns `0`,...,`coloff+cols-1` (see below: has block_tag with same bits as vector and maskedlastblock = true)
- `subvector(coloff, cols)`: returns a vector (const) view to columns `coloff`,...,`coloff+cols-1` (see below: has default_block_tag)

Note that actual const-ness of the return type may depend on the const-ness of the underlying vector contents.
I.e., const for `const vec`, `cvec_view`, `cvec_view_it`, and non-const for `vec`, `vec_view`, `vec_view_it`.

### Common vector member functions:
- `v_clear()`, `v_set(b = true)`: clear vector to 0, set all bits of vector to `b`
- `v_not()`: invert all bits
- `clearbit(c)`, `flipbit(c)`, `setbit(c)`, `setbit(c,b)`: modify bit in column `c` (clear to 0, flip, set to 1, set to `b`)
- `clearcolumns(c_off, c_cnt)`, `flipcolumns(c_off,c_cnt)`, `setcolumns(c_off,c_cnt)`, `setcolumns(c_off,c_cnt,b)`: modify columns `c_off`,..,`c_off+c_cnt-1`: clear to 0, flip, set to 1, set to `b`
- `v_swap(v)`: swap contents with `v`
- `v_copy(v)`: assign contents of `v`
- `v_not(v)`: assign contents of `v` and invert all bits
- `v_xor(v)`, `v_and(v)`, `v_or(v)`: assign the xor/and/or of this vector with `v`
- `v_nxor(v)`, `v_nand(v)`, `v_nor(v)`: assign the nxor/nand/nor of this vector with `v`
- `v_andin(v)`, `v_andni(v)`, `v_orin(v)`, `v_orni(v)`: assign the andin/andni/orin/orni of this vector with `v`
- `v_xor(v1,v2)`, `v_and(v1,v2)`, `v_or(v1,v2)`: assign the xor/and/or of `v1` and `v2`
- `v_nxor(v1,v2)`, `v_nand(v1,v2)`, `v_nor(v1,v2)`: assign the nxor/nand/nor of `v1` and `v2`
- `v_andin(v1,v2)`, `v_andni(v1,v2)`, `v_orin(v1,v2)`, `v_orni(v1,v2)`: assign the andin/andni/orin/orni of `v1` and `v2`
- `operator^=(v)`, `operator&=(v)`, `operator|=(v)`: assign the xor/and/or of this vector with `v`

Note that `andin(b1,b2)=b1 & (!b2)`, `andni(b1,b2)=(!b1) & b2`, `orin(b1,b2)=b1 | (!b2)`, `orni(b1,b2)=(!b1) | b2` (the two additional letters stand for the modification  `i=identity` & `n=not` of the respective input).

### Additional `vec_view`, `cview_view`, `vec_view_it`, `cvec_view_it` member functions:
- `operator=(vector_result)`: assign the vector result
- `reset(vv)`: assign the view `vv`

### Additional `vec_view_it`, `cvec_view_it` member functions:
- `word_stride()`: nr of words to jump to the next row
- `operator++()`, `operator--()`: jump to next / previous row
- `operator+=(n)`, `operator-=(n)`: jump to `n`-th next / previous row
- `operator++(int)`, `operator--(int)`: jump to next / previous row, but return copy of original value
- `operator+(n)`, `operator-(n)`: return a new vector iterator to `n`-th next / previous row
- `operator-(vit)`: return difference in nr of rows between this vector iterator and `vit`
- `operator*()`: returns `*this`, such that it can be used as iterator
- `operator->()`: returns `this`, such that it can be used as iterator

### Additional `vec` member functions:
- `resize(cols,b=false)`: resizes vector, set new bits to `b`
- `operator=(v)`: resize, and assign the contents of `v`
- `operator=(vector_result)`: resize, and assign the vector result
- `operator=(vec&& m)`, `swap(vec& m)`: swap underlying contents with another `vec` object

### Global vector functions:
- `operator==/!=(const vector& v1, const vector& v2)`: compares *content* of v1 & v2
- `operator<<(std::ostream& o, v)`: prints `v` to ostream `o`
- `v_copy(v), v_not(v)`: returns `vector_result` object to assign the contents of `v` (as is / inverting all bits)
- `v_and(v1,v2), v_or(v1,v2), v_xor(v1,v2)`: returns a `vector_result` to assign the contents of the and/or/xor of `v1` and `v2`
- `v_nand(v1,v2), v_nor(v1,v2), v_nxor(v1,v2)`: returns a `vector_result` to assign the contents of the nand/nor/nxor of `v1` and `v2`
- `v_andin(v1,v2), v_andni(v1,v2), v_orin(v1,v2), v_orni(v1,v2)`: returns a `vector_result` to assign the contents of the andin/andni/orin/orni of `v1` and `v2`
- `operator & (v1,v2), operator | (v1,v2), operator ^ (v1,v2)`: returns a `vector_result` to assign the contents of the and/or/xor of `v1` and `v2`
- `hammingweight(v)`: returns the hammingweight of `v`
- `hammingweight_and(v1,v2)`, `hammingweight_or(v1,v2)`, `hammingweight_xor(v1,v2)`: returns the hammingweight of the and/or/xor of `v1` and `v2`

# Optimizing for SIMD operations: block_tag<bits,maskedlastword>

### Block processing tags

By default the library operates on `uint64_t` words and uses a bitmask for the last word of each row/vector.
However, matrix and vector operations can typically benefit from SIMD operating on several `uint64_t` at a time.
Moreover, in some cases it is also not necessary to use a bitmask for the last word.
To that end, the library introduces special *tags*, i.e. trivial empty structs whose type carries compile time information:
```
template<size_t _bits, bool _maskedlastblock>
struct block_tag {
        typedef block_tag<_bits, _maskedlastblock> type;
        static const size_t bits = _bits;                     // must be 64, 128, 256, or 512
        static const bool maskedlastblock = _maskedlastblock; // whether to use a bitmask for the last block
};
```

First a warning!: using `maskedlastblock=false` avoids a bitmask for the last word and is dangerous.
It should only be used if you don't care about the value in the remaining bits.
But do not make any assumptions about the value in those remaining bits.
Note that regardless of the value of `maskedlastblock`, the `hammingweight` and `isequal` functions will still use a bitmask to guarantee correct answers.

Matrix and vector operations support matrices and vectors with different tags. In those cases, the operations will use the smallest block size of the inputs and the `maskedlastblock` value of the destination. I.e., even if an input uses `maskedlastblock=true`, when the destination has `maskedlastblock=false` then no bitmask will be used for the lastblock. 

### Generalized matrix and vector types

The library actually defines generalized types for matrix and vector and views templated on the block_tag. The above default types are specializations thereof using `default_block_tag = block_tag<64,true>`:
- `mat = mat_t<block_tag<256,false>>`
- `vec = vec_t<block_tag<256,false>>`
- `mat_view = mat_view_t<default_block_tag>`
- `cmat_view = cmat_view_t<default_block_tag>`
- `vec_view_it = vec_view_it_t<default_block_tag>`
- `cvec_view_it = cvec_view_it_t<default_block_tag>`
- `vec_view = vec_view_t<default_block_tag>`
- `cvec_view = cvec_view_t<default_block_tag>`

Note that `mat` and `vec` are instantiated using `block_tag<256,false>` and by default may benefit from SIMD.
Both ensure aligned rows / vectors on 512-bit boundaries regardless of the tag, so can also be instantiated up to `block_tag<512,false>`.
However, as 256-bit SIMD is prevalent and 512-bit SIMD is not, using `block_tag<512,false>` may result in some unnecessary overhead in many cases.

Regardless of the tag of a type, the member functions `subvector` or `submatrix` always return a view with `default_block_tag` for safety:
it cannot know at compile time whether the starting column is aligned on a block (but it must at least be 64-bit aligned), 
nor whether it includes the last column (thus it must use `maskedlastblock=true` by default to avoid undesired results). 

This can be overruled by appending a tag as parameter: e.g. `subvector( ..., block_tag<bits,mask>() )`.

In contrast, matrix member functions `operator[](r)`, `begin()`, `end()`, that return vector iterators to an entire row, will have the same `block_tag` as the matrix. This allows operations on entire rows to indeed benefit from SIMD.

### Additional block processing related member functions

In extension of `word_ptr(..)`, `row_words()`, `word_stride()` there are also analogous member functions for blocks:
- `block_ptr()`: return a pointer to the first `bits`-bit block of a row / vector
- `row_blocks()`: nr of blocks in a row
- `block_stride()`: nr of blocks to jump to the next row

The used block tag can also be overridden for the following functions, by appending a different tag as parameter `( ..., block_tag<bits2,mask2>() )`:
- `block_ptr()`
- `row_blocks()`
- `block_stride()`
- `subvector()`
- `submatrix()`

Finally, to overrule the current tag of an object: `m.as(block_tag<bits2,masked2>())` returns a reference to itself with an altered type using the specified tag.
Thus `m.block_ptr(block_tag<bits2,masked2>())` has the same result as `m.as(block_tag<bits2,masked2>()).block_ptr()`.
