#ifndef MCCL_CORE_MATRIX_HPP
#define MCCL_CORE_MATRIX_HPP

#include <mccl/core/matrix_detail.hpp>
#include <stdlib.h>

MCCL_BEGIN_NAMESPACE

template<typename data_t>
class vector_ptr_t;
template<typename data_t>
class vector_ref_t;
template<typename data_t>
class matrix_ptr_t;
template<typename data_t>
class matrix_ref_t;

template<typename data_t>
class vector_t;
template<typename data_t>
class matrix_t;

/*
    1) matrix_base_ref_t: struct containing pointer to submatrix (ptr, rows, stride, columns, scratchcolumns)
    (note: scratchcolumns are 'extra' columns that potentially may be altered, but are ignored as content)

    2) vector_ptr_t, vector_ref_t, matrix_ptr_t, matrix_ref_t are all simple wrappers around matrix_base_ref_t
    which itself is a pointer struct, but with different semantics:
    - matrix_ref_t, vector_ref_t have reference semantics: constructor creates reference, all operations act on the content
    - matrix_ptr_t, vector_ptr_t have pointer semantics: dereference just recasts *_ptr_t object as *_ref_t&
    - vector_ref_t, vector_ptr_t are almost equal to matrix_*_t, but have forced rows=1 and index operators are different as expected

    3) Reusing the same matrix_base_ref_t when dereferencing *_ptr_t is for efficiency, avoiding unnecessary copyies.
    The main operations on *_ref_t do not change the underlying matrix_base_ref_t and are safe to use.

    4)
    *** THE USE OF THESE ADDITIONAL MEMBER FUNCTIONS OF matrix_ref_t/vector_ref_t MAY CAUSE CONSTNESS VIOLATIONS AND SHOULD BE USE WITH CARE:
    ***    reset_*, as_ptr(), to_ref()
    *** This is due the fact that dereferencing a const pointer returns a non-const reference,
    *** hence for `const matrix_ref_t m`, the expression `*m.as_ptr()` returns a non-const `matrix_ref_t&` to m
    *** similarly for `const matrix_ptr_t p`, the expression `p->reset(base)` changes p
    
    5) matrix_t, vector_t: class derived from matrix_ref_t/vector_ref_t
    - adds memory allocation
    - does not allow to change the underlying *_ref_t: hides reset_*(), as_ptr(), base() functions

*/

// reference semantics: 
// constructors initialize the reference, other member functions/operators change the referenced object
template<typename data_t>
class vector_ref_t
{
public:
    typedef detail::matrix_base_ref_t<      data_t> base_ref_t;
    typedef detail::matrix_base_ref_t<const data_t> cbase_ref_t;
    typedef vector_ptr_t<data_t> vector_ptr;
    typedef vector_ref_t<data_t> vector_ref;
    typedef matrix_ptr_t<data_t> matrix_ptr;
    typedef matrix_ref_t<data_t> matrix_ref;

    /* constructors: initialize the reference */
    // default constructor & move constructor
    vector_ref_t() = default;
    vector_ref_t(vector_ref_t&& m) : _base(m._base) {}
    // do not allow const copy constructor, but allow non-const copy constructor
    vector_ref_t(const vector_ref_t& m) = delete;
    vector_ref_t(vector_ref_t& m) : _base(m._base) {}
    // construct from base_ref_t
    explicit vector_ref_t(const base_ref_t& m)
        : _base(m)
    {
        MCCL_MATRIX_BASE_ASSERT(_base.rows == 1);
    }
    explicit vector_ref_t(base_ref_t&& m)
        : _base(std::move(m))
    {
        MCCL_MATRIX_BASE_ASSERT(_base.rows == 1);
    }
    vector_ref_t(data_t* ptr, size_t columns, size_t scratchcolumns = 0, size_t stride = 0)
        : _base(ptr, 1, columns, scratchcolumns, stride)
    {
    }

    size_t rows()           const { return 1; }
    size_t columns()        const { return base().columns; }
    size_t scratchcolumns() const { return base().scratchcolumns; }
    size_t allcolumns()     const { return base().columns + base().scratchcolumns; }
    size_t stride()         const { return base().stride; }

    // assign & modify operators
    // reference semantics: (override) assign and modify&assign by acting on matrix content
    vector_ref_t& operator=(const vector_ref_t& m) { detail::vector_copy(base(), m.base()); return *this; }
    vector_ref_t& operator=(vector_ref_t&&) = delete;

    vector_ref_t& operator^=(const vector_ref_t& m2) { detail::vector_xor(base(), m2.base()); return *this; }
    vector_ref_t& operator|=(const vector_ref_t& m2) { detail::vector_or(base(), m2.base()); return *this; }
    vector_ref_t& operator&=(const vector_ref_t& m2) { detail::vector_and(base(), m2.base()); return *this; }

    // *this = ~(*this)
    vector_ref_t& op_not()                       { detail::vector_not(base()); return *this; }
    // *this = ~m2
    vector_ref_t& op_not(const vector_ref_t& m2) { detail::vector_copynot(base()); return *this; }
    // perform binary operation: *this = op(*this, m2)
    vector_ref_t& op_and  (const vector_ref_t& m2) { detail::vector_and  (base(), m2.base()); return *this; } // a & b
    vector_ref_t& op_xor  (const vector_ref_t& m2) { detail::vector_xor  (base(), m2.base()); return *this; } // a ^ b
    vector_ref_t& op_or   (const vector_ref_t& m2) { detail::vector_or   (base(), m2.base()); return *this; } // a | b
    vector_ref_t& op_nand (const vector_ref_t& m2) { detail::vector_nand (base(), m2.base()); return *this; } // ~(a&b)
    vector_ref_t& op_nxor (const vector_ref_t& m2) { detail::vector_nxor (base(), m2.base()); return *this; } // ~(a^b)
    vector_ref_t& op_nor  (const vector_ref_t& m2) { detail::vector_nor  (base(), m2.base()); return *this; } // ~(a|b)
    vector_ref_t& op_andin(const vector_ref_t& m2) { detail::vector_andin(base(), m2.base()); return *this; } // a&~b
    vector_ref_t& op_andni(const vector_ref_t& m2) { detail::vector_andni(base(), m2.base()); return *this; } // ~a&b
    vector_ref_t& op_orin (const vector_ref_t& m2) { detail::vector_orin (base(), m2.base()); return *this; } // a|~b
    vector_ref_t& op_orni (const vector_ref_t& m2) { detail::vector_orni (base(), m2.base()); return *this; } // ~a|b
    // perform binary operation: *this = op(m1, m2)
    vector_ref_t& op_and  (const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_and  (base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_xor  (const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_xor  (base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_or   (const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_or   (base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_nand (const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_nand (base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_nxor (const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_nxor (base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_nor  (const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_nor  (base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_andin(const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_andin(base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_andni(const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_andni(base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_orin (const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_orin (base(), m1.base(), m2.base()); return *this; }
    vector_ref_t& op_orni (const vector_ref_t& m1, const vector_ref_t& m2) { detail::vector_orni (base(), m1.base(), m2.base()); return *this; }

    // comparison operators
    bool operator==(const vector_ref_t& m) const { return vector_compare(base(), m.base()); }
    bool operator!=(const vector_ref_t& m) const { return !vector_compare(base(), m.base()); }

    // indexing
    bool operator[](size_t c) const { return base()(0, c); }
    bool operator()(size_t c) const { return base()(0, c); }

    /* reset the reference */
    void reset(const base_ref_t& m)
    {
        base() = m;
        MCCL_MATRIX_BASE_ASSERT(base().rows == 1);
    }
    void reset(data_t* ptr, size_t columns, size_t scratchcolumns = 0, size_t stride = 0)
    {
        base().reset(ptr, 1, columns, scratchcolumns, stride);
    }
    void reset_subvector(size_t column_offset, size_t columns, size_t scratchcolumns = 0)
    {
        base().reset_subvector(0, column_offset, columns, scratchcolumns);
    }
    vector_ref_t subvector(size_t column_offset, size_t columns, size_t scratchcolumns = 0)
    {
        return vector_ref_t(base().subvector(0, column_offset, columns, scratchcolumns));
    }

    /* clear scratch columns with value */
    void clear_scratchcolumns(bool value = false)
    {
        detail::vector_setscratch(base(), value);
    }

    // automatic conversions
    // a vector is a matrix, so allow automatic conversion to matrix_ref_t
    operator matrix_ref& () { return *reinterpret_cast<matrix_ref*>(this); }
    operator const matrix_ref& () const { return *reinterpret_cast<const matrix_ref*>(this); }

    // explicit conversion
          vector_ptr& as_ptr()       { return *reinterpret_cast<vector_ptr*>(this); }
    const vector_ptr& as_ptr() const { return *reinterpret_cast<vector_ptr*>(this); }
    vector_ptr ptr() { return vector_ptr(base()); }

    base_ref_t& base() { return _base; }
    const cbase_ref_t& base() const { return _base.as_const(); }

private:
    // do not access _base directly, use `base()` to maintain reference semantics
    base_ref_t _base;
};



// pointer semantics
template<typename data_t>
class vector_ptr_t
{
public:
    typedef detail::matrix_base_ref_t<      data_t> base_ref_t;
    typedef detail::matrix_base_ref_t<const data_t> cbase_ref_t;
    typedef vector_ptr_t<data_t> vector_ptr;
    typedef vector_ref_t<data_t> vector_ref;
    typedef matrix_ptr_t<data_t> matrix_ptr;
    typedef matrix_ref_t<data_t> matrix_ref;

    /* constructors: initialize the pointer */
    // default constructor & move constructor
    vector_ptr_t() = default;
    vector_ptr_t(const vector_ptr_t&) = default;
    vector_ptr_t(vector_ptr_t&&) = default;
    // construct from base_ref_t
    explicit vector_ptr_t(const base_ref_t& m)
        : _base(m)
    {
        MCCL_MATRIX_BASE_ASSERT(_base.rows == 1);
    }
    explicit vector_ptr_t(base_ref_t&& m)
        : _base(std::move(m))
    {
        MCCL_MATRIX_BASE_ASSERT(_base.rows == 1);
    }
    vector_ptr_t(data_t* ptr, size_t columns, size_t scratchcolumns = 0, size_t stride = 0)
        : _base(ptr, 1, columns, scratchcolumns, stride)
    {
    }

    // assign & modify operators
    vector_ptr_t& operator=(const vector_ptr_t& m) = default;
    vector_ptr_t& operator=(vector_ptr_t&&) = default;

    // comparison operators
    bool operator==(const vector_ptr_t& m) const { return _base == m._base; }
    bool operator!=(const vector_ptr_t& m) const { return _base != m._base; }

    // iterator steps
    vector_ptr_t& operator++() { _base.ptr += _base.stride; return *this; }
    vector_ptr_t& operator--() { _base.ptr -= _base.stride; return *this; }
    vector_ptr_t& operator+=(ptrdiff_t a) { _base.ptr += a * ptrdiff_t(_base.stride); return *this; }
    vector_ptr_t& operator-=(ptrdiff_t a) { _base.ptr -= a * ptrdiff_t(_base.stride); return *this; }
    vector_ptr_t operator++(int) const { vector_ptr_t tmp(*this); ++tmp; return tmp; }
    vector_ptr_t operator--(int) const { vector_ptr_t tmp(*this); --tmp; return tmp; }
    vector_ptr_t operator+(ptrdiff_t a) const { vector_ptr_t tmp(*this); tmp += a; return tmp; }
    vector_ptr_t operator-(ptrdiff_t a) const { vector_ptr_t tmp(*this); tmp -= a; return tmp; }
    vector_ptr_t operator-(const vector_ptr_t& r) const { return (_base.ptr - r._base.ptr) / _base.stride; }

    // dereference operators
    vector_ref& operator*() const { return to_ref(); }
    vector_ref* operator->() const { return &to_ref(); }

    /* reset the reference */
    void reset(const base_ref_t& m)
    {
        _base = m;
        MCCL_MATRIX_BASE_ASSERT(_base.rows == 1);
    }
    void reset(data_t* ptr, size_t columns, size_t scratchcolumns = 0, size_t stride = 0)
    {
        _base.reset(ptr, 1, columns, scratchcolumns, stride);
    }
    void reset_subvector(size_t column_offset, size_t columns, size_t scratchcolumns = 0)
    {
        _base.reset_subvector(0, column_offset, columns, scratchcolumns);
    }
    vector_ptr_t subvector_ptr(size_t column_offset, size_t columns, size_t scratchcolumns = 0) const
    {
        return vector_ptr_t(_base.subvector(0, column_offset, columns, scratchcolumns));
    }
    vector_ref subvector_ref(size_t column_offset, size_t columns, size_t scratchcolumns = 0) const
    {
        return vector_ref(_base.subvector(0, column_offset, columns, scratchcolumns));
    }

    // automatic conversions
    // a vector is a matrix, so allow automatic conversion to matrix_ptr_t
    operator matrix_ptr& () { return *reinterpret_cast<matrix_ptr*>(this); }
    operator const matrix_ptr& () const { return *reinterpret_cast<const matrix_ptr*>(this); }

    // explicit conversion to vector_ref
    // - as_ref returns *this as vector_ref& and maintains constness
    // - to_ref returns *this as vector_ref&, but removes constness, as a pointer usually does
    // - ref returns a new vector_ref object
          vector_ref& as_ref()       { return *reinterpret_cast<vector_ref*>(this); }
    const vector_ref& as_ref() const { return *reinterpret_cast<vector_ref*>(this); }
    vector_ref& to_ref() const { return *reinterpret_cast<vector_ref*>(this); }
    vector_ref ref() const { return vector_ref(_base); }

private:
    base_ref_t _base;
};




// reference semantics: 
// constructors initialize the reference, other member functions/operators change the referenced object
template<typename data_t>
class matrix_ref_t
{
public:
    typedef detail::matrix_base_ref_t<      data_t> base_ref_t;
    typedef detail::matrix_base_ref_t<const data_t> cbase_ref_t;
    typedef vector_ptr_t<data_t> vector_ptr;
    typedef vector_ref_t<data_t> vector_ref;
    typedef matrix_ptr_t<data_t> matrix_ptr;
    typedef matrix_ref_t<data_t> matrix_ref;

    /* constructors: initialize the reference */
    // default constructor & move constructor
    matrix_ref_t() = default;
    matrix_ref_t(matrix_ref_t&& m) : _base(m._base) {}
    // do not allow const copy constructor, but allow non-const copy constructor
    matrix_ref_t(const matrix_ref_t& m) = delete;
    matrix_ref_t(matrix_ref_t& m) : _base(m._base) {}
    // construct from base_ref_t
    explicit matrix_ref_t(const base_ref_t& m) : _base(m) {}
    explicit matrix_ref_t(base_ref_t&& m) : _base(std::move(m)) {}
    matrix_ref_t(data_t* ptr, size_t rows, size_t columns, size_t scratchcolumns = 0, size_t stride = 0)
        : _base(ptr, rows, columns, scratchcolumns, stride)
    {
    }

    size_t rows()           const { return base().rows; }
    size_t columns()        const { return base().columns; }
    size_t scratchcolumns() const { return base().scratchcolumns; }
    size_t allcolumns()     const { return base().columns + base().scratchcolumns; }
    size_t stride()         const { return base().stride; }

    // assign & modify operators
    // reference semantics: (override) assign and modify&assign by acting on matrix content
    matrix_ref_t& operator=(const matrix_ref_t& m) { detail::matrix_copy(base(), m.base()); return *this; }
    matrix_ref_t& operator=(matrix_ref_t&&) = delete;

    matrix_ref_t& operator^=(const matrix_ref_t& m) { detail::matrix_xor(base(), m.base()); return *this; }
    matrix_ref_t& operator|=(const matrix_ref_t& m) { detail::matrix_or(base(), m.base()); return *this; }
    matrix_ref_t& operator&=(const matrix_ref_t& m) { detail::matrix_and(base(), m.base()); return *this; }

    // *this = ~(*this)
    matrix_ref_t & op_not() { detail::matrix_not(base()); return *this; }
    // *this = ~m2
    matrix_ref_t & op_not(const matrix_ref_t& m2) { detail::matrix_copynot(base()); return *this; }
    // perform binary operation: *this = op(*this, m2)
    matrix_ref_t& op_and  (const matrix_ref_t& m2) { detail::matrix_and  (base(), m2.base()); return *this; } // a & b
    matrix_ref_t& op_xor  (const matrix_ref_t& m2) { detail::matrix_xor  (base(), m2.base()); return *this; } // a ^ b
    matrix_ref_t& op_or   (const matrix_ref_t& m2) { detail::matrix_or   (base(), m2.base()); return *this; } // a | b
    matrix_ref_t& op_nand (const matrix_ref_t& m2) { detail::matrix_nand (base(), m2.base()); return *this; } // ~(a&b)
    matrix_ref_t& op_nxor (const matrix_ref_t& m2) { detail::matrix_nxor (base(), m2.base()); return *this; } // ~(a^b)
    matrix_ref_t& op_nor  (const matrix_ref_t& m2) { detail::matrix_nor  (base(), m2.base()); return *this; } // ~(a|b)
    matrix_ref_t& op_andin(const matrix_ref_t& m2) { detail::matrix_andin(base(), m2.base()); return *this; } // a&~b
    matrix_ref_t& op_andni(const matrix_ref_t& m2) { detail::matrix_andni(base(), m2.base()); return *this; } // ~a&b
    matrix_ref_t& op_orin (const matrix_ref_t& m2) { detail::matrix_orin (base(), m2.base()); return *this; } // a|~b
    matrix_ref_t& op_orni (const matrix_ref_t& m2) { detail::matrix_orni (base(), m2.base()); return *this; } // ~a|b
    // perform binary operation: *this = op(m1, m2)
    matrix_ref_t& op_and  (const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_and  (base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_xor  (const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_xor  (base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_or   (const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_or   (base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_nand (const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_nand (base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_nxor (const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_nxor (base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_nor  (const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_nor  (base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_andin(const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_andin(base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_andni(const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_andni(base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_orin (const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_orin (base(), m1.base(), m2.base()); return *this; }
    matrix_ref_t& op_orni (const matrix_ref_t& m1, const matrix_ref_t& m2) { detail::matrix_orni (base(), m1.base(), m2.base()); return *this; }

    matrix_ref_t& transpose(const matrix_ref_t& src) { detail::matrix_transpose(base(), src.base()); return *this; }
    
    // comparison operators
    bool operator==(const matrix_ref_t& m) const { return matrix_compare(base(), m.base()); }
    bool operator!=(const matrix_ref_t& m) const { return !matrix_compare(base(), m.base()); }

    // indexing
          vector_ref operator[](size_t r)       { return vector_ref(base().subvector(r, 0, columns(), scratchcolumns())); }
    const vector_ref operator[](size_t r) const { return vector_ref(base().subvector(r, 0, columns(), scratchcolumns())); }
    bool operator()(size_t r, size_t c) const { return base()(r, c); }

    /* reset the reference */
    void reset(const base_ref_t& m)
    {
        base() = m;
    }
    void reset(data_t* ptr, size_t rows, size_t columns, size_t scratchcolumns, size_t stride)
    {
        base().reset(ptr, rows, columns, scratchcolumns, stride);
    }
    void reset_submatrix(size_t row_offset, size_t rows, size_t column_offset, size_t columns, size_t scratchcolumns = 0)
    {
        base().reset_submatrix(row_offset, rows, column_offset, columns, scratchcolumns);
    }
    matrix_ref_t submatrix(size_t row_offset, size_t rows, size_t column_offset, size_t columns, size_t scratchcolumns = 0)
    {
        return matrix_ref_t(base().submatrix(row_offset, rows, column_offset, columns, scratchcolumns));
    }
    vector_ref submatrix(size_t row_offset, size_t column_offset, size_t columns, size_t scratchcolumns = 0)
    {
        return vector_ref(base().subvector(row_offset, column_offset, columns, scratchcolumns));
    }

    /* clear scratch columns with value */
    void clear_scratchcolumns(bool value = false)
    {
        detail::matrix_setscratch(base(), value);
    }

    /* iterators */
    template<typename T>
    class iterator_t: public std::iterator<std::random_access_iterator_tag, T>
    {
    public:
        typedef iterator_t<const T> const_iterator_t;

        iterator_t(const base_ref_t& m)
            : _vptr(m)
        {}
        iterator_t(const iterator_t& it) = default;
        iterator_t(iterator_t&& it) = default;
        iterator_t& operator=(const iterator_t& it) = default;
        iterator_t& operator=(iterator_t&& it) = default;

        T& operator*() const { return _vptr.operator*(); }
        T* operator->() const { return _vptr.operator->(); }

        bool operator==(const const_iterator_t& it) const { return _vptr == it._vptr; }
        bool operator!=(const const_iterator_t& it) const { return _vptr != it._vptr; }

        iterator_t& operator++()            { ++_vptr; return *this; }
        iterator_t& operator--()            { --_vptr; return *this; }
        iterator_t& operator+=(ptrdiff_t a) { _vptr += a; return *this; }
        iterator_t& operator-=(ptrdiff_t a) { _vptr -= a; return *this; }
        iterator_t  operator++(int)         const { iterator_t tmp(*this); ++tmp; return tmp; }
        iterator_t  operator--(int)         const { iterator_t tmp(*this); --tmp; return tmp; }
        iterator_t  operator+ (ptrdiff_t a) const { iterator_t tmp(*this); tmp += a; return tmp; }
        iterator_t  operator- (ptrdiff_t a) const { iterator_t tmp(*this); tmp -= a; return tmp; }
        iterator_t  operator- (const const_iterator_t& r) const { return _vptr - r._vptr; }

        operator const_iterator_t& ()       { return *reinterpret_cast<      const_iterator_t*>(this); }
        operator const_iterator_t& () const { return *reinterpret_cast<const const_iterator_t*>(this); }
    private:
        vector_ptr _vptr;
    };
    typedef iterator_t<vector_ref> iterator;
    typedef iterator_t<const vector_ref> const_iterator;

          iterator  begin()       { return       iterator(_base.subvector(0,      0, columns(), scratchcolumns())); }
          iterator  end()         { return       iterator(_base.subvector(rows(), 0, columns(), scratchcolumns())); }
    const_iterator  begin() const { return const_iterator(_base.subvector(0,      0, columns(), scratchcolumns())); }
    const_iterator  end()   const { return const_iterator(_base.subvector(rows(), 0, columns(), scratchcolumns())); }
    const_iterator cbegin() const { return const_iterator(_base.subvector(0,      0, columns(), scratchcolumns())); }
    const_iterator cend()   const { return const_iterator(_base.subvector(rows(), 0, columns(), scratchcolumns())); }

    base_ref_t& base() { return _base; }
    const cbase_ref_t& base() const { return _base.as_const(); }

private:
    // do not access _ref directly, use `ref()` to maintain reference semantics
    // exception is for iterators, they maintain reference semantics themselves
    base_ref_t _base;
};


// pointer semantics
template<typename data_t>
class matrix_ptr_t
{
public:
    typedef detail::matrix_base_ref_t<      data_t> base_ref_t;
    typedef detail::matrix_base_ref_t<const data_t> cbase_ref_t;
    typedef vector_ptr_t<data_t> vector_ptr;
    typedef vector_ref_t<data_t> vector_ref;
    typedef matrix_ptr_t<data_t> matrix_ptr;
    typedef matrix_ref_t<data_t> matrix_ref;

    /* constructors: initialize the pointer */
    // default constructor & move constructor
    matrix_ptr_t() = default;
    matrix_ptr_t(const matrix_ptr_t&) = default;
    matrix_ptr_t(matrix_ptr_t&&) = default;
    // construct from base_ref_t
    explicit matrix_ptr_t(const base_ref_t& m) : _base(m) {}
    explicit matrix_ptr_t(base_ref_t&& m) : _base(std::move(m)) {}
    matrix_ptr_t(data_t* ptr, size_t rows, size_t columns, size_t scratchcolumns = 0, size_t stride = 0)
        : _base(ptr, rows, columns, scratchcolumns, stride)
    {
    }

    // assign & modify operators
    matrix_ptr_t& operator=(const matrix_ptr_t& m) = default;
    matrix_ptr_t& operator=(matrix_ptr_t&&) = default;

    // comparison operators
    bool operator==(const matrix_ptr_t& m) const { return _base == m._base; }
    bool operator!=(const matrix_ptr_t& m) const { return _base != m._base; }

    // iterator steps
    matrix_ptr_t& operator++()            { _base.ptr += _base.stride * _base.rows; return *this; }
    matrix_ptr_t& operator--()            { _base.ptr -= _base.stride * _base.rows; return *this; }
    matrix_ptr_t& operator+=(ptrdiff_t a) { _base.ptr += a * ptrdiff_t(_base.stride * _base.rows); return *this; }
    matrix_ptr_t& operator-=(ptrdiff_t a) { _base.ptr -= a * ptrdiff_t(_base.stride * _base.rows); return *this; }
    matrix_ptr_t  operator++(int)        const { matrix_ptr_t tmp(*this); ++tmp; return tmp; }
    matrix_ptr_t  operator--(int)        const { matrix_ptr_t tmp(*this); --tmp; return tmp; }
    matrix_ptr_t  operator+(ptrdiff_t a) const { matrix_ptr_t tmp(*this); tmp += a; return tmp; }
    matrix_ptr_t  operator-(ptrdiff_t a) const { matrix_ptr_t tmp(*this); tmp -= a; return tmp; }
    matrix_ptr_t  operator-(const matrix_ptr_t& r) const { return (_base.ptr - r._base.ptr) / (_base.stride * _base.rows); }

    // dereference operators
    matrix_ref& operator*() const { return to_ref(); }
    matrix_ref* operator->() const { return &to_ref(); }

    /* reset the reference */
    void reset(const base_ref_t& m)
    {
        _base = m;
    }
    void reset(data_t* ptr, size_t rows, size_t columns, size_t scratchcolumns, size_t stride)
    {
        _base.reset(ptr, rows, columns, scratchcolumns, stride);
    }
    void reset_submatrix(size_t row_offset, size_t rows, size_t column_offset, size_t columns, size_t scratchcolumns = 0)
    {
        _base.reset_submatrix(row_offset, rows, column_offset, columns, scratchcolumns);
    }
    matrix_ptr_t subvector_ptr(size_t column_offset, size_t columns, size_t scratchcolumns = 0) const
    {
        return matrix_ptr_t(_base.subvector(0, column_offset, columns, scratchcolumns));
    }
    vector_ref subvector_ref(size_t column_offset, size_t columns, size_t scratchcolumns = 0) const
    {
        return vector_ref(_base.subvector(0, column_offset, columns, scratchcolumns));
    }

    // explicit conversion to matrix_ref
    // - as_ref returns *this as matrix_ref& and maintains constness
    // - to_ref returns *this as matrix_ref&, but removes constness, as a pointer usually does
    // - ref returns a new matrix_ref object
    matrix_ref& as_ref() { return *reinterpret_cast<matrix_ref*>(this); }
    const matrix_ref& as_ref() const { return *reinterpret_cast<matrix_ref*>(this); }
    matrix_ref& to_ref() const { return *reinterpret_cast<matrix_ref*>(this); }
    matrix_ref ref() const { return matrix_ref(_base); }

private:
    base_ref_t _base;
};



template<typename data_t>
class matrix_t : public matrix_ref_t<data_t>
{
public:
    typedef vector_ptr_t<data_t> vector_ptr;
    typedef vector_ref_t<data_t> vector_ref;
    typedef matrix_ptr_t<data_t> matrix_ptr;
    typedef matrix_ref_t<data_t> matrix_ref;
    using matrix_ref::iterator;
    using matrix_ref::const_iterator;
    using matrix_ref::base;
    using matrix_ref::rows;
    using matrix_ref::columns;
    using matrix_ref::scratchcolumns;
    using matrix_ref::stride;
    using matrix_ref::begin;
    using matrix_ref::cbegin;
    using matrix_ref::end;
    using matrix_ref::cend;
private:
    using matrix_ref::reset;
    using matrix_ref::reset_submatrix;
public:
    static const size_t bitalignment = 512;
    static const size_t bytealignment = bitalignment / 8;

    ~matrix_t() { _free(); }
    /* constructors */
    matrix_t(size_t rows = 0, size_t columns = 0) { _realloc(rows, columns, true); }
    matrix_t(const matrix_t& m) : matrix_t(m.rows(), m.columns()) { *this = m; }
    matrix_t(matrix_t&& m) : matrix_t(0, 0) { swap(m); }

    matrix_t& operator=(const matrix_t& m) { _realloc(m.rows(), m.columns()); detail::matrix_copy(base(), m.base()); return *this; }
    matrix_t& operator=(matrix_t&& m) { swap(m); }

    matrix_t(const matrix_ref& m) : matrix_t(m.rows(), m.columns()) { *this = m; }
    matrix_t& operator=(const matrix_ref& m) { _realloc(m.rows(), m.columns()); detail::matrix_copy(base(), m.base()); return *this; }

    matrix_t& operator^=(const matrix_ref& m) { detail::matrix_xor(base(), m.base()); return *this; }
    matrix_t& operator|=(const matrix_ref& m) { detail::matrix_or(base(), m.base()); return *this; }
    matrix_t& operator&=(const matrix_ref& m) { detail::matrix_and(base(), m.base()); return *this; }

    // resize as needed for write-only operations (e.g. dst = f(src), dst = f(src1,src2))
    matrix_t& transpose(const matrix_ref& src) { _realloc(src.columns(),src.rows()); detail::matrix_transpose(base(), src.base()); return *this; }
    
    matrix_t& op_not(const matrix_ref& m2) { _realloc(m2.rows(), m2.columns()); detail::matrix_copynot(base()); return *this; }
    
    matrix_t& op_and  (const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_and  (base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_xor  (const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_xor  (base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_or   (const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_or   (base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_nand (const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_nand (base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_nxor (const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_nxor (base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_nor  (const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_nor  (base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_andin(const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_andin(base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_andni(const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_andni(base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_orin (const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_orin (base(), m1.base(), m2.base()); return *this; }
    matrix_t& op_orni (const matrix_ref& m1, const matrix_ref& m2) { _realloc(m1.rows(), m1.columns()); detail::matrix_orni (base(), m1.base(), m2.base()); return *this; }

    // comparison operators
    bool operator==(const matrix_ref& m) const { return matrix_compare(base(), m.base()); }
    bool operator!=(const matrix_ref& m) const { return !matrix_compare(base(), m.base()); }

    // indexing
    vector_ref operator[](size_t r) { return vector_ref(base().subvector(r, 0, columns(), scratchcolumns())); }
    const vector_ref operator[](size_t r) const { return vector_ref(base().subvector(r, 0, columns(), scratchcolumns())); }
    bool operator()(size_t r, size_t c) const { return base()(r, c); }

private:
    void _realloc(size_t _rows, size_t _columns, bool zero = false)
    {
        if (rows() == _rows && columns() == _columns)
            return;
        size_t totalcol = ((_columns + bitalignment - 1) / bitalignment) * bitalignment;
        size_t totalbytes = _rows * totalcol / 8;
        size_t newallocbytes = totalbytes + bytealignment;
        if (newallocbytes > _allocbytes)
        {
            if (_allocptr != nullptr)
               free(_allocptr);
            _allocptr = (data_t*)malloc(newallocbytes);
            _allocbytes = newallocbytes;
        }
        if (zero)
            memset(_allocptr, 0, _allocbytes);
        uintptr_t alignedptr = ((uintptr_t(_allocptr) + bytealignment - 1) / bytealignment) * bytealignment;
        matrix_ref::reset((data_t*)alignedptr, _rows, _columns, totalcol - _columns, totalcol / sizeof(data_t));
    }
    void _free()
    {
        if (_allocptr != nullptr)
            free(_allocptr);
    }

    data_t* _allocptr;
    size_t _allocbytes;
};



template<typename data_t>
class vector_t : public vector_ref_t<data_t>
{
public:
    typedef vector_ptr_t<data_t> vector_ptr;
    typedef vector_ref_t<data_t> vector_ref;
    typedef matrix_ptr_t<data_t> matrix_ptr;
    typedef matrix_ref_t<data_t> matrix_ref;
    using vector_ref::base;
    using vector_ref::rows;
    using vector_ref::columns;
    using vector_ref::scratchcolumns;
    using vector_ref::stride;
private:
    using vector_ref::reset;
    using vector_ref::reset_subvector;
public:
    static const size_t bitalignment = matrix_t<data_t>::bitalignment;
    static const size_t bytealignment = bitalignment / 8;


    ~vector_t() { _free(); }
    /* constructors */
    vector_t(size_t columns = 0) { _realloc(1, columns, true); }
    vector_t(const vector_t& m) : vector_t(m.columns()) { *this = m; }
    vector_t(vector_t&& m) : vector_t(0) { swap(m); }

    vector_t& operator=(const vector_t& m) { _realloc(1, m.columns()); detail::vector_copy(base(), m.base()); return *this; }
    vector_t& operator=(vector_t&& m) { swap(m); }

    vector_t(const vector_ref& m) : vector_t(m.rows(), m.columns()) { *this = m; }
    vector_t& operator=(const vector_ref& m) { _realloc(1, m.columns()); detail::vector_copy(base(), m.base()); return *this; }

    vector_t& operator^=(const vector_ref& m) { detail::vector_xor(base(), m.base()); return *this; }
    vector_t& operator|=(const vector_ref& m) { detail::vector_or(base(), m.base()); return *this; }
    vector_t& operator&=(const vector_ref& m) { detail::vector_and(base(), m.base()); return *this; }

    vector_t& op_not(const matrix_ref& m2) { _realloc(m2.rows(), m2.columns()); detail::matrix_copynot(base()); return *this; }
    
    vector_t& op_and  (const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_and  (base(), m1.base(), m2.base()); return *this; }
    vector_t& op_xor  (const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_xor  (base(), m1.base(), m2.base()); return *this; }
    vector_t& op_or   (const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_or   (base(), m1.base(), m2.base()); return *this; }
    vector_t& op_nand (const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_nand (base(), m1.base(), m2.base()); return *this; }
    vector_t& op_nxor (const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_nxor (base(), m1.base(), m2.base()); return *this; }
    vector_t& op_nor  (const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_nor  (base(), m1.base(), m2.base()); return *this; }
    vector_t& op_andin(const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_andin(base(), m1.base(), m2.base()); return *this; }
    vector_t& op_andni(const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_andni(base(), m1.base(), m2.base()); return *this; }
    vector_t& op_orin (const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_orin (base(), m1.base(), m2.base()); return *this; }
    vector_t& op_orni (const vector_ref& m1, const vector_ref& m2) { _realloc(1, m1.columns()); detail::vector_orni (base(), m1.base(), m2.base()); return *this; }

    // comparison operators
    bool operator==(const vector_ref& m) const { return vector_compare(base(), m.base()); }
    bool operator!=(const vector_ref& m) const { return !vector_compare(base(), m.base()); }

    // indexing
    bool operator()(size_t c) const { return base()(0, c); }

private:
    void _realloc(size_t _rows, size_t _columns, bool zero = false)
    {
        if (rows() == _rows && columns() == _columns)
            return;
        size_t totalcol = ((_columns + bitalignment - 1) / bitalignment) * bitalignment;
        size_t totalbytes = _rows * totalcol / 8;
        size_t newallocbytes = totalbytes + bytealignment;
        if (newallocbytes > _allocbytes)
        {
            if (_allocptr != nullptr)
               free(_allocptr);
            _allocptr = (data_t*)malloc(newallocbytes);
            _allocbytes = newallocbytes;
        }
        if (zero)
            memset(_allocptr, 0, _allocbytes);
        uintptr_t alignedptr = ((uintptr_t(_allocptr) + bytealignment - 1) / bytealignment) * bytealignment;
        vector_ref::reset((data_t*)alignedptr, _rows, _columns, totalcol - _columns, totalcol / sizeof(data_t));
    }
    void _free()
    {
        if (_allocptr != nullptr)
            free(_allocptr);
    }

    data_t* _allocptr;
    size_t _allocbytes;
};

MCCL_END_NAMESPACE

#endif
