#ifndef MCCL_CORE_MATRIX_HPP
#define MCCL_CORE_MATRIX_HPP

#include <mccl/config/config.hpp>
#include <mccl/core/matrix_base.hpp>
#include <mccl/core/matrix_ops.hpp>
#include <mccl/core/vector.hpp>

#include <array>
#include <vector>
#include <iostream>
#include <functional>

MCCL_BEGIN_NAMESPACE

/* ONE BASE MATRIX TYPE TO RULE ALL MATRIX VIEWS & MATRIX OWNER */
template<typename bt, bool cv, bool al> class base_matrix_t;


/* MAIN MATRIX TYPE DEFINITIONS */
// templated on block_tag
template<typename _block_tag = default_block_tag>
using  mat_view_t = base_matrix_t<_block_tag, false, false>;
template<typename _block_tag = default_block_tag>
using cmat_view_t = base_matrix_t<_block_tag, true , false>;

template<typename _block_tag = block_tag<256, false> >
using mat_t = base_matrix_t<_block_tag, false, true>;

// non-templated with chosen default block tags
typedef  mat_view_t   <default_block_tag>  mat_view;
typedef cmat_view_t   <default_block_tag> cmat_view;

typedef  mat_t        <block_tag<256,false>>  mat;




/* IMPLEMENTATION HELPERS */
namespace detail
{

// the underlying base matrix pointer type to use
template<bool _const_view> struct matrix_pointer_t;
template<> struct matrix_pointer_t<false> { typedef   m_ptr type; };
template<> struct matrix_pointer_t<true > { typedef  cm_ptr type; };

// different behaviour in default copy/move constructor & assignment cannot be captured in base_matrix_t
// so we capture this specific behaviour only in two variant classes core_matrix_t

// this is the core_matrix_t for views:
// - contains a matrix pointer: m_ptr, cm_ptr
// - copy/move constructor copies the matrix pointer
// - copy/move assignment have been deleted
template<typename _block_tag, bool _const_view, bool _allocate>
struct core_matrix_t
{
    core_matrix_t() {}
    
    core_matrix_t(const core_matrix_t&  ) = default;
    core_matrix_t(      core_matrix_t&& ) = default;
    
    core_matrix_t& operator=(const core_matrix_t& ) = delete;
    core_matrix_t& operator=(      core_matrix_t&&) = delete;

    static _block_tag tag() { return _block_tag(); }

    typename matrix_pointer_t<_const_view>::type _ptr;

protected:
    void _swap(core_matrix_t& m)
    {
        std::swap(_ptr, m._ptr);
    }

    // these are placeholder functions, but should never be called
    template<typename T> void _assign(T& ) { throw; }
};

// this is the core_matrix_t for allocating vec_t:
// - contains a matrix pointer: v_ptr
// - contains a matrix<uint64_t> to allocate memory
// - copy constructor & assignment: allocates new memory and copies the matrix contents
// - move constructor & assignment: swaps the internal matrix pointer and matrix<uint64_t>
template<typename _block_tag, bool _const_view>
struct core_matrix_t<_block_tag, _const_view, true>
{
    core_matrix_t() {}
    
    core_matrix_t(const core_matrix_t&  m) { _assign(m); }
    core_matrix_t(      core_matrix_t&& m) { _swap(m); }
    
    core_matrix_t& operator=(const core_matrix_t&  m) { _assign(m); return *this; }
    core_matrix_t& operator=(      core_matrix_t&& m) { _swap(m); return *this; }

    static _block_tag tag() { return _block_tag(); }

    static const size_t bit_alignment = 512;
    static const size_t byte_alignment = bit_alignment/8;
    static const size_t word_alignment = bit_alignment/64;

    typename matrix_pointer_t<_const_view>::type _ptr;

protected:
    std::vector<uint64_t> _mem;

    void _swap(core_matrix_t& m)
    {
        std::swap(_ptr, m._ptr);
        _mem.swap(m._mem);
    }

    void _alloc(size_t rows, size_t columns, bool value = false)
    {
        if (rows == 0 || columns == 0)
        {
            _mem.clear();
            _ptr.ptr = nullptr;
            _ptr.rows = _ptr.columns = _ptr.stride = 0;
            return;
        }
        
        // compute number of words to allocate, includes additional words to 'round-up' to desired alignment
        size_t rowwords = (columns + 63) / 64;
        size_t stride = (rowwords + word_alignment-1) & ~uint64_t(word_alignment-1);
        size_t totalwords = rows * stride + word_alignment;
        
        // reallocate memory
        if (totalwords > _mem.size())
            _mem.resize(totalwords, value ? ~uint64_t(0) : uint64_t(0));

        // set ptr that is 'round-up' to desired alignment
        _ptr.ptr = reinterpret_cast<uint64_t*>( (uintptr_t(&_mem[0]) + byte_alignment-1) & ~uintptr_t(byte_alignment-1) );
        _ptr.stride = stride;
        _ptr.rows = rows;
        _ptr.columns = columns;
    }
    
    template<typename bt, bool cv, bool al>
    void _assign(const core_matrix_t<bt,cv,al>& m)
    {
        _alloc(m._ptr.rows, m._ptr.columns);
        m_copy(_ptr, m._ptr, tag(), m.tag());
    }

    // don't allow weird instantiations
    static_assert( _const_view == false, "core_matrix_t: cannot have both _allocate and _const_view be true");
};


} // namespace detail




// meta-programming construct to convert 'm.m_and(m1,m2)' to 'm = m_and(m1,m2)';
// m_and(m1,m2) returns a matrix_result<R> such that 'r' (of type R) contains the pointers to m1 & m2 
// and the expression 'r(m)' calls the respective function 'm.vand(m1,m2)'
// note: to allow matrix to automatically resize to the correct result dimensions
//   r should have a member:
//     'template<matrix_t> resize_me(matrix_t&)'
template<typename m_ptr_op_result>
struct matrix_result
{
    m_ptr_op_result r;
    matrix_result(): r() {}
    matrix_result(const matrix_result&) = default;
    matrix_result(matrix_result&&) = default;
    matrix_result& operator=(const matrix_result&) = default;
    matrix_result& operator=(matrix_result&&) = default;
    template<typename... Args>
    matrix_result(Args&&... args): r(std::forward<Args>(args)...) {}
};




/*
    the base_matrix_t class from which mat_t and all matrix views (mat_view_t, cmat_view_t) are instantiated
*/
// _block_tag    : controls block-size during matrix operations and whether to use a bitmask for the last block
// _const_view   : true results in matrix view to const data
// _allocate     : true results in matrix owner that handles memory allocates
template<typename _block_tag, bool _const_view = true, bool _allocate = false>
class base_matrix_t
    final : public detail::core_matrix_t<_block_tag,_const_view,_allocate>
{
public:
    /* TYPEDEFS */
    typedef base_matrix_t<_block_tag, _const_view, _allocate> this_type;
    // matrix pointer type to (const) data
    typedef typename detail::matrix_pointer_t<_const_view>::type pointer_t;
    // matrix pointer type to const data
    typedef typename detail::matrix_pointer_t<true>::type        const_pointer_t;
    // matrix pointer type to use when matrix owner object is const
    typedef typename std::conditional<_allocate, const_pointer_t, pointer_t>::type pointer_t_const;

    typedef _block_tag this_block_tag;

    typedef typename std::conditional<_const_view, const uint64_t*, uint64_t*>::type word_ptr_t;
    typedef typename std::conditional<_const_view, const uint64_block_t<_block_tag::bits>*, uint64_block_t<_block_tag::bits>*>::type block_ptr_t;
    typedef const uint64_block_t<_block_tag::bits>* const_block_ptr_t;
    typedef const uint64_t* const_word_ptr_t;
    typedef typename std::conditional<_allocate, const_word_ptr_t, word_ptr_t>::type word_ptr_t_const;
    typedef typename std::conditional<_allocate, const_block_ptr_t, block_ptr_t>::type block_ptr_t_const;

    /* STATIC MEMBERS AND MEMBER FUNCTIONS */
    static const bool is_const = _const_view;
    static const bool is_nonconst = !is_const;
    static const bool is_owner = _allocate;
    static const bool is_view = !is_owner;
    static const bool is_matrix = true;
    static_assert( is_owner == false || is_const    == false, "cannot have both is_owner == true && is_const == true");

    static const bool maskedlastblock = this_block_tag::maskedlastblock;
    static const size_t block_bits = this_block_tag::bits;

    static this_block_tag tag() { return this_block_tag(); }

    // convenience macro to conditionally enable external functions on matrices
#define MCCL_ENABLE_IF_MATRIX(MT)          typename std::enable_if< MT ::is_matrix,bool>::type* = nullptr
#define MCCL_ENABLE_IF_NONCONST_MATRIX(MT) typename std::enable_if< MT ::is_matrix && ! MT ::is_const && (!std::is_const< MT >::value || ! MT ::is_owner),bool>::type* = nullptr

    // helper types to conditionally enable member functions
    template<typename matrix_t = this_type> struct _needs_allocating_matrix_type    : public std::enable_if< std::is_same<matrix_t,this_type>::value && is_owner   , bool > {};
    template<typename matrix_t = this_type> struct _needs_matrix_view_type          : public std::enable_if< std::is_same<matrix_t,this_type>::value && is_view    , bool > {};
    template<typename matrix_t = this_type> struct _needs_nonconst_matrix_type      : public std::enable_if< std::is_same<matrix_t,this_type>::value && is_nonconst, bool > {};
    template<typename matrix_t = this_type> struct _needs_nonconst_matrix_view_type : public std::enable_if< std::is_same<matrix_t,this_type>::value && is_nonconst && is_view, bool > {};
    // opposite helper types to generate deleted member functions with the same spec
    template<typename matrix_t = this_type> struct _needs2_allocating_matrix_type    : public std::enable_if< std::is_same<matrix_t,this_type>::value && !is_owner   , bool > {};
    template<typename matrix_t = this_type> struct _needs2_matrix_view_type          : public std::enable_if< std::is_same<matrix_t,this_type>::value && !is_view    , bool > {};
    template<typename matrix_t = this_type> struct _needs2_nonconst_matrix_type      : public std::enable_if< std::is_same<matrix_t,this_type>::value && !is_nonconst, bool > {};
    template<typename matrix_t = this_type> struct _needs2_nonconst_matrix_view_type : public std::enable_if< std::is_same<matrix_t,this_type>::value && !(is_nonconst && is_view), bool > {};

    // convenience macro to conditionally enable member functions
#define MCCL_BASEMAT_ENABLE_IF(s,err)       typename err = this_type, typename std::enable_if< std::is_same<err,this_type>::value && ( s )      , bool >::type* = nullptr
#define MCCL_BASEMAT_ENABLE_IF_OWNER        typename func_requires_allocating_matrix_type    = this_type, typename _needs_allocating_matrix_type   <func_requires_allocating_matrix_type   >::type* = nullptr
#define MCCL_BASEMAT_ENABLE_IF_VIEW         typename func_requires_matrix_view_type          = this_type, typename _needs_matrix_view_type         <func_requires_matrix_view_type         >::type* = nullptr
#define MCCL_BASEMAT_ENABLE_IF_NONCONST     typename func_requires_nonconst_matrix_type      = this_type, typename _needs_nonconst_matrix_type     <func_requires_nonconst_matrix_type     >::type* = nullptr
#define MCCL_BASEMAT_ENABLE_IF_NONCONSTVIEW typename func_requires_nonconst_matrix_view_type = this_type, typename _needs_nonconst_matrix_view_type<func_requires_nonconst_matrix_view_type>::type* = nullptr
    // convenience macros to conditionally generate deleted member functions
#define MCCL_BASEMAT_DELETE_IF(s,err)           typename err = this_type, typename std::enable_if< std::is_same<err,this_type>::value && ( s )      , bool >::type* = nullptr
#define MCCL_BASEMAT_DELETE_IF_NOT_OWNER        typename func_requires_allocating_matrix_type    = this_type, typename _needs2_allocating_matrix_type   <func_requires_allocating_matrix_type   >::type* = nullptr
#define MCCL_BASEMAT_DELETE_IF_NOT_VIEW         typename func_requires_matrix_view_type          = this_type, typename _needs2_matrix_view_type         <func_requires_matrix_view_type         >::type* = nullptr
#define MCCL_BASEMAT_DELETE_IF_NOT_NONCONST     typename func_requires_nonconst_matrix_type      = this_type, typename _needs2_nonconst_matrix_type     <func_requires_nonconst_matrix_type     >::type* = nullptr
#define MCCL_BASEMAT_DELETE_IF_NOT_NONCONSTVIEW typename func_requires_nonconst_matrix_view_type = this_type, typename _needs2_nonconst_matrix_view_type<func_requires_nonconst_matrix_view_type>::type* = nullptr

// convenience macro to assert that another block tag is compatible
#define MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked) static_assert( bits <= block_bits && (masked == true || maskedlastblock == false), "base_matrix_t: cannot cast to specified block_tag" );

private:
    // _copy_ptr: only for views
    template<typename M>
    void _copy_ptr(M& m)
    {
        static_assert( M::block_bits >= block_bits && (M::maskedlastblock == false || maskedlastblock == true), "base_matrix_t(base_matrix_t): input type has incompatible block_tag" );
        if (is_owner) throw;
        ptr() = m.ptr();
    }
    template<typename M>
    void _assign(M& m)
    {
        if (is_view) throw;
        this->_assign(m);
    }
    // if owner then call resize_me on result r, otherwise ignore
    template<typename R>
    void _resize_me_if_owner(R& r, std::true_type)
    {
        r.resize_me(*this);
    }
    template<typename R>
    void _resize_me_if_owner(R&, std::false_type) const
    {
    }
    template<typename R>
    void _assign_result(R& r)
    {
        _resize_me_if_owner(r, std::integral_constant<bool,is_owner>());
        r(ptr(), tag());        
    }
public:

    /* CONSTRUCTORS & ASSIGNMENT */

    base_matrix_t() {}

    // default copy/move constructor & assignment: behaviour is controlled by core_matrix_t
    base_matrix_t(const base_matrix_t&  ) = default;
    base_matrix_t(      base_matrix_t&& ) = default;

    // if is_view == true then these default copy/move assignment are deleted because of core_matrix_t
    base_matrix_t& operator= (const base_matrix_t&  ) = default;
    base_matrix_t& operator= (      base_matrix_t&& ) = default;

    // constructor taking a matrix pointer
    template<MCCL_BASEMAT_DELETE_IF_NOT_VIEW> explicit base_matrix_t(const pointer_t& p) = delete;
    template<MCCL_BASEMAT_ENABLE_IF_VIEW>     explicit base_matrix_t(const pointer_t& p)
    {
        ptr() = p;
    }

    // constructor that creates a new matrix with a specified number of columns
    template<MCCL_BASEMAT_DELETE_IF_NOT_OWNER> explicit base_matrix_t(size_t _rows, size_t _columns, bool value = false) = delete;
    template<MCCL_BASEMAT_ENABLE_IF_OWNER>     explicit base_matrix_t(size_t _rows, size_t _columns, bool value = false) 
    {
        this->_alloc(_rows, _columns, value);
    }

    // construct from another base_matrix_t: copy view or assign content depending on is_view
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_DELETE_IF(is_view && !(is_const || cv==false), nonconst_view_requires_nonconst_matrix)>
    explicit base_matrix_t(      base_matrix_t<bt,cv,al>& m) = delete;
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_DELETE_IF(is_view && !(is_const || (cv==false&&al==false)), nonconst_view_requires_nonconst_matrix)>
    explicit base_matrix_t(const base_matrix_t<bt,cv,al>& m) = delete;

    // for views we have two constructors: with non-const ref (to be initialized from owner), and const ref (other non-const views)
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_ENABLE_IF(is_view && (is_const || cv==false), nonconst_view_requires_nonconst_matrix)>
    explicit base_matrix_t(base_matrix_t<bt,cv,al>& m)
    {
        _copy_ptr(m);
    }
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_ENABLE_IF(is_view && (is_const || (cv==false&&al==false)), nonconst_view_requires_nonconst_matrix)>
    explicit base_matrix_t(const base_matrix_t<bt,cv,al>& m)
    {
        _copy_ptr(m);
    }
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_ENABLE_IF_OWNER>
    explicit base_matrix_t(const base_matrix_t<bt,cv,al>& m)
    {
        _assign(m);
    }

    // construct from a matrix_result
    template<typename F, MCCL_BASEMAT_DELETE_IF_NOT_OWNER> base_matrix_t(matrix_result<F>&& mr) = delete;
    template<typename F, MCCL_BASEMAT_ENABLE_IF_OWNER>     base_matrix_t(matrix_result<F>&& mr)
    {
        _assign_result(mr.r);
    }


    // assign from another base_matrix_t
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_DELETE_IF_NOT_OWNER> base_matrix_t& operator= (const base_matrix_t<bt,cv,al>& m) = delete;
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_ENABLE_IF_OWNER>     base_matrix_t& operator= (const base_matrix_t<bt,cv,al>& m)
    {
        this->_assign(m);
        return *this;
    }

    // assign from a matrix_result
    template<typename F, MCCL_BASEMAT_DELETE_IF_NOT_NONCONST> base_matrix_t& operator= (matrix_result<F>&& mr) = delete;
    template<typename F, MCCL_BASEMAT_ENABLE_IF_NONCONST>     base_matrix_t& operator= (matrix_result<F>&& mr)
    {
        _assign_result(mr.r);
        return *this;
    }
    template<typename F, MCCL_BASEMAT_DELETE_IF_NOT_NONCONSTVIEW> const base_matrix_t& operator= (matrix_result<F>&& mr) const = delete;
    template<typename F, MCCL_BASEMAT_ENABLE_IF_NONCONSTVIEW>     const base_matrix_t& operator= (matrix_result<F>&& mr) const
    {
        _assign_result(mr.r);
        return *this;
    }
    

    /* OWNER specific member functions */

    // resize matrix to specific dimensions
    // WARNING: if cols > columns() then this may reallocate memory and any pointers to this matrix become invalid!
    template<MCCL_BASEMAT_DELETE_IF_NOT_OWNER> this_type& resize(size_t _rows, size_t _columns, bool value = false) = delete;
    template<MCCL_BASEMAT_ENABLE_IF_OWNER>     this_type& resize(size_t _rows, size_t _columns, bool value = false)
    {
        // when shrinking we only update dimensions
        if (_columns <= columns() && _rows <= rows())
        {
            if (_rows == 0 || _columns == 0)
            {
                ptr().ptr = nullptr;
                ptr().columns = ptr().rows = 0;
            } else
            {
                ptr().columns = _columns;
                ptr().rows = _rows;
            }
            return *this;
        }
        // otherwise we create a new properly dimensioned matrix
        base_matrix_t tmp(_rows, _columns, value);
        // copy old content
        if (columns() > 0)
            tmp.     submatrix(0, std::min<size_t>(_rows,rows()), std::min<size_t>(_columns,columns()))
            .m_copy( submatrix(0, std::min<size_t>(_rows,rows()), std::min<size_t>(_columns,columns())) );
        // and swap
        tmp.swap(*this);
        return *this;
    }


    /* VIEW specific member functions */

    // reset view to another view given by compatible pointer or matrix view
    template<MCCL_BASEMAT_DELETE_IF_NOT_VIEW> void reset(const pointer_t& p) = delete;
    template<MCCL_BASEMAT_ENABLE_IF_VIEW>     void reset(const pointer_t& p)
    {
        ptr() = p;
    }
    template<MCCL_BASEMAT_DELETE_IF_NOT_VIEW> void reset(const this_type& m) = delete;
    template<MCCL_BASEMAT_ENABLE_IF_VIEW>     void reset(const this_type& m)
    {
        ptr() = m.ptr();
    }


    /* COMMON basic member functions */

    size_t rows() const { return ptr().rows; }
    size_t columns() const { return ptr().columns; }
    size_t hw() const { return detail::m_hw(ptr()); }
    
    bool operator()(size_t r, size_t c) const { return detail::m_getbit(ptr(),r,c); }

    template<typename bt, bool cv, bool al>
    bool is_equal(const base_matrix_t<bt,cv,al>& m2) const
    {
        return detail::m_isequal(ptr(),m2.ptr(),tag(),m2.tag());
    }

    // swap matrixs by swapping member variables
    this_type& swap(this_type& m)
    {
        this->_swap(m);
        return *this;
    }


    /* SUBVECTOR functions */
    
    // subvector always returns a view iterator. for a const owner this view is forced to be const
    template<typename block_tag>
    using this_vector_view = base_vector_t<block_tag, is_const, false, false>;
    template<typename block_tag>
    using this_vector_view_const = base_vector_t<block_tag, is_const||is_owner, false, false>;

    typedef block_tag<block_bits,true> this_block_tag_masked;
    
    // whole subvector maintains block_tag
    this_vector_view      <this_block_tag>        subvector(size_t row)                    { return this_vector_view      <this_block_tag>( ptr().subvector(row,0,columns()) ); }
    this_vector_view_const<this_block_tag>        subvector(size_t row)              const { return this_vector_view_const<this_block_tag>( ptr().subvector(row,0,columns()) ); }
    // subvector starting from 0 maintains block_tag::bits but has maskedlastblock = true;
    this_vector_view      <this_block_tag_masked> subvector(size_t row, size_t cols)       { return this_vector_view      <this_block_tag_masked>( ptr().subvector(row,0,cols) ); }
    this_vector_view_const<this_block_tag_masked> subvector(size_t row, size_t cols) const { return this_vector_view_const<this_block_tag_masked>( ptr().subvector(row,0,cols) ); }
    // otherwise returned view has default_block_tag
    this_vector_view      <default_block_tag>     subvector(size_t row, size_t coloff, size_t cols)       { return this_vector_view      <default_block_tag>( ptr().subvector(row,coloff,cols) ); }
    this_vector_view_const<default_block_tag>     subvector(size_t row, size_t coloff, size_t cols) const { return this_vector_view_const<default_block_tag>( ptr().subvector(row,coloff,cols) ); }

    // unless the user overrides with a compatible tag
    template<size_t bits, bool masked>
    this_vector_view<block_tag<bits,masked>> subvector(size_t row, block_tag<bits,masked>) 
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_vector_view<block_tag<bits,masked>>( ptr().subvector(row,0,columns()) ); 
    }
    template<size_t bits, bool masked>
    this_vector_view_const<block_tag<bits,masked>> subvector(size_t row, block_tag<bits,masked>) const
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_vector_view_const <block_tag<bits,masked>>( ptr().subvector(row,0,columns()) );
    }
    template<size_t bits, bool masked>
    this_vector_view<block_tag<bits,masked>> subvector(size_t row, size_t cols, block_tag<bits,masked>) 
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_vector_view<block_tag<bits,masked>>( ptr().subvector(row,0,cols) ); 
    }
    template<size_t bits, bool masked>
    this_vector_view_const<block_tag<bits,masked>> subvector(size_t row, size_t cols, block_tag<bits,masked>) const
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_vector_view_const <block_tag<bits,masked>>( ptr().subvector(row,0,cols) );
    }
    template<size_t bits, bool masked>
    this_vector_view<block_tag<bits,masked>> subvector(size_t row, size_t coloff, size_t cols, block_tag<bits,masked>)
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_vector_view<block_tag<bits,masked>>( ptr().subvector(row,coloff,cols) );
    }
    template<size_t bits, bool masked>
    this_vector_view_const<block_tag<bits,masked>> subvector(size_t row, size_t coloff, size_t cols, block_tag<bits,masked>) const
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_vector_view_const<block_tag<bits,masked>>( ptr().subvector(row,coloff,cols) );
    }


    /* ARRAY OPERATOR & ITERATOR */

    this_vector_view      <this_block_tag> operator[](size_t r)       { return subvector(r); }
    this_vector_view_const<this_block_tag> operator[](size_t r) const { return subvector(r); }
    this_vector_view      <this_block_tag> operator()(size_t r)       { return subvector(r); }
    this_vector_view_const<this_block_tag> operator()(size_t r) const { return subvector(r); }

    // three iterator types:
    // - iterator is const if vector is const => convertible to iterator_const & const_iterator
    // - iterator_const is const if vector is const or owner => convertible to const_iterator
    // - const_iterator is always const
    template<typename block_tag>
    using this_vector_iterator = base_vector_t<block_tag, is_const, true, false>;
    template<typename block_tag>
    using this_vector_iterator_const = base_vector_t<block_tag, is_const||is_owner, true, false>;
    template<typename block_tag>
    using this_const_vector_iterator = base_vector_t<block_tag, true, true, false>;

    typedef this_vector_iterator      <this_block_tag>       iterator;
    typedef this_vector_iterator_const<this_block_tag>       iterator_const;
    typedef this_const_vector_iterator<this_block_tag>       const_iterator;
        
    iterator       begin()       { return iterator      ( ptr().subvectorit(0) ); }
    iterator_const begin() const { return iterator_const( ptr().subvectorit(0) ); }
    iterator       end()         { return iterator      ( ptr().subvectorit(rows()) ); }
    iterator_const end()   const { return iterator_const( ptr().subvectorit(rows()) ); }


    /* SUBMATRIX functions */
    
    // submatrix always returns a view matrix. for a const owner this view is forced to be const
    template<typename block_tag>
    using this_matrix_view = base_matrix_t<block_tag, is_const, false>;
    template<typename block_tag>
    using this_matrix_view_const = base_matrix_t<block_tag, is_const||is_owner, false>;

    // submatrix with whole rows maintains block_tag
    this_matrix_view      <this_block_tag>        submatrix(size_t rowoff, size_t rows)                    { return this_matrix_view      <this_block_tag>( ptr().submatrix(rowoff,rows,0,columns()) ); }
    this_matrix_view_const<this_block_tag>        submatrix(size_t rowoff, size_t rows)              const { return this_matrix_view_const<this_block_tag>( ptr().submatrix(rowoff,rows,0,columns()) ); }
    // submatrix with rows starting at column 0 maintains block_tag::bits, but maskedlastblock = true
    this_matrix_view      <this_block_tag_masked> submatrix(size_t rowoff, size_t rows, size_t cols)       { return this_matrix_view      <this_block_tag_masked>( ptr().submatrix(rowoff,rows,0,cols) ); }
    this_matrix_view_const<this_block_tag_masked> submatrix(size_t rowoff, size_t rows, size_t cols) const { return this_matrix_view_const<this_block_tag_masked>( ptr().submatrix(rowoff,rows,0,cols) ); }
    // otherwise returned view has default_block_tag
    this_matrix_view      <default_block_tag>     submatrix(size_t rowoff, size_t rows, size_t coloff, size_t cols)       { return this_matrix_view      <default_block_tag>( ptr().submatrix(rowoff,rows,coloff,cols) ); }
    this_matrix_view_const<default_block_tag>     submatrix(size_t rowoff, size_t rows, size_t coloff, size_t cols) const { return this_matrix_view_const<default_block_tag>( ptr().submatrix(rowoff,rows,coloff,cols) ); }

    // unless the user overrides with a compatible tag
    template<size_t bits, bool masked>
    this_matrix_view<block_tag<bits,masked>> submatrix(size_t rowoff, size_t rows, block_tag<bits,masked>) 
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_matrix_view<block_tag<bits,masked>>( ptr().submatrix(rowoff,rows,0,columns()) ); 
    }
    template<size_t bits, bool masked>
    this_matrix_view_const<block_tag<bits,masked>> submatrix(size_t rowoff, size_t rows, block_tag<bits,masked>) const
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_matrix_view_const <block_tag<bits,masked>>( ptr().submatrix(rowoff,rows,0,columns()) );
    }
    template<size_t bits, bool masked>
    this_matrix_view<block_tag<bits,masked>> submatrix(size_t rowoff, size_t rows, size_t cols, block_tag<bits,masked>) 
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_matrix_view<block_tag<bits,masked>>( ptr().submatrix(rowoff,rows,0,cols) ); 
    }
    template<size_t bits, bool masked>
    this_matrix_view_const<block_tag<bits,masked>> submatrix(size_t rowoff, size_t rows, size_t cols, block_tag<bits,masked>) const
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_matrix_view_const <block_tag<bits,masked>>( ptr().submatrix(rowoff,rows,0,cols) );
    }
    template<size_t bits, bool masked>
    this_matrix_view<block_tag<bits,masked>> submatrix(size_t rowoff, size_t rows, size_t coloff, size_t cols, block_tag<bits,masked>)
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_matrix_view<block_tag<bits,masked>>( ptr().submatrix(rowoff,rows,coloff,cols) );
    }
    template<size_t bits, bool masked>
    this_matrix_view_const<block_tag<bits,masked>> submatrix(size_t rowoff, size_t rows, size_t coloff, size_t cols, block_tag<bits,masked>) const
    {
        MCCL_BASEMAT_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_matrix_view_const<block_tag<bits,masked>>( ptr().submatrix(rowoff,rows,coloff,cols) );
    }


    /* ACTUAL MATRIX OPERATIONS */
    
    // modifing member functions
    // note that a const version is defined only for views, not for owners
#define MCCL_BASEMAT_MEMBER_FUNC(spec,expr) \
    template<MCCL_BASEMAT_DELETE_IF_NOT_NONCONST>           this_type& spec       = delete;               \
    template<MCCL_BASEMAT_DELETE_IF_NOT_NONCONSTVIEW> const this_type& spec const = delete;               \
    template<MCCL_BASEMAT_ENABLE_IF_NONCONST>               this_type& spec       { expr; return *this; } \
    template<MCCL_BASEMAT_ENABLE_IF_NONCONSTVIEW>     const this_type& spec const { expr; return *this; }
    
    MCCL_BASEMAT_MEMBER_FUNC( m_clear ()                 , detail::m_clear(ptr(), tag());    )
    MCCL_BASEMAT_MEMBER_FUNC( m_not   ()                 , detail::m_not  (ptr(), tag());    )
    MCCL_BASEMAT_MEMBER_FUNC( m_set   ()                 , detail::m_set  (ptr(), tag());    )
    MCCL_BASEMAT_MEMBER_FUNC( m_set   (bool b)           , detail::m_set  (ptr(), b, tag()); )
    
    MCCL_BASEMAT_MEMBER_FUNC( transpose(const cmat_view& src), detail::m_transpose(ptr(), src.ptr()); )
    MCCL_BASEMAT_MEMBER_FUNC( set_identity()                 , m_clear(); for (size_t i=0; i<rows()&&i<columns(); ++i) setbit(i,i); )
    
    MCCL_BASEMAT_MEMBER_FUNC( clearbit(size_t r, size_t c)         , detail::m_clearbit(ptr(), r, c);     )
    MCCL_BASEMAT_MEMBER_FUNC( flipbit (size_t r, size_t c)         , detail::m_flipbit (ptr(), r, c);     )
    MCCL_BASEMAT_MEMBER_FUNC( setbit  (size_t r, size_t c)         , detail::m_setbit  (ptr(), r, c);     )
    MCCL_BASEMAT_MEMBER_FUNC( setbit  (size_t r, size_t c, bool b) , detail::m_setbit  (ptr(), r, c, b);  )
    
    MCCL_BASEMAT_MEMBER_FUNC( setcolumns  (size_t c_off, size_t c_cnt, bool b) , detail::m_setcolumns  (ptr(), c_off, c_cnt, b); )
    MCCL_BASEMAT_MEMBER_FUNC( setcolumns  (size_t c_off, size_t c_cnt)         , detail::m_setcolumns  (ptr(), c_off, c_cnt);    )
    MCCL_BASEMAT_MEMBER_FUNC( clearcolumns(size_t c_off, size_t c_cnt)         , detail::m_clearcolumns(ptr(), c_off, c_cnt);    )
    MCCL_BASEMAT_MEMBER_FUNC( flipcolumns (size_t c_off, size_t c_cnt)         , detail::m_flipcolumns (ptr(), c_off, c_cnt);    )

    MCCL_BASEMAT_MEMBER_FUNC( swapcolumns (size_t c1   , size_t c2   )         , detail::m_swapcolumns (ptr(), c1, c2); )
    
    // 1-matrix member functions that translate to 2-matrix operations with current object the destination matrix
    // note that a const version is defined only for views, not for owners
#define MCCL_BASEMAT_1OP_MEMBER_FUNC(func) \
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_DELETE_IF_NOT_NONCONST>           this_type& func (const base_matrix_t<bt,cv,al>& m1)       = delete; \
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_DELETE_IF_NOT_NONCONSTVIEW> const this_type& func (const base_matrix_t<bt,cv,al>& m1) const = delete; \
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_ENABLE_IF_NONCONST>     \
    this_type& func (const base_matrix_t<bt,cv,al>& m1)                                \
    { detail:: func (ptr(), m1.ptr(), tag(), m1.tag()); return *this; }                   \
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_ENABLE_IF_NONCONSTVIEW> \
    const this_type& func (const base_matrix_t<bt,cv,al>& m1) const                    \
    { detail:: func (ptr(), m1.ptr(), tag(), m1.tag()); return *this; }

    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_copy)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_not)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_and)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_xor)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_or)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_nand)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_nxor)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_nor)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_andin)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_andni)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_orin)
    MCCL_BASEMAT_1OP_MEMBER_FUNC(m_orni)


    // 1-matrix member operators that translate to 2-matrix operations with current object the destination matrix
    // note that a const version is defined only for views, not for owners
#define MCCL_BASEMAT_1OP_MEMBER_FUNC_SYM(sym, func) \
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_DELETE_IF_NOT_NONCONST>           this_type& operator sym (const base_matrix_t<bt,cv,al>& m1)       = delete; \
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_DELETE_IF_NOT_NONCONSTVIEW> const this_type& operator sym (const base_matrix_t<bt,cv,al>& m1) const = delete; \
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_ENABLE_IF_NONCONST>     \
    this_type& operator sym (const base_matrix_t<bt,cv,al>& m1)                        \
    { detail:: func (ptr(), m1.ptr(), tag(), m1.tag()); return *this; }                   \
    template<typename bt, bool cv, bool al, MCCL_BASEMAT_ENABLE_IF_NONCONSTVIEW> \
    const this_type& operator sym (const base_matrix_t<bt,cv,al>& m1) const            \
    { detail:: func (ptr(), m1.ptr(), tag(), m1.tag()); return *this; }

    MCCL_BASEMAT_1OP_MEMBER_FUNC_SYM( &= , m_and )
    MCCL_BASEMAT_1OP_MEMBER_FUNC_SYM( |= , m_or )
    MCCL_BASEMAT_1OP_MEMBER_FUNC_SYM( ^= , m_xor )

    
    // 2-matrix member functions that translate to 3-matrix operations with current object the destination matrix
    // note that a const version is defined only for views, not for owners
#define MCCL_BASEMAT_2OP_MEMBER_FUNC(func) \
    template<typename bt1, bool cv1, bool al1, typename bt2, bool cv2, bool al2, MCCL_BASEMAT_DELETE_IF_NOT_NONCONST>           this_type& func (const base_matrix_t<bt1,cv1,al1>& m1, const base_matrix_t<bt2,cv2,al2>& m2)       = delete; \
    template<typename bt1, bool cv1, bool al1, typename bt2, bool cv2, bool al2, MCCL_BASEMAT_DELETE_IF_NOT_NONCONSTVIEW> const this_type& func (const base_matrix_t<bt1,cv1,al1>& m1, const base_matrix_t<bt2,cv2,al2>& m2) const = delete; \
    template<typename bt1, bool cv1, bool al1, typename bt2, bool cv2, bool al2, MCCL_BASEMAT_ENABLE_IF_NONCONST>     \
    this_type& func (const base_matrix_t<bt1,cv1,al1>& m1, const base_matrix_t<bt2,cv2,al2>& m2)                                  \
    { detail:: func (ptr(), m1.ptr(), m2.ptr(), tag(), m1.tag(), m2.tag()); return *this; }                                               \
    template<typename bt1, bool cv1, bool al1, typename bt2, bool cv2, bool al2, MCCL_BASEMAT_ENABLE_IF_NONCONSTVIEW> \
    const this_type& func (const base_matrix_t<bt1,cv1,al1>& m1, const base_matrix_t<bt2,cv2,al2>& m2) const                      \
    { detail:: func (ptr(), m1.ptr(), m2.ptr(), tag(), m1.tag(), m2.tag()); return *this; }

    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_and)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_xor)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_or)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_nand)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_nxor)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_nor)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_andin)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_andni)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_orin)
    MCCL_BASEMAT_2OP_MEMBER_FUNC(m_orni)


    /* RAW DATA FUNCTIONS */
    
          pointer_t      & ptr()       { return this->_ptr; }
    const pointer_t_const& ptr() const { return this->_ptr; }

    size_t row_words()  const { return (ptr().columns+63)/64; }
    size_t word_stride() const { return ptr().stride; }
    auto word_ptr(size_t r = 0)       -> decltype(ptr().ptr) { return ptr().ptr + r*ptr().stride; }
    auto word_ptr(size_t r = 0) const -> decltype(ptr().ptr) { return ptr().ptr + r*ptr().stride; }
    
    size_t row_blocks() const { return (ptr().columns+block_bits-1)/block_bits; }
    size_t block_stride() const { return ptr().stride / (block_bits/64); }
    auto block_ptr(size_t r = 0)       -> decltype(make_block_ptr(ptr().ptr, tag())) { return make_block_ptr(ptr().ptr + r*ptr().stride, tag()); }
    auto block_ptr(size_t r = 0) const -> decltype(make_block_ptr(ptr().ptr, tag())) { return make_block_ptr(ptr().ptr + r*ptr().stride, tag()); }

    template<size_t bits, bool maskedlastword>
    size_t row_blocks(block_tag<bits,maskedlastword>) const 
    {
        return (ptr().columns+bits-1)/bits;
    }
    template<size_t bits, bool maskedlastword>
    size_t block_stride(block_tag<bits,maskedlastword>) const
    {
        return ptr().stride / (bits/64);
    }
    template<size_t bits, bool maskedlastword>
    auto block_ptr(size_t r, block_tag<bits,maskedlastword>)       -> decltype(make_block_ptr(ptr().ptr, block_tag<bits,maskedlastword>()))
    {
        return make_block_ptr(ptr().ptr + r*ptr().stride, block_tag<bits,maskedlastword>());
    }
    template<size_t bits, bool maskedlastword>
    auto block_ptr(size_t r, block_tag<bits,maskedlastword>) const -> decltype(make_block_ptr(ptr().ptr, block_tag<bits,maskedlastword>()))
    {
        return make_block_ptr(ptr().ptr + r*ptr().stride, block_tag<bits,maskedlastword>());
    }


    /* CONVERSION OPERATOR */

    // conversion from 'base_matrix_t<..>&' to another 'base_matrix_t<..>&' is allowed IF:
    // - bt is compatible: bt::bits <= block_bits AND (bt::maskedlastblock==true OR maskedlastblock==false)
    // - cv==true   if  is_const == true
    template<typename bt, bool cv, MCCL_BASEMAT_ENABLE_IF( (cv|is_const) == cv && bt::bits <= block_bits && (bt::maskedlastblock == true || maskedlastblock == false), destination_type_is_incompatible )>
    operator base_matrix_t<bt,cv,false>& ()
    {
        return * reinterpret_cast< base_matrix_t<bt,cv,false>* >(this);
    }
    
    // conversion from '(const) base_matrix_t<..>&' to another 'const base_matrix_t<..>&' is allowed IF:
    // - bt is compatible: bt::bits <= block_bits AND (bt::maskedlastblock==true OR maskedlastblock==false)
    // - cv==true   if  is_const == true OR is_owner == true
    template<typename bt, bool cv, MCCL_BASEMAT_ENABLE_IF( (cv|is_const) == cv && (cv|is_owner) == cv && bt::bits <= block_bits && (bt::maskedlastblock == true || maskedlastblock == false), destination_type_is_incompatible )>
    operator const base_matrix_t<bt,cv,false>& ()
    {
        return * reinterpret_cast< const base_matrix_t<bt,cv,false>* >(this);
    }
    template<typename bt, bool cv, MCCL_BASEMAT_ENABLE_IF( (cv|is_const) == cv && (cv|is_owner) == cv && bt::bits <= block_bits && (bt::maskedlastblock == true || maskedlastblock == false), destination_type_is_incompatible )>
    operator const base_matrix_t<bt,cv,false>& () const
    {
        return * reinterpret_cast< const base_matrix_t<bt,cv,false>* >(this);
    }


    /* OVERRULE BLOCKTAG, note: returns reference */
    
    template<size_t bits, bool masked>
    base_matrix_t<block_tag<bits,masked>,is_const,false>& as(block_tag<bits,masked>)
    {
        return * reinterpret_cast< base_matrix_t<block_tag<bits,masked>,is_const,false>* >(this);
    }
    template<size_t bits, bool masked>
    const base_matrix_t<block_tag<bits,masked>,is_const||is_owner,false>& as(block_tag<bits,masked>) const
    {
        return * reinterpret_cast< const base_matrix_t<block_tag<bits,masked>,is_const||is_owner,false>* >(this);
    }

};


/* COMPARISON */

template<typename bt1, bool cv1, bool al1, typename bt2, bool cv2, bool al2>
inline bool operator==(const base_matrix_t<bt1,cv1,al1>& m1, const base_matrix_t<bt2,cv2,al2>& m2)
{ return m1.is_equal(m2); }

template<typename bt1, bool cv1, bool al1, typename bt2, bool cv2, bool al2>
inline bool operator!=(const base_matrix_t<bt1,cv1,al1>& m1, const base_matrix_t<bt2,cv2,al2>& m2)
{ return ! m1.is_equal(m2); }


/* OUTPUT */

template<typename bt1, bool cv1, bool al1>
inline std::ostream& operator<<(std::ostream& o, const base_matrix_t<bt1,cv1,al1>& m) { detail::m_print(o, m.ptr()); return o; }


/* HAMMINGWEIGHT */

template<typename bt, bool cv, bool al>
inline size_t hammingweight(const base_matrix_t<bt,cv,al>& m) { return m.hw(); }



/* MATRIX OPERATIONS THAT RETURN A MATRIX_RESULT */

struct m_ptr_op2_result_m_transpose
{
    const cm_ptr* m2;
    m_ptr_op2_result_m_transpose(): m2(nullptr) {}
    m_ptr_op2_result_m_transpose(const cm_ptr& _m2) { m2 = &_m2; }
    m_ptr_op2_result_m_transpose(const m_ptr_op2_result_m_transpose &) = default;
    m_ptr_op2_result_m_transpose(      m_ptr_op2_result_m_transpose &&) = default;
    m_ptr_op2_result_m_transpose& operator=(const m_ptr_op2_result_m_transpose &) = default;
    m_ptr_op2_result_m_transpose& operator=(      m_ptr_op2_result_m_transpose &&) = default;
    template<size_t bits1, bool masked1> void operator()(const m_ptr& m1, block_tag<bits1,masked1>) { detail::m_transpose(m1,*m2); }
    template<typename matrix_t> void resize_me(matrix_t& m) { m.resize(m2->columns, m2->rows); }
};
template<typename bt, bool cv, bool al>
inline matrix_result<m_ptr_op2_result_m_transpose> m_transpose(const base_matrix_t<bt,cv,al>& m2)
{
    return matrix_result<m_ptr_op2_result_m_transpose>(m2.ptr());
}


#define MCCL_MATRIX_RESULT_FUNCTION_OP2(func) \
template<typename _block_tag2> \
struct m_ptr_op2_result_ ## func \
{ \
    typedef _block_tag2 block_tag2; \
    const cm_ptr* m2; \
    m_ptr_op2_result_ ## func (): m2(nullptr) {} \
    m_ptr_op2_result_ ## func (const cm_ptr& _m2) { m2 = &_m2; } \
    m_ptr_op2_result_ ## func (const m_ptr_op2_result_ ## func &) = default; \
    m_ptr_op2_result_ ## func (      m_ptr_op2_result_ ## func &&) = default; \
    m_ptr_op2_result_ ## func & operator=(const m_ptr_op2_result_ ## func &) = default; \
    m_ptr_op2_result_ ## func & operator=(      m_ptr_op2_result_ ## func &&) = default; \
    template<size_t bits1, bool masked1> void operator()(const m_ptr& m1, block_tag<bits1,masked1>) { detail::  func (m1,*m2, block_tag<bits1,masked1>(), block_tag2()); } \
    template<typename matrix_t> void resize_me(matrix_t& m) { m.resize(m2->rows,m2->columns); } \
}; \
template<typename bt, bool cv, bool al> \
inline matrix_result<m_ptr_op2_result_ ## func <bt> > func (const base_matrix_t<bt,cv,al>& m2) \
{ \
return matrix_result<m_ptr_op2_result_ ## func <bt> >(m2.ptr()); \
}

MCCL_MATRIX_RESULT_FUNCTION_OP2(m_copy)
MCCL_MATRIX_RESULT_FUNCTION_OP2(m_not)

#define MCCL_MATRIX_RESULT_FUNCTION_OP3(func) \
template<typename _block_tag2, typename _block_tag3> \
struct m_ptr_op3_result_ ## func \
{ \
    typedef _block_tag2 block_tag2; \
    typedef _block_tag3 block_tag3; \
    const cm_ptr* m2; \
    const cm_ptr* m3; \
    m_ptr_op3_result_ ## func (): m2(nullptr), m3(nullptr) {} \
    m_ptr_op3_result_ ## func (const cm_ptr& _m2, const cm_ptr& _m3) { m2 = &_m2; m3 = &_m3; } \
    m_ptr_op3_result_ ## func (const m_ptr_op3_result_ ## func &) = default; \
    m_ptr_op3_result_ ## func (      m_ptr_op3_result_ ## func &&) = default; \
    m_ptr_op3_result_ ## func & operator=(const m_ptr_op3_result_ ## func &) = default; \
    m_ptr_op3_result_ ## func & operator=(      m_ptr_op3_result_ ## func &&) = default; \
    template<size_t bits1, bool masked1> void operator()(const m_ptr& m1, block_tag<bits1,masked1>) { detail::  func (m1,*m2,*m3, block_tag<bits1,masked1>(), block_tag2(), block_tag3()); } \
    template<typename matrix_t> void resize_me(matrix_t& m) { m.resize(m2->rows,m2->columns); } \
}; \
template<typename bt2, bool cv2, bool al2, typename bt3, bool cv3, bool al3> \
inline matrix_result<m_ptr_op3_result_ ## func <bt2,bt3> > func (const base_matrix_t<bt2,cv2,al2>& m2, const base_matrix_t<bt3,cv3,al3>& m3) \
{ \
return matrix_result<m_ptr_op3_result_ ## func <bt2,bt3> >(m2.ptr(), m3.ptr()); \
}

MCCL_MATRIX_RESULT_FUNCTION_OP3(m_and)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_or)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_xor)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_nand)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_nor)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_nxor)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_andin)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_andni)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_orin)
MCCL_MATRIX_RESULT_FUNCTION_OP3(m_orni)

template<typename bt2, bool cv2, bool al2, typename bt3, bool cv3, bool al3>
inline auto operator & (const base_matrix_t<bt2,cv2,al2>& m2, const base_matrix_t<bt3,cv3,al3>& m3)
-> decltype(m_and(m2,m3))
{
    return m_and(m2,m3);
}
template<typename bt2, bool cv2, bool al2, typename bt3, bool cv3, bool al3>
inline auto operator | (const base_matrix_t<bt2,cv2,al2>& m2, const base_matrix_t<bt3,cv3,al3>& m3)
-> decltype(m_or(m2,m3))
{
    return m_or(m2,m3);
}
template<typename bt2, bool cv2, bool al2, typename bt3, bool cv3, bool al3>
inline auto operator ^ (const base_matrix_t<bt2,cv2,al2>& m2, const base_matrix_t<bt3,cv3,al3>& m3)
-> decltype(m_xor(m2,m3))
{
    return m_xor(m2,m3);
}

MCCL_END_NAMESPACE

#endif
