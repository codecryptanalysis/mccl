#ifndef MCCL_CORE_VECTOR_HPP
#define MCCL_CORE_VECTOR_HPP

#include <mccl/config/config.hpp>
#include <mccl/core/matrix_base.hpp>
#include <mccl/core/matrix_ops.hpp>

#include <array>
#include <vector>
#include <iostream>
#include <functional>

MCCL_BEGIN_NAMESPACE

/* ONE BASE VECTOR TYPE TO RULE ALL VECTOR VIEWS & VECTOR OWNER */
template<typename bt, bool cv, bool iv, bool al> class base_vector_t;

/* MAIN VECTOR TYPE DEFINITIONS */
// templated on block_tag
template<typename _block_tag = default_block_tag>
using vec_view_t = base_vector_t<_block_tag, false, false, false>;
template<typename _block_tag = default_block_tag>
using cvec_view_t = base_vector_t<_block_tag, true, false, false>;

template<typename _block_tag = default_block_tag>
using vec_view_it_t = base_vector_t<_block_tag, false, true, false>;
template<typename _block_tag = default_block_tag>
using cvec_view_it_t = base_vector_t<_block_tag, true, true, false>;

template<typename _block_tag = block_tag<256, false> >
using vec_t = base_vector_t<_block_tag, false, false, true>;

// non-templated with chosen default block tags
typedef  vec_view_t   <default_block_tag>  vec_view;
typedef cvec_view_t   <default_block_tag> cvec_view;
typedef  vec_view_it_t<default_block_tag>  vec_view_it;
typedef cvec_view_it_t<default_block_tag> cvec_view_it;

typedef  vec_t        <block_tag<256,false>>  vec;




/* IMPLEMENTATION HELPERS */
namespace detail
{

// the underlying base vector pointer type to use
template<bool _const_view, bool _iterator_view> struct vector_pointer_t;
template<> struct vector_pointer_t<false,false> { typedef   v_ptr type; };
template<> struct vector_pointer_t<true ,false> { typedef  cv_ptr type; };
template<> struct vector_pointer_t<false,true > { typedef  vi_ptr type; };
template<> struct vector_pointer_t<true ,true > { typedef cvi_ptr type; };

// different behaviour in default copy/move constructor & assignment cannot be captured in base_vector_t
// so we capture this specific behaviour only in two variant classes core_vector_t

// this is the core_vector_t for views:
// - contains a vector pointer: v_ptr, cv_ptr, vi_ptr or cvi_ptr
// - copy/move constructor copies the vector pointer
// - copy/move assignment have been deleted
template<typename _block_tag, bool _const_view, bool _iterator_view, bool _allocate>
struct core_vector_t
{
    core_vector_t() {}
    
    core_vector_t(const core_vector_t&  ) = default;
    core_vector_t(      core_vector_t&& ) = default;
    
    core_vector_t& operator=(const core_vector_t& ) = delete;
    core_vector_t& operator=(      core_vector_t&&) = delete;

    static _block_tag tag() { return _block_tag(); }

    typename vector_pointer_t<_const_view,_iterator_view>::type _ptr;

protected:
    void _swap(core_vector_t& v)
    {
        std::swap(_ptr, v._ptr);
    }

    // these are placeholder functions, but should never be called
    template<typename T> void _assign(T& ) { throw; }
};

// this is the core_vector_t for allocating vec_t:
// - contains a vector pointer: v_ptr
// - contains a vector<uint64_t> to allocate memory
// - copy constructor & assignment: allocates new memory and copies the vector contents
// - move constructor & assignment: swaps the internal vector pointer and vector<uint64_t>
template<typename _block_tag, bool _const_view, bool _iterator_view>
struct core_vector_t<_block_tag, _const_view, _iterator_view, true>
{
    core_vector_t() {}
    
    core_vector_t(const core_vector_t&  v) { _assign(v); }
    core_vector_t(      core_vector_t&& v) { _swap(v); }
    
    core_vector_t& operator=(const core_vector_t&  v) { _assign(v); return *this; }
    core_vector_t& operator=(      core_vector_t&& v) { _swap(v); return *this; }

    static _block_tag tag() { return _block_tag(); }

    static const size_t bit_alignment = 512;
    static const size_t byte_alignment = bit_alignment/8;
    static const size_t word_alignment = bit_alignment/64;

    typename vector_pointer_t<_const_view,_iterator_view>::type _ptr;

protected:
    std::vector<uint64_t> _mem;

    void _swap(core_vector_t& v)
    {
        std::swap(_ptr, v._ptr);
        _mem.swap(v._mem);
    }

    void _alloc(size_t columns, bool value = false)
    {
        if (columns == 0)
        {
            _mem.clear();
            _ptr.ptr = nullptr;
            _ptr.columns = 0;
            return;
        }
        // compute number of words to allocate, includes additional words to 'round-up' to full block PLUS desired alignment
        size_t rowwords = (columns + 63) / 64;
        rowwords = (rowwords + word_alignment-1) & ~uint64_t(word_alignment-1);
        size_t totalwords = rowwords + word_alignment;
        
        // allocate memory
        if (totalwords > _mem.size())
            _mem.resize(totalwords, value ? ~uint64_t(0) : uint64_t(0));

        // set ptr that is 'round-up' to desired alignment
        _ptr.ptr = reinterpret_cast<uint64_t*>( (uintptr_t(&_mem[0]) + byte_alignment-1) & ~uintptr_t(byte_alignment-1) );
        _ptr.columns = columns;
    }
    
    template<typename bt, bool cv, bool iv, bool al>
    void _assign(const core_vector_t<bt,cv,iv,al>& v)
    {
        _alloc(v._ptr.columns);
        v_copy(_ptr, v._ptr, tag(), v.tag());
    }

    // don't allow weird instantiations
    static_assert( _const_view == false, "core_vector_t: cannot have both _allocate and _const_view be true");
    static_assert( _iterator_view == false, "core_vector_t: cannot have both _allocate and _iterator_view be true");
};


} // namespace detail




// meta-programming construct to convert 'v.vand(v1,v2)' to 'v = v_and(v1,v2)';
// v_and(v1,v2) returns a vector_result<R> such that 'r' (of type R) contains the pointers to v1 & v2 
// and the expression 'r(v)' calls the respective function 'v.vand(v1,v2)'
// note: to allow vector to automatically resize to the correct result dimensions
//   r should have a member:
//     'template<vector_t> resize_me(vector_t&)'
template<typename v_ptr_op_result>
struct vector_result
{
    v_ptr_op_result r;
    vector_result(): r() {}
    vector_result(const vector_result&) = default;
    vector_result(vector_result&&) = default;
    vector_result& operator=(const vector_result&) = default;
    vector_result& operator=(vector_result&&) = default;
    template<typename... Args>
    vector_result(Args&&... args): r(std::forward<Args>(args)...) {}
};




/*
    the base_vector_t class from which vec_t and all vector views (vec_view_t, cvec_view_t, vec_view_it_t and cvec_view_it_t) are instantiated
*/
// _block_tag    : controls block-size during vector operations and whether to use a bitmask for the last block
// _const_view   : true results in vector view to const data
// _iterator_view: true results in vector view with row stride (i.e., a row iterator for matrices)
// _allocate     : true results in vector owner that handles memory allocates
template<typename _block_tag, bool _const_view = true, bool _iterator_view = false, bool _allocate = false>
class base_vector_t
    final : public detail::core_vector_t<_block_tag,_const_view,_iterator_view,_allocate>
{
public:
    /* TYPEDEFS */
    typedef base_vector_t<_block_tag, _const_view, _iterator_view, _allocate> this_type;
    // vector pointer type to (const) data
    typedef typename detail::vector_pointer_t<_const_view,_iterator_view>::type pointer_t;
    // vector pointer type to const data
    typedef typename detail::vector_pointer_t<true,_iterator_view>::type        const_pointer_t;
    // vector pointer type to use when vector owner object is const
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
    static const bool is_iterator = _iterator_view;
    static const bool is_owner = _allocate;
    static const bool is_view = !is_owner;
    static const bool is_vector = true;
    static_assert( is_owner == false || is_const    == false, "cannot have both is_owner == true && is_const == true");
    static_assert( is_owner == false || is_iterator == false, "cannot have both is_owner == true && is_iterator == true");

    static const bool maskedlastblock = this_block_tag::maskedlastblock;
    static const size_t block_bits = this_block_tag::bits;

    static this_block_tag tag() { return this_block_tag(); }

    // convenience macro to conditionally enable external functions on vectors
#define MCCL_ENABLE_IF_VECTOR(VT)          typename std::enable_if< VT ::is_vector,bool>::type* = nullptr
#define MCCL_ENABLE_IF_NONCONST_VECTOR(VT) typename std::enable_if< VT ::is_vector && ! VT ::is_const && (!std::is_const< VT >::value || ! VT ::is_owner),bool>::type* = nullptr

    // helper types to conditionally enable member functions
    template<typename vector_t = this_type> struct _needs_allocating_vector_type    : public std::enable_if< std::is_same<vector_t,this_type>::value && is_owner   , bool > {};
    template<typename vector_t = this_type> struct _needs_vector_view_type          : public std::enable_if< std::is_same<vector_t,this_type>::value && is_view    , bool > {};
    template<typename vector_t = this_type> struct _needs_vector_iterator_type      : public std::enable_if< std::is_same<vector_t,this_type>::value && is_iterator, bool > {};
    template<typename vector_t = this_type> struct _needs_nonconst_vector_type      : public std::enable_if< std::is_same<vector_t,this_type>::value && is_nonconst, bool > {};
    template<typename vector_t = this_type> struct _needs_nonconst_vector_view_type : public std::enable_if< std::is_same<vector_t,this_type>::value && is_nonconst && is_view, bool > {};
    // opposite helper types to generate deleted member functions with the same spec
    template<typename vector_t = this_type> struct _needs2_allocating_vector_type    : public std::enable_if< std::is_same<vector_t,this_type>::value && !is_owner   , bool > {};
    template<typename vector_t = this_type> struct _needs2_vector_view_type          : public std::enable_if< std::is_same<vector_t,this_type>::value && !is_view    , bool > {};
    template<typename vector_t = this_type> struct _needs2_vector_iterator_type      : public std::enable_if< std::is_same<vector_t,this_type>::value && !is_iterator, bool > {};
    template<typename vector_t = this_type> struct _needs2_nonconst_vector_type      : public std::enable_if< std::is_same<vector_t,this_type>::value && !is_nonconst, bool > {};
    template<typename vector_t = this_type> struct _needs2_nonconst_vector_view_type : public std::enable_if< std::is_same<vector_t,this_type>::value && !(is_nonconst && is_view), bool > {};
    // convenience macro to conditionally enable member functions
#define MCCL_BASEVEC_ENABLE_IF(s,err)       typename err = this_type, typename std::enable_if< std::is_same<err,this_type>::value && ( s ), bool>::type* = nullptr
#define MCCL_BASEVEC_ENABLE_IF_OWNER        typename func_requires_allocating_vector_type    = this_type, typename _needs_allocating_vector_type   <func_requires_allocating_vector_type   >::type* = nullptr
#define MCCL_BASEVEC_ENABLE_IF_VIEW         typename func_requires_vector_view_type          = this_type, typename _needs_vector_view_type         <func_requires_vector_view_type         >::type* = nullptr
#define MCCL_BASEVEC_ENABLE_IF_ITERATOR     typename func_requires_vector_iterator_type      = this_type, typename _needs_vector_iterator_type     <func_requires_vector_iterator_type     >::type* = nullptr
#define MCCL_BASEVEC_ENABLE_IF_NONCONST     typename func_requires_nonconst_vector_type      = this_type, typename _needs_nonconst_vector_type     <func_requires_nonconst_vector_type     >::type* = nullptr
#define MCCL_BASEVEC_ENABLE_IF_NONCONSTVIEW typename func_requires_nonconst_vector_view_type = this_type, typename _needs_nonconst_vector_view_type<func_requires_nonconst_vector_view_type>::type* = nullptr
    // convenience macros to conditionally generate deleted member functions
#define MCCL_BASEVEC_DELETE_IF(s,err)           typename err = this_type, typename std::enable_if< std::is_same<err,this_type>::value && ( s ), bool>::type* = nullptr
#define MCCL_BASEVEC_DELETE_IF_NOT_OWNER        typename func_requires_allocating_vector_type    = this_type, typename _needs2_allocating_vector_type   <func_requires_allocating_vector_type   >::type* = nullptr
#define MCCL_BASEVEC_DELETE_IF_NOT_VIEW         typename func_requires_vector_view_type          = this_type, typename _needs2_vector_view_type         <func_requires_vector_view_type         >::type* = nullptr
#define MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR     typename func_requires_vector_iterator_type      = this_type, typename _needs2_vector_iterator_type     <func_requires_vector_iterator_type     >::type* = nullptr
#define MCCL_BASEVEC_DELETE_IF_NOT_NONCONST     typename func_requires_nonconst_vector_type      = this_type, typename _needs2_nonconst_vector_type     <func_requires_nonconst_vector_type     >::type* = nullptr
#define MCCL_BASEVEC_DELETE_IF_NOT_NONCONSTVIEW typename func_requires_nonconst_vector_view_type = this_type, typename _needs2_nonconst_vector_view_type<func_requires_nonconst_vector_view_type>::type* = nullptr

// convenience macro to assert that another block tag is compatible
#define MCCL_BASEVEC_CHECK_DESTINATION_BLOCKTAG(bits,masked) static_assert( bits <= block_bits && (masked == true || maskedlastblock == false), "base_vector_t: cannot cast to specified block_tag" );

private:
    // _copy_ptr: only for views
    template<typename V>
    void _copy_ptr(V& v)
    {
        static_assert( V::block_bits >= block_bits && (V::maskedlastblock == false || maskedlastblock == true), "base_vector_t(base_vector_t): input type has incompatible block_tag" );
        if (is_owner) throw;
        ptr() = v.ptr();
    }
    template<typename V>
    void _assign(V& v)
    {
        if (is_view) throw;
        this->_assign(v);
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
        // empty for views
    }
    template<typename R>
    void _assign_result(R& r)
    {
        _resize_me_if_owner(r, std::integral_constant<bool,is_owner>());
        r(ptr(), tag());        
    }
public:

    /* CONSTRUCTORS & ASSIGNMENT */

    base_vector_t() {}

    // default copy/move constructor & assignment: behaviour is controlled by core_vector_t
    base_vector_t(const base_vector_t&  ) = default;
    base_vector_t(      base_vector_t&& ) = default;

    // if is_view == true then these default copy/move assignment are deleted because of core_vector_t
    base_vector_t& operator= (const base_vector_t&  ) = default;
    base_vector_t& operator= (      base_vector_t&& ) = default;

    // constructor taking a vector pointer
    template<MCCL_BASEVEC_DELETE_IF_NOT_VIEW> explicit base_vector_t(const pointer_t& p) = delete;
    template<MCCL_BASEVEC_ENABLE_IF_VIEW>     explicit base_vector_t(const pointer_t& p)
    {
        ptr() = p;
    }

    // constructor that creates a new vector with a specified number of columns
    template<MCCL_BASEVEC_DELETE_IF_NOT_OWNER> explicit base_vector_t(size_t _columns, bool value = false) = delete;
    template<MCCL_BASEVEC_ENABLE_IF_OWNER>     explicit base_vector_t(size_t _columns, bool value = false) 
    {
        this->_alloc(_columns, value);
    }

    // construct from another base_vector_t: copy view or assign content depending on is_view
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF(is_view && !(is_const || cv==false), nonconst_view_requires_nonconst_vector)>
    explicit base_vector_t(      base_vector_t<bt,cv,iv,al>& v) = delete;
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF(is_view && !(is_const || (cv==false&&al==false)), nonconst_view_requires_nonconst_vector)>
    explicit base_vector_t(const base_vector_t<bt,cv,iv,al>& v) = delete;

    // for views we have two constructors: with non-const ref (to be initialized from owner), and const ref (other non-const views)
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF(is_view && (is_const || cv==false), nonconst_view_requires_nonconst_vector)>
    explicit base_vector_t(base_vector_t<bt,cv,iv,al>& v)
    {
        _copy_ptr(v);
    }
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF(is_view && (is_const || (cv==false&&al==false)), nonconst_view_requires_nonconst_vector)>
    explicit base_vector_t(const base_vector_t<bt,cv,iv,al>& v)
    {
        _copy_ptr(v);
    }
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF_OWNER>
    explicit base_vector_t(const base_vector_t<bt,cv,iv,al>& v)
    {
        _assign(v);
    }

    // construct from a vector_result
    template<typename F, MCCL_BASEVEC_DELETE_IF_NOT_OWNER> base_vector_t(vector_result<F>&& vr) = delete;
    template<typename F, MCCL_BASEVEC_ENABLE_IF_OWNER>     base_vector_t(vector_result<F>&& vr)
    {
        _assign_result(vr.r);
    }


    // assign from another base_vector_t
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF_NOT_OWNER> base_vector_t& operator= (const base_vector_t<bt,cv,iv,al>& v) = delete;
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF_OWNER>     base_vector_t& operator= (const base_vector_t<bt,cv,iv,al>& v)
    {
        this->_assign(v);
        return *this;
    }

    // assign from a vector_result
    template<typename F, MCCL_BASEVEC_DELETE_IF_NOT_NONCONST> base_vector_t& operator= (vector_result<F>&& vr) = delete;
    template<typename F, MCCL_BASEVEC_ENABLE_IF_NONCONST>     base_vector_t& operator= (vector_result<F>&& vr)
    {
        _assign_result(vr.r);
        return *this;
    }
    template<typename F, MCCL_BASEVEC_DELETE_IF_NOT_NONCONSTVIEW> const base_vector_t& operator= (vector_result<F>&& vr) const = delete;
    template<typename F, MCCL_BASEVEC_ENABLE_IF_NONCONSTVIEW>     const base_vector_t& operator= (vector_result<F>&& vr) const
    {
        _assign_result(vr.r);
        return *this;
    }
    

    /* OWNER specific member functions */

    // resize vector to specific nr of columns
    // WARNING: if cols > columns() then this may reallocate memory and any pointers to this vector become invalid!
    template<MCCL_BASEVEC_DELETE_IF_NOT_OWNER> this_type& resize(size_t cols, bool value = false) = delete;
    template<MCCL_BASEVEC_ENABLE_IF_OWNER>     this_type& resize(size_t cols, bool value = false)
    {
        // when shrinking we only update nr of columns
        if (cols <= columns())
        {
            ptr().columns = cols;
            if (cols == 0)
                ptr().ptr = nullptr;
            return *this;
        }
        // otherwise we create a new larger vector
        base_vector_t tmp(cols, value);
        // copy old content
        if (columns() > 0)
            tmp.subvector(0, columns()).v_copy(*this);
        // and swap
        tmp.swap(*this);
        return *this;
    }


    /* VIEW specific member functions */

    // reset view to another view given by compatible pointer or vector view
    template<MCCL_BASEVEC_DELETE_IF_NOT_VIEW> void reset(const pointer_t& p) = delete;
    template<MCCL_BASEVEC_ENABLE_IF_VIEW>     void reset(const pointer_t& p)
    {
        ptr() = p;
    }
    template<MCCL_BASEVEC_DELETE_IF_NOT_VIEW> void reset(const this_type& v) = delete;
    template<MCCL_BASEVEC_ENABLE_IF_VIEW>     void reset(const this_type& v)
    {
        ptr() = v.ptr();
    }


    /* VIEW ITERATOR specific member functions */
    
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     this_type& operator++()               { ++ptr(); return *this; }
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     this_type& operator--()               { --ptr(); return *this; }
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     this_type& operator+=(size_t n)       { ptr()+=n; return *this; }
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     this_type& operator-=(size_t n)       { ptr()-=n; return *this; }
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     this_type  operator++(int)            { return this_type(ptr()++); }
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     this_type  operator--(int)            { return this_type(ptr()--); }
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     this_type  operator+ (size_t n) const { return this_type(ptr()+n); }
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     this_type  operator- (size_t n) const { return this_type(ptr()-n); }
    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>     ptrdiff_t  operator- (const this_type& v2) const { return ptr() - v2.ptr(); }

    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> this_type& operator++()               = delete;
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> this_type& operator--()               = delete;
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> this_type& operator+=(size_t n)       = delete;
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> this_type& operator-=(size_t n)       = delete;
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> this_type  operator++(int)            = delete;
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> this_type  operator--(int)            = delete;
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> this_type  operator+ (size_t n) const = delete;
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> this_type  operator- (size_t n) const = delete;
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR> ptrdiff_t  operator- (const this_type& v2) const = delete;

    template<MCCL_BASEVEC_ENABLE_IF_ITERATOR>
    size_t word_stride() const { return ptr().stride; }
    template<MCCL_BASEVEC_DELETE_IF_NOT_ITERATOR>
    size_t word_stride() const = delete;
    
    /* COMMON basic member functions */

    size_t columns() const { return ptr().columns; }
    size_t hw() const { return detail::v_hw(ptr()); }
    
    bool operator[](size_t c) const { return detail::v_getbit(ptr(),c); }
    bool operator()(size_t c) const { return detail::v_getbit(ptr(),c); }

    // swap vectors by swapping member variables
    this_type& swap(this_type& v)
    {
        this->_swap(v);
        return *this;
    }


    /* SUBVECTOR functions */
    
    // subvector always returns a view. for a const owner this view is forced to be const
    template<typename block_tag>
    using this_view = base_vector_t<block_tag, is_const, is_iterator, false>;
    template<typename block_tag>
    using this_view_const = base_vector_t<block_tag, is_const||is_owner, is_iterator, false>;
    
    typedef block_tag<block_bits,true> this_block_tag_masked;

    // whole subvector maintains block_tag
    this_view      <this_block_tag>    subvector()       { return this_view      <this_block_tag>( ptr() ); }
    this_view_const<this_block_tag>    subvector() const { return this_view_const<this_block_tag>( ptr() ); }
    // subvector starting from 0 maintains block_tag::bits, but maskedlastblock = true
    this_view      <this_block_tag_masked>    subvector(size_t cols)       { return this_view      <this_block_tag_masked>( ptr().subvector(0,cols) ); }
    this_view_const<this_block_tag_masked>    subvector(size_t cols) const { return this_view_const<this_block_tag_masked>( ptr().subvector(0,cols) ); }
    // otherwise returned view has default_block_tag = block_tag<64,true>
    this_view      <default_block_tag> subvector(size_t coloff, size_t cols)       { return this_view      <default_block_tag>( ptr().subvector(coloff,cols) ); }
    this_view_const<default_block_tag> subvector(size_t coloff, size_t cols) const { return this_view_const<default_block_tag>( ptr().subvector(coloff,cols) ); }

    // unless the user overrides with a compatible tag
    template<size_t bits, bool masked>
    this_view<block_tag<bits,masked>> subvector(block_tag<bits,masked>) 
    {
        MCCL_BASEVEC_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_view<block_tag<bits,masked>>( ptr() ); 
    }
    template<size_t bits, bool masked>
    this_view_const<block_tag<bits,masked>> subvector(block_tag<bits,masked>) const
    {
        MCCL_BASEVEC_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_view_const <block_tag<bits,masked>>( ptr() );
    }
    template<size_t bits, bool masked>
    this_view<block_tag<bits,masked>> subvector(size_t cols, block_tag<bits,masked>) 
    {
        MCCL_BASEVEC_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_view<block_tag<bits,masked>>( ptr().subvector(0,cols) ); 
    }
    template<size_t bits, bool masked>
    this_view_const<block_tag<bits,masked>> subvector(size_t cols, block_tag<bits,masked>) const
    {
        MCCL_BASEVEC_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_view_const <block_tag<bits,masked>>( ptr().subvector(0,cols) );
    }
    template<size_t bits, bool masked>
    this_view<block_tag<bits,masked>> subvector(size_t coloff, size_t cols, block_tag<bits,masked>)
    {
        MCCL_BASEVEC_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_view<block_tag<bits,masked>>( ptr().subvector(coloff,cols) );
    }
    template<size_t bits, bool masked>
    this_view_const<block_tag<bits,masked>> subvector(size_t coloff, size_t cols, block_tag<bits,masked>) const
    {
        MCCL_BASEVEC_CHECK_DESTINATION_BLOCKTAG(bits,masked);
        return this_view_const<block_tag<bits,masked>>( ptr().subvector(coloff,cols) );
    }


    /* ACTUAL VECTOR OPERATIONS */
    
    // swap vectors by swapping content
    // deleted for const vectors
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF_NOT_NONCONST>           this_type& v_swap(const base_vector_t<bt,cv,iv,al>& v)       = delete;
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF_NOT_NONCONSTVIEW> const this_type& v_swap(const base_vector_t<bt,cv,iv,al>& v) const = delete;

    // in order to swap: owners require a nonconst reference    
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF_NONCONST>
    this_type& v_swap(base_vector_t<bt,cv,iv,al>& v)
    {
        detail::v_swap(ptr(), v.ptr(), tag(), v.tag());
        return *this;
    }
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF_NONCONSTVIEW>
    this_type& v_swap(base_vector_t<bt,cv,iv,al>& v) const
    {
        detail::v_swap(ptr(), v.ptr(), tag(), v.tag());
        return *this;
    }
    // in order to swap: views remain modifiable when passed as const reference
    template<typename bt, bool cv, bool iv, MCCL_BASEVEC_ENABLE_IF_NONCONST>
    this_type& v_swap(const base_vector_t<bt,cv,iv,false>& v)
    {
        detail::v_swap(ptr(), v.ptr(), tag(), v.tag());
        return *this;
    }
    template<typename bt, bool cv, bool iv, MCCL_BASEVEC_ENABLE_IF_NONCONSTVIEW>
    this_type& v_swap(const base_vector_t<bt,cv,iv,false>& v) const
    {
        detail::v_swap(ptr(), v.ptr(), tag(), v.tag());
        return *this;
    }


    // modifing member functions
    // note that a const version is defined only for views, not for owners
#define MCCL_BASEVEC_MEMBER_FUNC(spec,expr) \
    template<MCCL_BASEVEC_DELETE_IF_NOT_NONCONST>           this_type& spec       = delete;               \
    template<MCCL_BASEVEC_DELETE_IF_NOT_NONCONSTVIEW> const this_type& spec const = delete;               \
    template<MCCL_BASEVEC_ENABLE_IF_NONCONST>               this_type& spec       { expr; return *this; } \
    template<MCCL_BASEVEC_ENABLE_IF_NONCONSTVIEW>     const this_type& spec const { expr; return *this; }
    
    MCCL_BASEVEC_MEMBER_FUNC( v_clear ()                 , detail::v_clear(ptr(), tag());    )
    MCCL_BASEVEC_MEMBER_FUNC( v_not   ()                 , detail::v_not  (ptr(), tag());    )
    MCCL_BASEVEC_MEMBER_FUNC( v_set   ()                 , detail::v_set  (ptr(), tag());    )
    MCCL_BASEVEC_MEMBER_FUNC( v_set   (bool b)           , detail::v_set  (ptr(), b, tag()); )
    MCCL_BASEVEC_MEMBER_FUNC( clearbit(size_t c)         , detail::v_clearbit(ptr(), c);     )
    MCCL_BASEVEC_MEMBER_FUNC( flipbit (size_t c)         , detail::v_flipbit (ptr(), c);     )
    MCCL_BASEVEC_MEMBER_FUNC( setbit  (size_t c)         , detail::v_setbit  (ptr(), c);     )
    MCCL_BASEVEC_MEMBER_FUNC( setbit  (size_t c, bool b) , detail::v_setbit  (ptr(), c, b);  )
    MCCL_BASEVEC_MEMBER_FUNC( setcolumns  (size_t c_off, size_t c_cnt, bool b) , detail::v_setcolumns  (ptr(), c_off, c_cnt, b); )
    MCCL_BASEVEC_MEMBER_FUNC( setcolumns  (size_t c_off, size_t c_cnt)         , detail::v_setcolumns  (ptr(), c_off, c_cnt);    )
    MCCL_BASEVEC_MEMBER_FUNC( clearcolumns(size_t c_off, size_t c_cnt)         , detail::v_clearcolumns(ptr(), c_off, c_cnt);    )
    MCCL_BASEVEC_MEMBER_FUNC( flipcolumns (size_t c_off, size_t c_cnt)         , detail::v_flipcolumns (ptr(), c_off, c_cnt);    )

    
    // 1-vector member functions that translate to 2-vector operations with current object the destination vector
    // note that a const version is defined only for views, not for owners
#define MCCL_BASEVEC_1OP_MEMBER_FUNC(func) \
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF_NOT_NONCONST>           this_type& func (const base_vector_t<bt,cv,iv,al>& v1)       = delete; \
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF_NOT_NONCONSTVIEW> const this_type& func (const base_vector_t<bt,cv,iv,al>& v1) const = delete; \
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF_NONCONST>     \
    this_type& func (const base_vector_t<bt,cv,iv,al>& v1)                                \
    { detail:: func (ptr(), v1.ptr(), tag(), v1.tag()); return *this; }                   \
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF_NONCONSTVIEW> \
    const this_type& func (const base_vector_t<bt,cv,iv,al>& v1) const                    \
    { detail:: func (ptr(), v1.ptr(), tag(), v1.tag()); return *this; }

    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_copy)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_not)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_and)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_xor)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_or)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_nand)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_nxor)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_nor)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_andin)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_andni)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_orin)
    MCCL_BASEVEC_1OP_MEMBER_FUNC(v_orni)


    // 1-vector member operators that translate to 2-vector operations with current object the destination vector
    // note that a const version is defined only for views, not for owners
#define MCCL_BASEVEC_1OP_MEMBER_FUNC_SYM(sym, func) \
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF_NOT_NONCONST>           this_type& operator sym (const base_vector_t<bt,cv,iv,al>& v1)       = delete; \
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_DELETE_IF_NOT_NONCONSTVIEW> const this_type& operator sym (const base_vector_t<bt,cv,iv,al>& v1) const = delete; \
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF_NONCONST>     \
    this_type& operator sym (const base_vector_t<bt,cv,iv,al>& v1)                        \
    { detail:: func (ptr(), v1.ptr(), tag(), v1.tag()); return *this; }                   \
    template<typename bt, bool cv, bool iv, bool al, MCCL_BASEVEC_ENABLE_IF_NONCONSTVIEW> \
    const this_type& operator sym (const base_vector_t<bt,cv,iv,al>& v1) const            \
    { detail:: func (ptr(), v1.ptr(), tag(), v1.tag()); return *this; }

    MCCL_BASEVEC_1OP_MEMBER_FUNC_SYM( &= , v_and )
    MCCL_BASEVEC_1OP_MEMBER_FUNC_SYM( |= , v_or )
    MCCL_BASEVEC_1OP_MEMBER_FUNC_SYM( ^= , v_xor )

    
    // 2-vector member functions that translate to 3-vector operations with current object the destination vector
    // note that a const version is defined only for views, not for owners
#define MCCL_BASEVEC_2OP_MEMBER_FUNC(func) \
    template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2, MCCL_BASEVEC_DELETE_IF_NOT_NONCONST>           this_type& func (const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2)       = delete; \
    template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2, MCCL_BASEVEC_DELETE_IF_NOT_NONCONSTVIEW> const this_type& func (const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2) const = delete; \
    template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2, MCCL_BASEVEC_ENABLE_IF_NONCONST>     \
    this_type& func (const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2)                                  \
    { detail:: func (ptr(), v1.ptr(), v2.ptr(), tag(), v1.tag(), v2.tag()); return *this; }                                               \
    template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2, MCCL_BASEVEC_ENABLE_IF_NONCONSTVIEW> \
    const this_type& func (const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2) const                      \
    { detail:: func (ptr(), v1.ptr(), v2.ptr(), tag(), v1.tag(), v2.tag()); return *this; }

    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_and)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_xor)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_or)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_nand)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_nxor)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_nor)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_andin)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_andni)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_orin)
    MCCL_BASEVEC_2OP_MEMBER_FUNC(v_orni)


    /* RAW DATA FUNCTIONS */
    
          pointer_t      & ptr()       { return this->_ptr; }
    const pointer_t_const& ptr() const { return this->_ptr; }

    size_t row_words()  const { return (ptr().columns+63)/64; }
    auto word_ptr()       -> decltype(ptr().ptr) { return ptr().ptr; }
    auto word_ptr() const -> decltype(ptr().ptr) { return ptr().ptr; }
    
    size_t row_blocks() const { return (ptr().columns+block_bits-1)/block_bits; }
    auto block_ptr()       -> decltype(make_block_ptr(ptr().ptr, tag())) { return make_block_ptr(ptr().ptr, tag()); }
    auto block_ptr() const -> decltype(make_block_ptr(ptr().ptr, tag())) { return make_block_ptr(ptr().ptr, tag()); }

    template<size_t bits, bool maskedlastword>
    size_t row_blocks(block_tag<bits,maskedlastword>) const 
    {
        return (ptr().columns+bits-1)/bits;
    }
    template<size_t bits, bool maskedlastword>
    auto block_ptr(block_tag<bits,maskedlastword>)       -> decltype(make_block_ptr(ptr().ptr, block_tag<bits,maskedlastword>()))
    {
        return make_block_ptr(ptr().ptr, block_tag<bits,maskedlastword>());
    }
    template<size_t bits, bool maskedlastword>
    auto block_ptr(block_tag<bits,maskedlastword>) const -> decltype(make_block_ptr(ptr().ptr, block_tag<bits,maskedlastword>()))
    {
        return make_block_ptr(ptr().ptr, block_tag<bits,maskedlastword>());
    }

    template<typename bt, bool cv, bool iv, bool al>
    bool is_equal(const base_vector_t<bt,cv,iv,al>& v2) const
    {
        return detail::v_isequal(ptr(),v2.ptr(),tag(),v2.tag());
    }


    /* CONVERSION OPERATOR */

    // conversion from 'base_vector_t<..>&' to another 'base_vector_t<..>&' is allowed IF:
    // - bt is compatible: bt::bits <= block_bits AND (bt::maskedlastblock==true OR maskedlastblock==false)
    // - cv==true   if  is_const == true
    // - iv==false  if  is_iterator == false
    template<typename bt, bool cv, bool iv, MCCL_BASEVEC_ENABLE_IF( (cv|is_const) == cv && (iv|is_iterator) == is_iterator && bt::bits <= block_bits && (bt::maskedlastblock == true || maskedlastblock == false), destination_type_is_incompatible )>
    operator base_vector_t<bt,cv,iv,false>& ()
    {
        return * reinterpret_cast< base_vector_t<bt,cv,iv,false>* >(this);
    }
    
    // conversion from '(const) base_vector_t<..>&' to another 'const base_vector_t<..>&' is allowed IF:
    // - bt is compatible: bt::bits <= block_bits AND (bt::maskedlastblock==true OR maskedlastblock==false)
    // - cv==true   if  is_const == true OR is_owner == true
    // - iv==false  if  is_iterator == false
    template<typename bt, bool cv, bool iv, MCCL_BASEVEC_ENABLE_IF( (cv|is_const) == cv && (cv|is_owner) == cv && (iv|is_iterator) == is_iterator && bt::bits <= block_bits && (bt::maskedlastblock == true || maskedlastblock == false), destination_type_is_incompatible )>
    operator const base_vector_t<bt,cv,iv,false>& ()
    {
        return * reinterpret_cast< const base_vector_t<bt,cv,iv,false>* >(this);
    }
    template<typename bt, bool cv, bool iv, MCCL_BASEVEC_ENABLE_IF( (cv|is_const) == cv && (cv|is_owner) == cv && (iv|is_iterator) == is_iterator && bt::bits <= block_bits && (bt::maskedlastblock == true || maskedlastblock == false), destination_type_is_incompatible )>
    operator const base_vector_t<bt,cv,iv,false>& () const
    {
        return * reinterpret_cast< const base_vector_t<bt,cv,iv,false>* >(this);
    }

    /* OVERRULE BLOCKTAG, note: returns reference */
    
    template<size_t bits, bool masked>
    base_vector_t<block_tag<bits,masked>,is_const,is_iterator,false>& as(block_tag<bits,masked>)
    {
        return * reinterpret_cast< base_vector_t<block_tag<bits,masked>,is_const,is_iterator,false>* >(this);
    }
    template<size_t bits, bool masked>
    const base_vector_t<block_tag<bits,masked>,is_const||is_owner,is_iterator,false>& as(block_tag<bits,masked>) const
    {
        return * reinterpret_cast< const base_vector_t<block_tag<bits,masked>,is_const||is_owner,is_iterator,false>* >(this);
    }

};


/* COMPARISON */

template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2>
inline bool operator==(const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2)
{ return v1.is_equal(v2); }

template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2>
inline bool operator!=(const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2)
{ return ! v1.is_equal(v2); }


/* OUTPUT */

template<typename bt1, bool cv1, bool iv1, bool al1>
inline std::ostream& operator<<(std::ostream& o, const base_vector_t<bt1,cv1,iv1,al1>& v) { detail::v_print(o, v.ptr()); return o; }


/* HAMMINGWEIGHT */

template<typename bt, bool cv, bool iv, bool al>
inline size_t hammingweight(const base_vector_t<bt,cv,iv,al>& v) { return v.hw(); }

template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2>
inline size_t hammingweight_and(const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2)
{
    return detail::v_hw_and(v1.ptr(),v2.ptr());
}
template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2>
inline size_t hammingweight_or(const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2)
{
    return detail::v_hw_or(v1.ptr(),v2.ptr());
}
template<typename bt1, bool cv1, bool iv1, bool al1, typename bt2, bool cv2, bool iv2, bool al2>
inline size_t hammingweight_xor(const base_vector_t<bt1,cv1,iv1,al1>& v1, const base_vector_t<bt2,cv2,iv2,al2>& v2)
{
    return detail::v_hw_xor(v1.ptr(),v2.ptr());
}


/* VECTOR OPERATIONS THAT RETURN A VECTOR_RESULT */

#define MCCL_VECTOR_RESULT_FUNCTION_OP2(func) \
template<typename _block_tag2> \
struct v_ptr_op2_result_ ## func \
{ \
    typedef _block_tag2 block_tag2; \
    const cv_ptr* v2; \
    v_ptr_op2_result_ ## func (): v2(nullptr) {} \
    v_ptr_op2_result_ ## func (const cv_ptr& _v2) { v2 = &_v2; } \
    v_ptr_op2_result_ ## func (const v_ptr_op2_result_ ## func &) = default; \
    v_ptr_op2_result_ ## func (      v_ptr_op2_result_ ## func &&) = default; \
    v_ptr_op2_result_ ## func & operator=(const v_ptr_op2_result_ ## func &) = default; \
    v_ptr_op2_result_ ## func & operator=(      v_ptr_op2_result_ ## func &&) = default; \
    template<size_t bits1, bool masked1> void operator()(const v_ptr& v1, block_tag<bits1,masked1>) { detail::  func (v1,*v2, block_tag<bits1,masked1>(), block_tag2()); } \
    template<typename vector_t> void resize_me(vector_t& v) { v.resize(v2->columns); } \
}; \
template<typename bt, bool cv, bool iv, bool al> \
inline vector_result<v_ptr_op2_result_ ## func <bt> > func (const base_vector_t<bt,cv,iv,al>& v2) \
{ \
return vector_result<v_ptr_op2_result_ ## func <bt> >(v2.ptr()); \
}

MCCL_VECTOR_RESULT_FUNCTION_OP2(v_copy)
MCCL_VECTOR_RESULT_FUNCTION_OP2(v_not)

#define MCCL_VECTOR_RESULT_FUNCTION_OP3(func) \
template<typename _block_tag2, typename _block_tag3> \
struct v_ptr_op3_result_ ## func \
{ \
    typedef _block_tag2 block_tag2; \
    typedef _block_tag3 block_tag3; \
    const cv_ptr* v2; \
    const cv_ptr* v3; \
    v_ptr_op3_result_ ## func (): v2(nullptr), v3(nullptr) {} \
    v_ptr_op3_result_ ## func (const cv_ptr& _v2, const cv_ptr& _v3) { v2 = &_v2; v3 = &_v3; } \
    v_ptr_op3_result_ ## func (const v_ptr_op3_result_ ## func &) = default; \
    v_ptr_op3_result_ ## func (      v_ptr_op3_result_ ## func &&) = default; \
    v_ptr_op3_result_ ## func & operator=(const v_ptr_op3_result_ ## func &) = default; \
    v_ptr_op3_result_ ## func & operator=(      v_ptr_op3_result_ ## func &&) = default; \
    template<size_t bits1, bool masked1> void operator()(const v_ptr& v1, block_tag<bits1,masked1>) { detail::  func (v1,*v2,*v3, block_tag<bits1,masked1>(), block_tag2(), block_tag3()); } \
    template<typename vector_t> void resize_me(vector_t& v) { v.resize(v2->columns); } \
}; \
template<typename bt2, bool cv2, bool iv2, bool al2, typename bt3, bool cv3, bool iv3, bool al3> \
inline vector_result<v_ptr_op3_result_ ## func <bt2,bt3> > func (const base_vector_t<bt2,cv2,iv2,al2>& v2, const base_vector_t<bt3,cv3,iv3,al3>& v3) \
{ \
return vector_result<v_ptr_op3_result_ ## func <bt2,bt3> >(v2.ptr(), v3.ptr()); \
}

MCCL_VECTOR_RESULT_FUNCTION_OP3(v_and)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_or)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_xor)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_nand)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_nor)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_nxor)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_andin)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_andni)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_orin)
MCCL_VECTOR_RESULT_FUNCTION_OP3(v_orni)

template<typename bt2, bool cv2, bool iv2, bool al2, typename bt3, bool cv3, bool iv3, bool al3>
inline auto operator & (const base_vector_t<bt2,cv2,iv2,al2>& v2, const base_vector_t<bt3,cv3,iv3,al3>& v3)
-> decltype(v_and(v2,v3))
{
    return v_and(v2,v3);
}
template<typename bt2, bool cv2, bool iv2, bool al2, typename bt3, bool cv3, bool iv3, bool al3>
inline auto operator | (const base_vector_t<bt2,cv2,iv2,al2>& v2, const base_vector_t<bt3,cv3,iv3,al3>& v3)
-> decltype(v_or(v2,v3))
{
    return v_or(v2,v3);
}
template<typename bt2, bool cv2, bool iv2, bool al2, typename bt3, bool cv3, bool iv3, bool al3>
inline auto operator ^ (const base_vector_t<bt2,cv2,iv2,al2>& v2, const base_vector_t<bt3,cv3,iv3,al3>& v3)
-> decltype(v_xor(v2,v3))
{
    return v_xor(v2,v3);
}

MCCL_END_NAMESPACE

#endif
