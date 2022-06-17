#ifndef MCCL_CORE_COLLECTION_HPP
#define MCCL_CORE_COLLECTION_HPP

#include <mccl/config/config.hpp>

#include <stdlib.h>

#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <new>
#include <array>
#include <vector>
#include <deque>
#include <algorithm>
#include <iterator>
#include <atomic>
#include <mutex>

MCCL_BEGIN_NAMESPACE

namespace detail
{
	// pointers should be allocated through aligned_alloc:
	void* mccl_aligned_alloc(std::size_t size, std::size_t alignment)
	{
#if 1
		return ::aligned_alloc(alignment, size);
#else
		void* ptr = nullptr;
		int err = ::posix_memalign(&ptr, alignment, size);
		if (err != 0)
		{
			if (err == EINVAL)
				throw std::runtime_error("mccl_aligned_alloc: posix_memalign: error invalid input");
			if (err == ENOMEM)
				throw std::runtime_error("mccl_aligned_alloc: posix_memalign: error insufficient memory");
			throw std::runtime_error("mccl_aligned_alloc: posix_memalign: error unknown");
		}
		return ptr;
#endif
	}
	
	// pointers should be freed through aligned_free:
	void mccl_aligned_free(void* p)
	{
		::free( p );
	}

	// multi consumer multi producer unbounded queue
	// implemented as simple wrapper around std::deque
	template<typename T, typename Mutex = std::mutex>
	class mccl_concurrent_queue
	{
	public:
		typedef Mutex mutex_type;
		typedef std::lock_guard<mutex_type> lock_type;
		typedef std::deque<T> queue_type;
		
		typedef T value_type;

		std::size_t size()
		{
			lock_type lock(_mutex);
			return _queue.size();
		}
		
		bool empty()
		{
			lock_type lock(_mutex);
			return _queue.empty();
		}
		
		void push_back(const value_type& v)
		{
			_emplace_back(v);
		}

		void push_back(value_type&& v)
		{
			_emplace_back(std::move(v));
		}

		template<typename... Args>
		void emplace_back(Args&&... args)
		{
			_emplace_back(std::forward<Args>(args)...);
		}
		
		bool try_pop_front(value_type& v)
		{
			lock_type lock(_mutex);
			if (_queue.empty())
				return false;
			v = std::move(_queue.front());
			_queue.pop_front();
			return true;
		}
		
	private:
		template<typename... Args>
		void _emplace_back(Args&&... args)
		{
			lock_type lock(_mutex);
			_queue.emplace_back( std::forward<Args>(args)... );
		}
		
		mutex_type _mutex;
		queue_type _queue;
	};
	
	// memory allocator pool for fixed size pages
	// do not use page_allocator before static members have been initialized
	// freeing pages after end of main (i.e. during static deconstructors) leads to undefined behaviour
	template<std::size_t PageSize = (1<<20)>
	class mccl_page_allocator {
	public:
		typedef mccl_concurrent_queue<void*> queue_type;

		static const std::size_t page_size = PageSize;

		static constexpr std::size_t page_alignment() { return _page_alignment; }
		
		// obtain page from free page pool, otherwise allocate a new one
		static void* alloc_page()
		{
			void* p = nullptr;
			if (! _helper._queue.try_pop_front(p))
			{
#if 1
				// allocate per page
				p = mccl_aligned_alloc(page_size, _page_alignment);
#else
				// allocate multiple pages at once
				std::lock_guard<std::mutex> lock(_helper._mutex);
				if (!_helper._queue.try_pop_front(p))
				{
					const std::size_t alloc_size = page_size * 32;
					p = mccl_aligned_alloc(alloc_size, _page_alignment);
					for (std::size_t x = page_size; x < alloc_size; x += page_size)
						_helper._queue.emplace_back(p + x);
				}
#endif
			}
			return p;
		}

		// free pages go into pool, not actually freed
		static void free_page(void* p)
		{
			_helper._queue.push_back(p);
		}
		
	private:
		static const std::size_t _page_alignment = 64;

		// freed pages are not returned to heap
		// but stored in queue for future page allocations instead
		// only at program end all pages are freed
		struct _static_helper {
			// concurrent queue to store freed pages
			queue_type _queue;
			std::size_t _alignment = _page_alignment;
			std::mutex _mutex;

			// free queue at program end
			~_static_helper()
			{
				void* p = nullptr;
				while (_queue.try_pop_front(p))
					mccl_aligned_free(p);
				if (_queue.size() > 0)
					throw std::runtime_error("page_allocator: could not free all pages in queue");
			}
		};
		static _static_helper _helper;
	};
	
}


/* class page_vector<T>: 
	mimics std::vector<T>: dynamic storage of contiguous array of elements of type T
	except it has a special allocator that allocates fixed size pages for storage
	hence its capacity is also fixed:
	- no reallocation during resizing/push_back
	- throws when attempting to exceed capacity
*/
template<typename T, typename Alloc = detail::mccl_page_allocator<> >
class page_vector
{
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;
	typedef const T* const_pointer;
	typedef const T& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

	typedef Alloc page_allocator_type;
	static const std::size_t page_size = page_allocator_type::page_size;
	static const std::size_t page_alignment = page_allocator_type::page_alignment;
	static const std::size_t alignment_cost = ((page_alignment % alignof(value_type)) == 0) ? 0 : alignof(value_type);
	static const std::size_t page_capacity = (page_size - alignment_cost) / sizeof(value_type);

	~page_vector()
	{
		_shrink(0);
		_free_page();
	}

	page_vector() = default;
	
	explicit page_vector(size_type count)
	{
		if (count > 0)
			_grow(count);
	}
	
	page_vector(size_type count, const value_type& value)
	{
		if (count > 0)
			_grow(count, value);
	}
	
	template<typename InputIt>
	page_vector(InputIt first, InputIt last)
	{
		_assign(std::distance(first,last), first, last);
	}
	
	page_vector(const page_vector& v)
	{
		_assign(v.size(), v.begin(), v.end());
	}
	
	page_vector(page_vector&& v) noexcept
		: _page(v._page), _data(v._data), _size(v._size)
	{
		v._page = nullptr;
		v._data = nullptr;
		v._size = 0;
	}
	
	page_vector(std::initializer_list<value_type> init) : page_vector()
	{
		_assign(init.size(), init.begin(), init.end());
	}
	

	page_vector& operator=(const page_vector& v)
	{
		_assign(v.size(), v.begin(), v.end());
		return *this;
	}
	page_vector& operator=(page_vector&& v) noexcept
	{
		std::swap(_page, v._page);
		std::swap(_data, v._data);
		std::swap(_size, v._size);
		return *this;
	}
	page_vector& operator=(std::initializer_list<value_type> init)
	{
		_assign(init.size(), init.begin(), init.end());
		return *this;
	}
		
	void assign(size_type count, const value_type& value)
	{
		if (count < _size)
			_shrink(count);
		for (auto it = begin(); it != end(); ++it)
			*it = value;
		if (count > _size)
			_grow(count, value);
	}
	
	template<typename InputIt>
	void assign(InputIt first, InputIt last)
	{
		_assign(std::distance(first,last), first, last);
	}
	
	void assign(std::initializer_list<value_type> init)
	{
		_assign(init.size(), init.begin(), init.end());
	}

	reference at(size_type pos)
	{
		if (pos >= _size)
			throw std::out_of_range("page_vector::at(): pos >= size()");
		return *(_data+pos);
	}
	const_reference at(size_type pos) const
	{
		if (pos >= _size)
			throw std::out_of_range("page_vector::at(): pos >= size()");
		return *(_data+pos);
	}
	reference operator[](size_type pos)
	{
		return *(_data+pos);
	}
	const_reference operator[](size_type pos) const
	{
		return *(_data+pos);
	}
	
	      reference front()       noexcept { return *_data; }
	const_reference front() const noexcept { return *_data; }
	      reference back()       noexcept { return *(_data+_size-1); }
	const_reference back() const noexcept { return *(_data+_size-1); }
	      pointer data()       noexcept { return _data; }
	const_pointer data() const noexcept { return _data; }
	
	      iterator  begin()       noexcept { return _data; }
	const_iterator  begin() const noexcept { return _data; }
	const_iterator cbegin() const noexcept { return _data; }
	      iterator  end()       noexcept { return _data+_size; }
	const_iterator  end() const noexcept { return _data+_size; }
	const_iterator cend() const noexcept { return _data+_size; }

	      reverse_iterator  rbegin()       noexcept { return reverse_iterator(end()); }
	const_reverse_iterator  rbegin() const noexcept { return const_reverse_iterator(end()); }
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
	      reverse_iterator  rend()       noexcept { return reverse_iterator(begin()); }
	const_reverse_iterator  rend() const noexcept { return const_reverse_iterator(begin()); }
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

	bool empty() const { return _size == 0; }
	size_type size() const { return _size; }
	size_type max_size() const { return page_capacity; }
	size_type capacity() const { return _page == nullptr ? 0 : page_capacity; }
	
	void reserve(size_type new_cap)
	{
		if (new_cap > 0)
			_alloc_page();
	}
	
	void shrink_to_fit()
	{
		if (_size == 0)
			_free_page();
	}
	
	void clear()
	{
		_shrink(0);
	}
	
	void free()
	{
		_shrink(0);
		_free_page();
	}
	
	iterator insert(const_iterator pos, const value_type& value)
	{
		return _insert_single(pos, value);
	}
	
	iterator insert(const_iterator pos, value_type&& value)
	{
		return _insert_single(pos, std::move(value));
	}
	
	iterator insert(const_iterator pos, size_type count, const value_type& value)
	{
		return _insert_multi(pos, count, value);
	}
	
	template<typename InputIt>
	iterator insert(const_iterator pos, InputIt first, InputIt last)
	{
		return _insert_multi(pos, std::distance(first, last), first, last);
	}
	
	iterator insert(const_iterator pos, std::initializer_list<value_type> ilist)
	{
		return _insert_multi(pos, ilist.size(), ilist.begin(), ilist.end());
	}
	
	template<typename... Args>
	iterator emplace(const_iterator pos, Args&&... args)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) <= _size);
		if (pos == end())
		{
			emplace_back(std::forward<Args>(args)...);
			return end()-1;
		}
		return insert(pos, value_type(std::forward<Args>(args)...));
	}
	
	iterator erase(const_iterator pos)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) < _size);
		for (auto it = _iterator(pos); it != end()-1; ++it)
			*it = std::move(*(it+1));
		pop_back();
		return _iterator(pos);
	}
	
	iterator erase(const_iterator first, const_iterator last)
	{
		assert((first-begin()) >= 0 && (last-begin()) <= _size);
		if (first != last)
		{
			auto it = _iterator(first);
			for (; last != end(); ++it,++last)
				*it = std::move(*last);
			_shrink(it - begin());
		}
		return _iterator(first);
	}
	
	void push_back(const value_type& value)
	{
		if (_size == page_capacity)
			throw std::runtime_error("page_vector::push_back(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		_construct(_size, value);
		++_size;
	}

	void push_back(value_type&& value)
	{
		if (_size == page_capacity)
			throw std::runtime_error("page_vector::push_back(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		_construct(_size, std::move(value));
		++_size;
	}

	template<typename... Args>
	void emplace_back(Args&&... args)
	{
		if (_size == page_capacity)
			throw std::runtime_error("page_vector::emplace_back(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		_construct(_size, std::forward<Args>(args)...);
		++_size;
	}
	
	void pop_back()
	{
		assert(_size > 0);
		--_size;
		_destruct(_size);
	}
	
	void resize(size_type count)
	{
		if (count < _size)
			_shrink(count);
		else if (count > _size)
			_grow(count);
	}
	
	void resize(size_type count, const value_type& value)
	{
		if (count < _size)
			_shrink(count);
		else if (count > _size)
			_grow(count, value);
	}

	void swap(page_vector& other)
	{
		std::swap(_page, other._page);
		std::swap(_data, other._data);
		std::swap(_size, other._size);
	}	

private:
	// upgrade a const_iterator into an iterator
	iterator _iterator(const_iterator it) 
	{
		return begin()+(it-begin());
	}
	
	template<typename V>
	iterator _insert_single(const_iterator pos, V&& value)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) <= _size);
		// simplest case: insert at end, in particular when _size == 0
		if (pos == end())
		{
			emplace_back(value);
			return end()-1;
		}
		// case _size >= 1 && begin() <= pos < end():
		emplace_back(std::move(back()));
		// pos <= end()-2
		auto it = end()-2;
		for (; it != pos; --it)
			*it = std::move(*(it-1));
		*it = std::forward<V>(value);
		return it;
	}

	iterator _insert_multi(const_iterator pos, size_type count, const value_type& value)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) <= _size);
		if (_size+count > page_capacity)
			throw std::runtime_error("page_vector::_insert_multi(): capacity exceeded");
		size_type movecount = end()-pos;
		// handle quick path: insert at end
		// but also potentially needs page allocation which invalidates pos
		if (movecount == 0)
		{
			if (_page == nullptr)
				_alloc_page();
			for (; count > 0; --count)
				emplace_back(value);
			return end();
		}
		// movecount > 0: thus no page allocation needed and pos will not be invalidated
		if (movecount <= count)
		{
			// in this case: pos+count >= end, thus:
			// - existing elements [pos,end) are moved to newly constructed places [pos+count,end+count)
			// - [pos,end): copy value into existing elements
			// - [end,pos+count): construct new elements with value
			auto it = _iterator(pos);
			for (; it != end(); ++it)
			{
				_construct(it+count, std::move(*it));
				*it = value;
			}
			for (; it != _iterator(pos)+count; ++it)
				_construct(it, value);
			_size += count;
		} else
		{
			// in this case: pos+count < end, thus:
			// - existing elements [end-count,end) are moved to newly constructed places [end,end+count)
			// - existing elements [pos,end-count) are moved to [pos+count,end), backwards
			// - [pos,pos+count): copy value into existing elements
			auto it = end()-1;
			for (; it != end()-count-1; --it)
				_construct(it+count, std::move(*it));
			for (; it != pos-1; --it)
				*(it+count) = std::move(*it);
			for (++it; it != pos+count; ++it)
				*it = value;
			_size += count;
		}
		return _iterator(pos);
	}

	template<typename InputIt>
	iterator _insert_multi(const_iterator pos, size_type count, InputIt first, InputIt last)
	{
		assert((pos - begin()) >= 0 && (pos-begin()) <= _size);
		if (_size+count > page_capacity)
			throw std::runtime_error("page_vector::_insert_multi(): capacity exceeded");
		size_type movecount = end()-pos;
		// handle quick path: insert at end
		// but also potentially needs page allocation which invalidates pos
		if (movecount == 0)
		{
			if (_page == nullptr)
				_alloc_page();
			for (; count > 0; --count,++first)
				emplace_back(*first);
			return end();
		}
		// movecount > 0: thus no page allocation needed and pos will not be invalidated
		if (movecount <= count)
		{
			// in this case: pos+count >= end, thus:
			// - existing elements [pos,end) are moved to newly constructed places [pos+count,end+count)
			// - [pos,end): copy value into existing elements
			// - [end,pos+count): construct new elements with value
			auto it = _iterator(pos);
			for (; it != end(); ++it,++first)
			{
				_construct(it+count, std::move(*it));
				*it = *first;
			}
			for (; it != _iterator(pos)+count; ++it,++first)
				_construct(it, *first);
			_size += count;
		} else
		{
			// in this case: pos+count < end, thus:
			// - existing elements [end-count,end) are moved to newly constructed places [end,end+count)
			// - existing elements [pos,end-count) are moved to [pos+count,end), backwards
			// - [pos,pos+count): copy value into existing elements
			auto it = end()-1;
			for (; it != end()-count-1; --it)
				_construct(it+count, std::move(*it));
			for (; it != pos-1; --it)
				*(it+count) = std::move(*it);
			for (++it; it != pos+count; ++it,++first)
				*it = *first;
			_size += count;
		}
		return _iterator(pos);
	}

	template<typename InputIt>
	void _assign(size_type count, InputIt first, InputIt last)
	{
		if (count < _size)
			_shrink(count);
		for (auto it = begin(); it != end(); ++it,++first)
			*it = *first;
		for (; first != last; ++first)
			emplace_back(*first);
	}

	// shrink vector
	// must be called with count <= _size
	void _shrink(size_type count)
	{
		assert(count <= _size);
		for (auto p = count; p != _size; ++p)
			_destruct(p);
		_size = count;
	}

	// grow vector
	// must be called with count > _size
	void _grow(size_type count)
	{
		assert(count > _size);
		if (count > page_capacity)
			throw std::runtime_error("page_vector::_grow(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		for (auto p = _size; p != count; ++p)
			_construct(p);
		_size = count;
	}

	// grow vector
	// must be called with count > _size
	void _grow(size_type count, const value_type& value)
	{
		assert(count > _size);
		if (count > page_capacity)
			throw std::runtime_error("page_vector::_grow(): capacity exceeded");
		if (_page == nullptr)
			_alloc_page();
		for (auto p = _size; p != count; ++p)
			_construct(p, value);
		_size = count;
	}
	
	template<typename... Args>
	void _construct(size_type pos, Args&&... args)
	{
		new (_data+pos) value_type(std::forward<Args>(args)...);
	}
	template<typename... Args>
	void _construct(iterator pos, Args&&... args)
	{
		new (pos) value_type(std::forward<Args>(args)...);
	}
	void _destruct(size_type pos)
	{
		(_data+pos)->~value_type();
	}
	
	void _alloc_page()
	{
		if (_page != nullptr)
			return;
		// obtain page
		_page = page_allocator_type::alloc_page();
		if (_page == nullptr)
			throw std::runtime_error("page_vector::_alloc_page(): allocation failed");
		// ensure alignment for value_type
		std::uintptr_t data = std::uintptr_t(_page);
		data += sizeof(value_type);
		data -= data % alignof(value_type);
		// set _data: casts between uintptr_t and value_type* must use intermediate cast to void*
		_data = static_cast<value_type*>( reinterpret_cast<void*>(data) );
		_size = 0;
	}
	
	void _free_page()
	{
		assert(_size == 0);
		if (_page == nullptr)
			return;
		page_allocator_type::free_page(_page);
		_page = nullptr;
		_data = nullptr;
	}

	void*       _page = nullptr;
	value_type* _data = nullptr;
	size_type   _size = 0;
};

template<typename T>
bool operator == (const page_vector<T>& lhs, const page_vector<T>& rhs)
{
	if (lhs.size() != rhs.size())
		return false;
	for (auto it1=lhs.begin(),it2=rhs.begin(),it1end=lhs.end(); it1 != it1end; ++it1,++it2)
		if (*it1 != *it2)
			return false;
	return true;
}

template<typename T>
bool operator <  (const page_vector<T>& lhs, const page_vector<T>& rhs)
{
	return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template<typename T>
bool operator != (const page_vector<T>& lhs, const page_vector<T>& rhs) { return !(lhs == rhs); }
template<typename T>
bool operator >= (const page_vector<T>& lhs, const page_vector<T>& rhs) { return !(lhs < rhs); }
template<typename T>
bool operator >  (const page_vector<T>& lhs, const page_vector<T>& rhs) { return rhs < lhs; }
template<typename T>
bool operator <= (const page_vector<T>& lhs, const page_vector<T>& rhs) { return !(rhs < lhs); }


/* class collection<T>:
	a dynamic storage of elements of type T
	stored in a sequence of page_vector<T>
	API mimics std::vector<T> but elements are not guaranteed to be stored contiguously
	Extended with API to operate on and iterate over page_vectors
	Well suited for parallel algorithms
*/

template<typename T>
class collection
{
public:
	// subcontainer types & capacity
	typedef page_vector<T> subcontainer_type;
	using typename subcontainer_type::value_type;
	using typename subcontainer_type::pointer;
	using typename subcontainer_type::reference;
	using typename subcontainer_type::const_pointer;
	using typename subcontainer_type::const_reference;
	using typename subcontainer_type::size_type;
	using typename subcontainer_type::difference_type;
	typedef typename subcontainer_type::iterator minor_iterator;
	typedef typename subcontainer_type::const_iterator const_minor_iterator;
	static const size_type page_capacity = subcontainer_type::page_capacity;
	
	// container type
	typedef std::vector<subcontainer_type> container_type;
	typedef typename container_type::iterator major_iterator;
	typedef typename container_type::const_iterator const_major_iterator;

	// implements const_iterator & iterator
	// using 3 member variables: collection*, major_iterator & minor_iterator
	// end iterator corresponds to values ( this, data().end(), minor_iterator() )
	template<bool IsConst>
	class Iterator
		: public std::iterator<std::random_access_iterator_tag, typename std::conditional<IsConst, const value_type, value_type>::type, difference_type>
	{
		friend class collection;
	public:
		typedef typename std::conditional<IsConst, const collection, collection>::type collection_type;
		typedef typename std::conditional<IsConst, const_major_iterator, major_iterator>::type major_iterator_type;
		typedef typename std::conditional<IsConst, const_minor_iterator, minor_iterator>::type minor_iterator_type;
		static const size_type page_capacity = collection_type::page_capacity;
		
		explicit Iterator(collection_type& c, major_iterator_type pageit, minor_iterator_type elemit = minor_iterator_type())
			: _ptr(&c), _pageit(pageit), _elemit(elemit)
		{}
		
		explicit Iterator(collection_type& c, difference_type index)
			: _ptr(&c), _pageit(c.data().end()), _elemit()
		{
			_from_index(index);
		}
		
		explicit Iterator(collection_type& c, difference_type major_index, difference_type minor_index)
			: _ptr(&c), _pageit(c.data().begin()+major_index), _elemit()
		{
			if (_pageit != c.data().end())
				_elemit = _pageit->begin()+minor_index;
		}
		
		Iterator(const Iterator&) = default;
		Iterator(Iterator&&) = default;
		
		Iterator& operator= (const Iterator&) = default;
		Iterator& operator= (Iterator&&) = default;
		
		// support conversion from iterator to const_iterator
		operator Iterator<true>() const
		{
			return Iterator<true>(_ptr, _pageit, _elemit);
		}
		
		template<bool IsConst2>
		bool operator==(const Iterator<IsConst2>& other) const
		{
			return _ptr == other._ptr && _pageit == other._pageit && _elemit == other._elemit;
		}
		
		template<bool IsConst2>
		bool operator!=(const Iterator<IsConst2>& other) const
		{
			return _ptr != other._ptr || _pageit != other._pageit || _elemit != other._elemit;
		}

		template<bool IsConst2>
		bool operator<(const Iterator<IsConst2>& other) const
		{
			if (_ptr != other._ptr)
				return _ptr < other._ptr;
			return _to_index() < other._to_index();
		}
		template<bool IsConst2>
		bool operator>=(const Iterator<IsConst2>& other) const { return !(*this < other); }
		template<bool IsConst2>
		bool operator>(const Iterator<IsConst2>& other) const { return other < *this; }
		template<bool IsConst2>
		bool operator<=(const Iterator<IsConst2>& other) const { return !(other < *this); }


		Iterator& operator++()
		{
			_increment();
			return *this;
		}
		
		Iterator operator++(int)
		{
			Iterator tmp = *this;
			_increment();
			return tmp;
		}

		Iterator& operator--()
		{
			_decrement();
			return *this;
		}

		Iterator operator--(int)
		{
			Iterator tmp = *this;
			_decrement();
			return tmp;
		}
		
		Iterator& operator+= (difference_type n)
		{
			_from_index(_to_index() + n);
			return *this;
		}
		
		Iterator& operator-= (difference_type n)
		{
			_from_index(_to_index() - n);
			return *this;
		}
		
		Iterator operator+ (difference_type n) const
		{
			return Iterator(*_ptr, _to_index() + n);
		}

		Iterator operator- (difference_type n) const
		{
			return Iterator(*_ptr, _to_index() - n);
		}

		template<bool IsConst2>
		difference_type operator- (const Iterator<IsConst2>& other) const
		{
			return _to_index() - other._to_index();
		}
		
		reference operator[](difference_type n) const
		{
			return (*_ptr)[_to_index() + n];
		}

	private:
		difference_type _to_index()
		{
			assert(_ptr != nullptr && _pageit != _ptr->data.end());
			if (_pageit == _ptr->data().end())
				return _ptr->size();
			return (_elemit - _pageit->begin()) + ((_pageit - _ptr->data().begin()) * difference_type(page_capacity));
		}

		void _from_index(difference_type index)
		{
			assert(_ptr != nullptr);
			if (index < 0 || index > _size)
				throw std::out_of_range("collection::Iterator::_from_index: out of range");
			if (index == _size)
			{
				_pageit = _ptr->data().end();
				_elemit = minor_iterator_type();
				return;
			}
			_pageit = _ptr->data().begin() + (index / page_capacity);
			_elemit = _pageit->begin() + (index % page_capacity);
		}
		
		void _increment()
		{
			if (++_elemit == _pageit->end())
			{
				if (++_pageit != _ptr->data().end())
					_elemit = _pageit->begin();
				else
					_elemit = minor_iterator_type();
			}
		}

		void _decrement()
		{
			if (_elemit == _pageit->begin())
			{
				--_pageit;
				_elemit = _pageit->end();
			}
			--_elemit;
		}
		
		collection_type* _ptr = nullptr;
		major_iterator_type _pageit;
		minor_iterator_type _elemit;
	};
	typedef Iterator<false> iterator;
	typedef Iterator<true> const_iterator;
	

	// empty constructor
	collection() = default;

	// default copy & move constructors
	collection(const collection&) = default;
	collection(collection&&) = default;
	
	// other constructions
	explicit collection(size_type count)
	{
		if (count > 0)
			_resize(count);
	}
	
	collection(size_type count, const value_type& v)
	{
		if (count > 0)
			_resize(count, v);
	}
	
	collection(std::initializer_list<value_type> init)
	{
		_assign(init.size(), init.begin(), init.end());
	}
	
	template<typename InputIt>
	collection(InputIt first, InputIt last)
	{
		_assign(std::distance(first,last), first, last);
	}

	// default copy & move assignment
	collection& operator= (const collection&) = default;
	collection& operator= (collection&&) = default;
	
	// other assignment
	collection& operator= (std::initializer_list<value_type> init)
	{
		_assign(init.size(), init.begin(), init.end());
		return *this;
	}
	
	void assign(size_type count, const value_type& value)
	{
		// determine the number of existing elements into which value must be copied
		size_type copycount = std::min<size_type>(count, _size);
		// resize to new size
		if (count != _size)
			_resize(count, value);
		// copy value into (remaining) pre-existing elements
		if (copycount == 0)
			return;
		for (auto& page : _data)
		{
			for (auto& v : page)
			{
				v = value;
				if (--copycount == 0)
					return;
			}
		}
	}
	
	template<typename InputIt>
	void assign(InputIt first, InputIt last)
	{
		_assign(std::distance(first,last), first, last);
	}
	
	void assign(std::initializer_list<value_type> init)
	{
		_assign(init.size(), init.begin(), init.end());
	}

	iterator begin() { return iterator(*this, 0); }
	iterator end()   { return iterator(*this, _data.end()); }
	const_iterator begin() const { return const_iterator(*this, 0); }
	const_iterator end()   const { return const_iterator(*this, _data.end()); }
	const_iterator cbegin() const { return const_iterator(*this, 0); }
	const_iterator cend()   const { return const_iterator(*this, _data.end()); }
		
	reference at(size_type pos)
	{
		if (pos >= _size)
			throw std::out_of_range("collection::at(): out of range");
		return _data[pos / page_capacity][pos % page_capacity];
	}
	
	const_reference at(size_type pos) const
	{
		if (pos >= _size)
			throw std::out_of_range("collection::at(): out of range");
		return _data[pos / page_capacity][pos % page_capacity];
	}
	
	reference operator[](size_type pos)
	{
		assert(pos < _size);
		return _data[pos / page_capacity][pos % page_capacity];
	}
	
	const_reference operator[](size_type pos) const
	{
		assert(pos < _size);
		return _data[pos / page_capacity][pos % page_capacity];
	}
	
	reference front()
	{
		assert(_size > 0);
		return _data.front().front();
	}
	
	const_reference front() const
	{
		assert(_size > 0);
		return _data.front().front();
	}

	reference back()
	{
		assert(_size > 0);
		return _data.back().back();
	}
	
	const_reference back() const
	{
		assert(_size > 0);
		return _data.back().back();
	}


	
	bool empty() const { return _size == 0; }
	size_type size() const { return _size; }
	size_type max_size() const { return _data.max_size(); }

	void clear()
	{
		_data.clear();
		_size = 0;
	}
	

	void swap(collection& other)
	{
		std::swap(_size, other._size);
		std::swap(_data, other._data);
	}

	void resize(size_type newsize)
	{
		if (newsize != _size)
			_resize(newsize);
	}

	void resize(size_type newsize, const value_type& value)
	{
		if (newsize != _size)
			_resize(newsize, value);
	}

	iterator insert(const_iterator pos, const value_type& value)
	{
		return _insert_value(pos, value);
	}

	iterator insert(const_iterator pos, value_type&& value)
	{
		return _insert_value(pos, std::move(value));
	}
	
	iterator insert(const_iterator pos, size_type count, const value_type& value)
	{
		if (pos._ptr != this)
			throw std::runtime_error("collection::insert(): invalid input iterator");
		if (count == 0)
			return _upgrade(pos);
		// insert at end
		if (pos._pageit == _data.end())
		{
			size_type pageidx = _data.size()-1, elemidx = _data.back().size();
			if (elemidx == page_capacity)
			{
				++pageidx;
				elemidx = 0;
			}
			_resize( _size + count, value );
			return iterator(this, pageidx, elemidx);
		}
		// insert without shifting across pages
		iterator it(_upgrade(pos));
		if (it._pageit->size()+count <= page_capacity)
		{
			it._elemit = it._pageit->insert(pos._elemit, count, value);
			_size += count;
			return it;
		}
		// handle insert with shifting across pages
		_shift(it, count, value);
		return it;
	}
	
	template<typename InputIt>
	iterator insert(const_iterator pos, InputIt first, InputIt last)
	{
		return _insert_range(pos, std::distance(first,last), first, last);
	}
	
	iterator insert(const_iterator pos, std::initializer_list<value_type> init)
	{
		return _insert_range(pos, init.size(), init.begin(), init.end());
	}
	
	template<typename... Args>
	iterator emplace(const_iterator pos, Args&&... args)
	{
		return _insert_value(pos, std::forward<Args>(args)...);
	}
	
	iterator erase(const_iterator pos)
	{
		if (pos._ptr != this || pos._pageit == _data.end())
			throw std::runtime_error("collection::erase(): invalid input iterator");
		iterator _pos(_upgrade(pos));
		iterator it(_pos), it2(_pos);
		for (++it2; it2._pageit != _data.end(); ++it2)
		{
			*it = std::move(*it2);
			it = it2;
		}
		_resize(_size - 1);
		return _pos;
	}
	
	iterator erase(const_iterator first, const_iterator last)
	{
		if (first._ptr != this || last._ptr != this)
			throw std::runtime_error("collection::erase(): invalid input iterator");
		iterator _first(_upgrade(first)), _last(_upgrade(last));
		size_type count = std::distance(first, last);
		if (_first == _last)
			return _first;
		// bad case: every element hereafter needs to be moved
		if ((count % page_capacity) != 0)
		{
			for (iterator it = _first; _last._pageit != _data.end(); ++it,++_last)
				*it = std::move( *_last );
			_resize( _size - count );
			return _first;
		}
		// good case: after moving elements keep the same index within a page
		// thus only need to move remaining elements from last page to first page in question,
		// do the necessary moves:
		auto it1 = _first.elemit, it2 = _last.elemit;
		for (; it2 != _last.pageit->end(); ++it1,++it2)
			*it1 = std::move( *it2 );
		assert(it1 == _first.pageit->end());
		// remove pages:
		_data.erase(_first._pageit+1, _last._pageit+1);
		_size -= count;
		return _first;
	}

	void pop_back()
	{
		_data.back().pop_back();
		if (_data.back().empty())
			_data.pop_back();
		--_size;
	}

	void push_back(const value_type& value)
	{
		if (_data.back().size() == page_capacity)
			_data.emplace_back();
		_data.back().push_back(value);
		++_size;
	}

	void push_back(value_type&& value)
	{
		if (_data.back().size() == page_capacity)
			_data.emplace_back();
		_data.back().push_back(std::move(value));
		++_size;
	}

	template<typename... Args>
	void emplace_back(Args&&... args)
	{
		if (_data.back().size() == page_capacity)
			_data.emplace_back();
		_data.back().emplace_back(std::forward<Args>(args)...);
		++_size;
	}

	// direct access to underlying data structure
	container_type& data() { return _data; }
	const container_type& data() const { return _data; }
	
	// verify internal invariants are met
	//  as we allow low level manipulation of collection
	//  this function is public to enable debugging
	void check()
	{
		// check:
		// 1) _size matches sum of page sizes
		size_type s = 0;
		for (auto& p : _data)
			s += p.size();
		if (s != _size)
			throw std::runtime_error("collection::_check(): internal size mismatch");
		// 2) every page, except last one, is at capacity
		for (size_type i = 0; i+1 < _data.size(); ++i)
			if (_data[i].size() != page_capacity)
				throw std::runtime_error("collection::_check(): internal allocation error");
		// 3) last page (if any) is not empty
		if (!_data.empty() && _data.back().empty())
			throw std::runtime_error("collection::_check(): last page empty");
	}

	// only call after manually manipulating the underlying pages to satisfy internal variants
	// reorder elements such that all pages are at capacity, except the last one, in the fastest possible way
	// and update _size accordingly
	void unstable_sanitize()
	{
		_size = 0;
		if (_data.empty())
			return;
		// sort from big to small to reduce the amount of moving necessary
		std::sort(_data.begin(), _data.end(), [](const subcontainer_type& lhs, const subcontainer_type& rhs) { return lhs.size() > rhs.size(); } );
		// move data loop: move data from _data.back() to _data[pageid] (when these are different pages)
		size_type pageid = 0;
		while (true)
		{
			// remove empty pages at end
			while (!_data.empty() && _data.back().empty())
				_data.pop_back();
			// skip any full pages
			while (pageid < _data.size() && _data[pageid].size() == page_capacity)
				++pageid;
			// check we are moving data between two different valid pages
			if (pageid+1 >= _data.size())
				break;
			// move data
			size_type count = std::min<size_type>(page_capacity - _data[pageid].size(), _data.back().size() );
			auto it = _data.back().end() - count;
			for (size_type i = count; i != 0; --i,++it)
				_data[pageid].emplace_back( std::move( *it ) );
			_data.back().resize(_data.back().size() - count);
		}
		if (_data.empty())
			return;
		_size = (_data.size()-1) * page_capacity + _data.back().size();
	}

	// only call after manually manipulating the underlying pages to satisfy internal variants
	// ensure all pages are at capacity, except the last one, while maintaining element order
	// and update _size accordingly
	void stable_sanitize()
	{
		container_type pages;
		for (auto it = _data.begin(); it != _data.end(); ++it)
		{
			if (it->size() > 0)
				pages.emplace_back(std::move(*it));
		}
		_data.clear();
		_size = 0;
		for (auto it = pages.begin(); it != pages.end(); ++it)
			append( std::move( *it ) );
	}
	
	// append iterator range to end
	template<typename InputIt>
	void append(InputIt first, InputIt last)
	{
		for (; first != last; ++first)
			emplace_back(*first);
	}

	// append elements from a page to end
	void append(const subcontainer_type& page)
	{
		append(page.begin(), page.end());
	}
	
	// append elements from a page to end, but we're allowed to move the page
	void append(subcontainer_type&& page)
	{
		if (page.empty())
		{
			page.free();
			return;
		}
		// if append happens at page border we can simply move-append page to _data
		if (_data.empty() || _data.back().size() == page_capacity)
		{
			_size += page.size();
			_data.emplace_back( std::move(page) );
			page.free();
			return;
		}
		// otherwise we first need to fill current last page to capacity
		auto it = page.begin(), itend = page.end();
		for (; it != itend && _data.back().size() < page_capacity; ++it)
		{
			_data.back().emplace_back( std::move(*it) );
			++_size;
		}
		// if any, shift the remaining elements to the begin of page and move-append page to _data
		if (it != itend)
		{
			auto it2 = page.begin();
			for (; it != itend; ++it,++it2)
				*it2 = std::move( *it );
			page.erase(it2, page.end());
			_size += page.size();
			_data.emplace_back( std::move(page) );
		}
		page.free();
	}

	// append other collection to end	
	void append(const collection& other)
	{
		append(other.begin(), other.end());
	}

	// append other collection to end, but we're allowed to move all its pages
	void append(collection&& other)
	{
		for (auto& p : other._data())
			append( std::move(p) );
		other._data.clear();
		other._size = 0;
	}
	
	// add elements from collection to this collection, without maintaining element order
	void merge_unstable(collection&& other)
	{
		// move all pages from both collections together
		for (auto& p : other._data)
			_data.emplace_back( std::move(p) );
		other._data.clear();
		other._size = 0;
		// and perform unstable sanitization
		unstable_sanitize();
	}

private:
	template<typename InputIt>
	iterator _insert_range(const_iterator pos, size_type count, InputIt first, InputIt last)
	{
		if (pos._ptr != this)
			throw std::runtime_error("collection::insert(): invalid input iterator");
		// fast path: insert at end
		if (pos._pageit == _data.end())
		{
			for (; first != last; ++first)
				emplace_back(*first);
			return iterator(this, _data.begin());
		}
	}
	
	// implements both insert(const_iterator, const value_type&) and insert(const_iterator, value_type&&)
	template<typename... Args>
	iterator _insert_value(const_iterator pos, Args&&... args)
	{
		if (pos._ptr != this)
			throw std::runtime_error("collection::insert(): invalid input iterator");
		// fast path: insert at end
		if (pos._pageit == _data.end())
		{
			emplace_back(std::forward<Args>(args)...);
			return iterator(this, _data.begin()+(_data.size()-1), _data.back().begin()+(_data.back().size()-1));
		}
		iterator it(_upgrade(pos));
		// 2nd fast path: insert in last page
		if (pos._pageit->size() < page_capacity)
		{
			it._elemit = it._pageit->emplace(pos._elemit, std::forward<Args>(args)...);
			++_size;
			return it;
		}
		// slow path: elements need to be shifted across pages
		_shift(it, 1, std::forward<Args>(args)...);
		return it;
	}

	iterator _upgrade(const const_iterator& it)
	{
		if (it._pageit == _data.end())
			return iterator(this, _data.end());
		return iterator(this, it._pageit - _data.begin(), it._elemit - it._pageit->begin() );
	}
	
	template<typename InputIt>
	void _assign(size_type count, InputIt first, InputIt last)
	{
		if (count < _size)
			_resize(count);
		for (auto& page : _data)
		{
			for (auto& v : page)
			{
				v = *first;
				++first;
			}
		}
		for (; first != last; ++first)
			emplace_back(*first);
	}

	void _resize(size_type newsize)
	{
		size_type oldlastpage = _data.size()==0 ? 0 : _data.size() - 1;
		size_type pages = (newsize + page_capacity - 1) / page_capacity;
		_data.resize(pages);
		for (size_type i = oldlastpage; i+1 < pages; ++i)
			_data[i].resize( page_capacity );
		if (newsize > 0)
			_data.back().resize( newsize - (pages-1)*page_capacity );
		_size = newsize;
	}

	void _resize(size_type newsize, const value_type& v)
	{
		size_type oldlastpage = _data.size()==0 ? 0 : _data.size() - 1;
		size_type pages = (newsize + page_capacity - 1) / page_capacity;
		_data.resize(pages);
		for (size_type i = oldlastpage; i+1 < pages; ++i)
			_data[i].resize( page_capacity, v );
		if (newsize > 0)
			_data.back().resize( newsize - (pages-1)*page_capacity, v );
		_size = newsize;
	}
	
	container_type _data;
	size_type _size = 0;
};

template<typename T>
bool operator == (const collection<T>& lhs, const collection<T>& rhs)
{
	if (lhs.size() != rhs.size())
		return false;
	return lhs.data() == rhs.data();
}

template<typename T>
bool operator <  (const collection<T>& lhs, const collection<T>& rhs)
{
	return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template<typename T>
bool operator != (const collection<T>& lhs, const collection<T>& rhs) { return !(lhs == rhs); }
template<typename T>
bool operator >= (const collection<T>& lhs, const collection<T>& rhs) { return !(lhs < rhs); }
template<typename T>
bool operator >  (const collection<T>& lhs, const collection<T>& rhs) { return rhs < lhs; }
template<typename T>
bool operator <= (const collection<T>& lhs, const collection<T>& rhs) { return !(rhs < lhs); }


MCCL_END_NAMESPACE

#endif
