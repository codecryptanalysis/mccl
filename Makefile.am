EXTRA_DIST = README.md LICENSE
AMLOCAL_AMFLAGS = -I m4

#lib_libmccl_la_SOURCES=\
#	mccl/contrib/string_algo.hpp \
#	mccl/contrib/thread_pool.hpp \
#	mccl/contrib/parallel_algorithms.hpp \
#	mccl/contrib/json.hpp \
#	mccl/contrib/program_options.hpp \
#	mccl/core/birthday.hpp \
#	mccl/core/matrix.hpp \
#	mccl/algorithm/decoding.hpp \
#	mccl/algorithm/dumer.hpp \
#	mccl/algorithm/dumer.cpp

TESTS = tests/test_compile

check_PROGRAMS= tests/test_compile

tests_test_compile_SOURCES=tests/test_compile.cpp

		
CLANGFORMAT ?= clang-format
.PHONY: check-style
check-style:
	-bash tools/update_m4.sh
	-bash tools/update_contrib.sh
	$(CLANGFORMAT) -i --style=file `find -name "*.{cpp,hpp,h}"`
