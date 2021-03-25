EXTRA_DIST = README.md LICENSE
ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libmccl.la

libmccl_la_LDFLAGS= -version-info 1:0:1 -no-undefined
libmccl_la_SOURCES=\
	mccl/contrib/string_algo.hpp \
	mccl/contrib/thread_pool.hpp \
	mccl/contrib/parallel_algorithms.hpp \
	mccl/contrib/json.hpp \
	mccl/contrib/program_options.hpp \
	\
	mccl/config/config.h \
	mccl/config/config.hpp \
	\
	mccl/core/matrix_detail.hpp \
	mccl/core/matrix_detail.inl \
	mccl/core/matrix.hpp \
	mccl/core/matrix_m4ri.hpp \
	mccl/core/matrix_m4ri.cpp \
	mccl/core/matrix_permute.hpp

TESTS= tests/test_compile tests/test_matrix tests/test_m4ri tests/test_parser tests/test_permute

check_PROGRAMS= tests/test_compile tests/test_matrix tests/test_m4ri tests/test_parser tests/test_permute

tests_test_compile_SOURCES= tests/test_compile.cpp
tests_test_compile_LDADD  = libmccl.la

tests_test_matrix_SOURCES= tests/test_matrix.cpp
tests_test_matrix_LDADD  = libmccl.la

tests_test_m4ri_SOURCES= tests/test_m4ri.cpp
tests_test_m4ri_LDADD  = libmccl.la

tests_test_parser_SOURCES= tests/test_parser.cpp
tests_test_parser_LDADD  = libmccl.la

tests_test_permute_SOURCES= tests/test_permute.cpp
tests_test_permute_LDADD  = libmccl.la
		
CLANGFORMAT ?= clang-format
.PHONY: check-style
check-style:
	-bash tools/update_m4.sh
	-bash tools/update_contrib.sh
	$(CLANGFORMAT) -i --style=file `find -name "*.{cpp,hpp,h}"`
