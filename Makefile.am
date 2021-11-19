EXTRA_DIST = README.md LICENSE
ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libmccl.la
bin_PROGRAMS = bin/isdsolver

libmccl_la_LDFLAGS= -version-info 1:0:1 -no-undefined
libmccl_la_SOURCES=\
	mccl/contrib/BigInt.hpp \
	mccl/contrib/string_algo.hpp \
	mccl/contrib/thread_pool.hpp \
	mccl/contrib/parallel_algorithms.hpp \
	mccl/contrib/json.hpp \
	mccl/contrib/program_options.hpp \
	\
	mccl/config/config.h \
	mccl/config/config.hpp \
	\
	mccl/core/matrix.hpp \
	mccl/core/matrix_base.hpp \
	mccl/core/matrix_ops.hpp \
	mccl/core/matrix_ops.inl \
	mccl/core/matrix_ops.cpp \
	mccl/core/matrix_m4ri.hpp \
	mccl/core/matrix_isdform.hpp \
	\
	mccl/algorithm/decoding.hpp \
	mccl/algorithm/decoding.cpp \
	mccl/algorithm/isdgeneric.hpp \
	mccl/algorithm/isdgeneric.cpp \
	mccl/algorithm/prange.hpp \
	mccl/algorithm/prange.cpp \
	mccl/algorithm/lee_brickell.hpp \
	mccl/algorithm/lee_brickell.cpp \
	mccl/algorithm/dumer.hpp \
	\
	mccl/tools/parser.hpp \
	mccl/tools/parser.cpp \
	mccl/tools/generator.hpp \
	mccl/tools/generator.cpp \
	mccl/tools/statistics.hpp \
	mccl/tools/utils.hpp \
	mccl/tools/unordered_multimap.hpp \
	mccl/tools/unordered_multimap.cpp \
	mccl/tools/bitfield.hpp \
	mccl/tools/enumerate.hpp


bin_isdsolver_SOURCES= src/isdsolver.cpp
bin_isdsolver_LDADD  = libmccl.la

TESTS=          tests/test_compile tests/test_unordered_multimap tests/test_matrix tests/test_parser tests/test_prange

check_PROGRAMS= tests/test_compile tests/test_unordered_multimap tests/test_matrix tests/test_parser tests/test_prange

tests_test_compile_SOURCES= tests/test_compile.cpp
tests_test_compile_LDADD  = libmccl.la

tests_test_unordered_multimap_SOURCES= tests/test_unordered_multimap.cpp
tests_test_unordered_multimap_LDADD  = libmccl.la

tests_test_matrix_SOURCES= tests/test_matrix.cpp
tests_test_matrix_LDADD  = libmccl.la

tests_test_parser_SOURCES= tests/test_parser.cpp
tests_test_parser_LDADD  = libmccl.la

tests_test_prange_SOURCES= tests/test_prange.cpp
tests_test_prange_LDADD  = libmccl.la


if HAVE_M4RI

libmccl_la_SOURCES += mccl/core/matrix_m4ri.cpp

TESTS += tests/test_m4ri
check_PROGRAMS += tests/test_m4ri

tests_test_m4ri_SOURCES= tests/test_m4ri.cpp
tests_test_m4ri_LDADD  = libmccl.la

endif

		
CLANGFORMAT ?= clang-format
.PHONY: check-style
check-style:
	-bash tools/update_m4.sh
	-bash tools/update_contrib.sh
	$(CLANGFORMAT) -i --style=file `find -name "*.{cpp,hpp,h,inl}"`
