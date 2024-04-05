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
	mccl/core/collection.hpp \
	\
	mccl/algorithm/decoding.hpp \
	mccl/algorithm/decoding.cpp \
	mccl/algorithm/isdgeneric.hpp \
	mccl/algorithm/isdgeneric.cpp \
	mccl/algorithm/prange.hpp \
	mccl/algorithm/prange.cpp \
	mccl/algorithm/lee_brickell.hpp \
	mccl/algorithm/lee_brickell.cpp \
	mccl/algorithm/stern_dumer.hpp \
	mccl/algorithm/stern_dumer.cpp \
	mccl/algorithm/sieving.hpp \
	mccl/algorithm/sieving.cpp \
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

TESTS=          tests/test_compile tests/test_unordered_multimap tests/test_matrix tests/test_parser tests/test_prange tests/test_dumer tests/test_sieving tests/test_collection

check_PROGRAMS= tests/test_compile tests/test_unordered_multimap tests/test_matrix tests/test_parser tests/test_prange tests/test_dumer tests/test_sieving tests/test_collection

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

tests_test_dumer_SOURCES= tests/test_dumer.cpp
tests_test_dumer_LDADD  = libmccl.la

tests_test_sieving_SOURCES= tests/test_sieving.cpp
tests_test_sieving_LDADD  = libmccl.la

tests_test_collection_SOURCES= tests/test_collection.cpp
tests_test_collection_LDADD  = libmccl.la

CLANGFORMAT ?= clang-format
.PHONY: check-style
check-style:
	-bash tools/update_m4.sh
	-bash tools/update_contrib.sh
	$(CLANGFORMAT) -i --style=file `find -name "*.{cpp,hpp,h,inl}"`


###################### M4RI conditional support ###############

if HAVE_M4RI

libmccl_la_SOURCES += mccl/core/matrix_m4ri.cpp

TESTS += tests/test_m4ri
check_PROGRAMS += tests/test_m4ri

tests_test_m4ri_SOURCES= tests/test_m4ri.cpp
tests_test_m4ri_LDADD  = libmccl.la

endif

###############################################################


#################### CUDA conditional support #################

if HAVE_CUDA

TESTS += tests/test_cuda
check_PROGRAMS += tests/test_cuda

tests_test_cuda_SOURCES= tests/test_cuda.cu tests/test_cuda_dummy.cpp
tests_test_cuda_LDADD  = libmccl.la $(CUDA_LIBS)

endif

NVCCFLAGS=-ccbin $(CXX) -O2 -v --maxrregcount=64 --ptxas-options=-v -Xcompiler -mcmodel=medium

CUDA_SMS=70 75
GENCODE_FLAGS=
$(foreach sm,$(CUDA_SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
HIGHEST_SM:=$(lastword $(sort $(CUDA_SMS)))
GENCODE_FLAGS+= -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

.cu.o:
	$(NVCC) $(GENCODE_FLAGS) $(NVCCFLAGS) $(CUDA_CFLAGS) $(addprefix -Xcompiler ,$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) -I $(top_srcdir) -Wno-pedantic ) -o $@ -c $<

###############################################################
		
