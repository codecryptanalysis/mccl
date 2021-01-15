AC_PREREQ([2.68])
# Important: DBLPBIBTEX version needs to match with: ./version src/core.hpp
AC_INIT([mccl], [0.1], [marc@marc-stevens.nl],[mccl],[https://github.com/cr-marcstevens/mccl])
AC_CONFIG_SRCDIR([README.md])
AC_CONFIG_HEADERS([mccl/config.h])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign subdir-objects])


DEFAULT_CXXFLAGS="-O2"
AS_IF([test "x$CXXFLAGS" = "x"],
	[CXXFLAGS="$DEFAULT_CXXFLAGS"]
	[usedefaultcxxflags=yes],
	[usedefaultcxxflags=no])

LT_INIT

AC_LANG([C++])
AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_LN_S
AC_PROG_MKDIR_P
AC_PROG_CXX
AX_CXX_COMPILE_STDCXX([11],[noext],[mandatory])

AC_TYPE_SIZE_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T

AS_IF([test "x$cross_compiling" != "xyes" && test "x$usedefaultcxxflags" = "xyes" ],
	[AX_CHECK_COMPILE_FLAG([-march=native], [CXXFLAGS="$CXXFLAGS -march=native"], [])])


AX_PTHREAD()

CHECK_ATOMIC()

LIBS="$PTHREAD_LIBS $LIBS"
CPPFLAGS="$CPPFLAGS"
CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS"

AC_CONFIG_FILES([
Makefile
])

AC_OUTPUT
