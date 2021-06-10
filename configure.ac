AC_PREREQ([2.68])
# Important: DBLPBIBTEX version needs to match with: ./version src/core.hpp
AC_INIT([mccl], [0.1], [marc@marc-stevens.nl],[mccl],[https://github.com/cr-marcstevens/mccl])
AC_CONFIG_SRCDIR([README.md])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign subdir-objects])


DEFAULT_CXXFLAGS="-O2"
AS_IF([test "x$CXXFLAGS" = "x"],
    [CXXFLAGS="$DEFAULT_CXXFLAGS"]
    [usedefaultcxxflags=yes]
    ,
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


AC_ARG_WITH([m4ri],
    AS_HELP_STRING([--with-m4ri@<:@=VALUE@:>@], [Optional prefix for M4RI library]),
    [m4riprefix=$withval]
    [usem4ri=1]
    ,
    [usem4ri=0]
    )
AS_IF([test "x$m4riprefix" = "xno"],[usem4ri=0])
AS_IF([test "x$usem4ri" = "x1"],
    AC_MSG_NOTICE([M4RI support enabled])
    AC_DEFINE([HAVE_M4RI],[1],[whether to compile with M4RI support])
    AS_IF([test "x$m4riprefix" != "x" ],
        [M4RI_CPPFLAGS="-I${m4riprefix}/include"]
        [M4RI_LIBS="-L${m4prefix}/lib -lm4ri"]
        ,
        [M4RI_CPPFLAGS=""]
        [M4RI_LIBS="-lm4ri"]
        )
    AX_M4RI_CFLAGS()
    M4RI_CFLAGS=$ax_cv_m4ri_cflags
    ,
    AC_MSG_NOTICE([M4RI support disabled])
    [M4RI_CPPFLAGS=""]
    [M4RI_LIBS=""]
    [M4RI_CFLAGS=""]
    )
AM_CONDITIONAL([HAVE_M4RI],[test "x$usem4ri" = "x1"])

CPPFLAGS="$CPPFLAGS $M4RI_CPPFLAGS"
CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS $M4RI_CFLAGS -Wall -Wextra -Wpedantic"
LIBS="$M4RI_LIBS $PTHREAD_LIBS $LIBS"

AC_CONFIG_HEADERS([config.h])
AX_PREFIX_CONFIG_H([mccl/config/config.h])

AC_CONFIG_FILES([
    Makefile
])

AC_OUTPUT
