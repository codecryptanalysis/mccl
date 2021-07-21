#ifndef MCCL_TESTS_TEST_UTILS_HPP
#define MCCL_TESTS_TEST_UTILS_HPP

#include <iostream>

#define LOG_CERR(s) std::cerr << s << std::endl;
#define LOG_COUT(s) std::cout << s << std::endl;


template<typename T> struct compile_test_helper { typedef void type; };
template<typename T> using compile_test_helper_t = typename compile_test_helper<T>::type;

#define DEFINE_COMPILE_TEST(testname,var,type,testcode) \
template<typename var = type, typename=void> struct testname : std::false_type {}; \
template<typename var> \
struct testname <var, compile_test_helper_t<decltype( testcode )>>: std::true_type {};
#define COMPILE_TEST_VALUE(testname) (testname<int>::value)

// lambda function that returns 0 if testname produces okvalue, otherwise -1 and outputs errmsg to cerr
#define CHECK_COMPILE_TEST(testname, okvalue, errmsg) \
    ( [](){ if (testname<>::value != okvalue) { LOG_CERR(errmsg); return -1; } return 0; }() )

#endif
