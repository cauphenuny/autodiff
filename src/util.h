#pragma once
#ifdef DEBUG
#include <iostream>
#    define debug(x)    std::cerr << #x << " = " << (x) << " "
#    define debugln(x)  std::cerr << #x << " = " << (x) << std::endl
#    define debugf(...) fprintf(stderr, __VA_ARGS__)
#    if defined(__clang__) || defined(__GNUC__)
#        define debugv(fmt, ...)                                    \
            fprintf(stderr, "%s/%d: " fmt "\n", __func__, __LINE__, \
                    ##__VA_ARGS__)
#    else
#        define debugv(...)
#    endif
#else
#    define debug(x)
#    define debugln(x)
#define debugf(...)
#    define debugv(...)
#endif