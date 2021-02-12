#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define ERRNO_DETAIL(err) getErrnoMsg(err)

const char *getErrnoMsg(unsigned int err_idx);

// 通用错误
#define SUCCESS (0)
#define ERR_COD (1)


// 1.消息信息宏==========

// 1.1.错误信息输出, 利用了可变参数宏
#define ERR_MSG(...) \
fprintf(stderr, "%s(), %s, line %d, ", __FUNCTION__, __FILE__, __LINE__); \
fprintf(stderr, __VA_ARGS__);


// 1.2.警告信息输出, 利用了可变参数宏
#define WARN_MSG(...) \
fprintf(stderr, "%s(), %s, line %d, ", __FUNCTION__, __FILE__, __LINE__); \
fprintf(stderr, __VA_ARGS__);

// 2.错误检查宏==========

// 2.1.支持自定义错误码

// 支持附加错误码的CHECK_NULL，返回自定义错误码
#define CHK_NIL_COD(val, code) \
do { \
    if ((val) == NULL) { \
        ERR_MSG("Null ptr var %s is found, error.\n", #val); \
        return code; \
    } \
} while (0)

// 支持附加错误码的CHECK_NOT_NULL，返回自定义错误码
#define CHK_NOT_NIL_COD(val, code) \
do { \
    if ((!(val)) == 0) { \
        ERR_MSG("Not Null ptr var: %s found, error.\n", #val); \
        return code; \
    } \
} while (0)

// 带返回错误码的CHECK_ERROR
#define CHK_ERR_COD(val, code) \
do { \
    int _tmp_val_res = (val); \
    if (_tmp_val_res != 0) { \
        ERR_MSG("catch err: %d from %s, throw error: %d\n", _tmp_val_res, #val, code); \
        return code; \
    } \
} while (0)

// 2.2.采用默认错误码

// 返回默认错误码
#define CHK_NIL(val) CHK_NIL_COD(val, ERR_COD)

// 支持附加错误码的CHECK_NOT_NULL，返回自定义错误码
#define CHK_NOT_NIL(val) CHK_NOT_NIL_COD(val, ERR_COD)

// CHECK_ERROR, 检查函数返回值是否为0, 相当于之前的CATCH_ERR, 2019.04.24 
#define CHK_ERR(val) \
do { \
    int _tmp_val_res = (val); \
    if (_tmp_val_res != 0) { \
        ERR_MSG("catch err: %d from %s, throw error: %d\n", _tmp_val_res, #val, _tmp_val_res); \
        return _tmp_val_res; \
    } \
} while (0)


