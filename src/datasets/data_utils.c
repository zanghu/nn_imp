#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/time.h>

#include "debug_macros.h"
#include "data_utils.h"

static int getElemSize(unsigned int *size, const char *dtype)
{
    CHK_NIL(size);
    CHK_NIL(dtype);

    if (strcasecmp(dtype, "float32") == 0) {
        *size = sizeof(float);
    } else if(strcasecmp(dtype, "float64") == 0) {
        *size = sizeof(double);
    } else if(strcasecmp(dtype, "int32") == 0) {
        *size = sizeof(int);
    } else if(strcasecmp(dtype, "int64") == 0) {
        *size = sizeof(long long);
    } else if(strcasecmp(dtype, "uint8") == 0) {
        *size = sizeof(unsigned char);
    } else {
        ERR_MSG("Unknow dtype: %s, error.\n", dtype);
        return ERR_COD;
    }

    return SUCCESS;
}

int transformOnehot(void **onehot, const char *dtype_onehot, void *orin, const char *dtype_orin, int n_samples, int n_classes)
{
    CHK_NIL(onehot);
    CHK_NIL(dtype_onehot);
    CHK_ERR((n_classes > 0)? 0: 1);  
    CHK_NIL(orin);
    CHK_ERR((n_samples > 0)? 0: 1);
    CHK_NIL(dtype_orin);

    unsigned int dst_elem_size;
    unsigned int src_elem_size;
    CHK_ERR(getElemSize(&dst_elem_size, dtype_onehot));
    CHK_ERR(getElemSize(&src_elem_size, dtype_orin));

    void *res = calloc(n_samples * n_classes, dst_elem_size);
    if (res == NULL) {
        ERR_MSG("calloc failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }

    int i = 0;
    if (strcasecmp(dtype_orin, "uint8") == 0) {
        if (strcasecmp(dtype_onehot, "uint8") == 0) {
            unsigned char *dst = res;
            unsigned char *src = orin;
            for (i = 0; i < n_samples; ++i) {
                dst[i * n_classes + src[i]] = 1;
            }
        }
        else {
            ERR_MSG("src_type: %s to dst_type: %s not implemented yet, error.\n", dtype_orin, dtype_onehot);
            goto err_end;
        }
    }
    else {
        ERR_MSG("src_type: %s to other dtype not implemented yet, error.\n", dtype_orin);
        goto err_end;
    }
    *onehot = res;
    return SUCCESS;

err_end:
    free(res);
    return ERR_COD;
}

/**
 * @param norm_cost 归一化常数，将像素值归一化到[0, 1)
 */
int transformToFloat32FromUint8(float *dst, const unsigned char *src, int n_elems)
{
    CHK_NIL(dst);
    CHK_NIL(src);
    CHK_ERR((n_elems > 0)? 0: 1);

    int i;
    for (i = 0; i < n_elems; ++i) {
        dst[i] = (float)(src[i]);
    }

    return SUCCESS;
}

static int getDataMeanFloat32(double *mean, const float *data, int n_elems)
{
    CHK_NIL(mean);
    CHK_NIL(data);
    CHK_ERR((n_elems > 0)? 0: 1);

    double sum = 0.;
    int i;
    for (i = 0; i < n_elems; ++i) {
        sum += data[i];
    }
    *mean = sum / n_elems;
    return SUCCESS;
}

static int getDataMeanFloat64(double *mean, const double *data, int n_elems)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

static int getDataMeanInt32(double *mean, const int *data, int n_elems)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

static int getDataMeanInt64(double *mean, const long long *data, int n_elems)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

static int getDataMeanUint8(double *mean, const unsigned char *data, int n_elems)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

int getDataMean(double *mean, const void *data, const char *dtype, int n_elems)
{
    CHK_NIL(dtype);

    if (strcasecmp(dtype, "float32") == 0) {
        CHK_ERR(getDataMeanFloat32(mean, data, n_elems));
    } else if (strcasecmp(dtype, "float64") == 0) {
        CHK_ERR(getDataMeanFloat64(mean, data, n_elems));
    } else if (strcasecmp(dtype, "int32") == 0) {
        CHK_ERR(getDataMeanInt32(mean, data, n_elems));
    } else if (strcasecmp(dtype, "int64") == 0) {
        CHK_ERR(getDataMeanInt64(mean, data, n_elems));
    } else if (strcasecmp(dtype, "uint8") == 0) {
        CHK_ERR(getDataMeanUint8(mean, data, n_elems));
    } else {
        ERR_MSG("dtype: %s not supported ye, error.\n", dtype);
    }
    return SUCCESS;
}

static int getDataStdFloat32(double *std, const float *data, int n_elems)
{
    CHK_NIL(std);
    CHK_NIL(data);
    CHK_ERR((n_elems > 0)? 0: 1);

    double mean = 0.;
    CHK_ERR(getDataMeanFloat32(&mean, data, n_elems));

    double sum = 0.;
    int i;
    for (i = 0; i < n_elems; ++i) {
        sum += pow((double)(data[i] - mean), 2.);
    }
    *std = sqrt(sum / (n_elems - 1));
    return SUCCESS;
}

static int getDataStdFloat64(double *std, const double *data, int n_elems)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

static int getDataStdInt32(double *std, const int *data, int n_elems)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

static int getDataStdInt64(double *std, const long long *data, int n_elems)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

static int getDataStdUint8(double *std, const unsigned char *data, int n_elems)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

int getDataStd(double *std, const void *data, const char *dtype, int n_elems)
{
    CHK_NIL(dtype);

    if (strcasecmp(dtype, "float32") == 0) {
        CHK_ERR(getDataStdFloat32(std, data, n_elems));
    } else if (strcasecmp(dtype, "float64") == 0) {
        CHK_ERR(getDataStdFloat64(std, data, n_elems));
    } else if (strcasecmp(dtype, "int32") == 0) {
        CHK_ERR(getDataStdInt32(std, data, n_elems));
    } else if (strcasecmp(dtype, "int64") == 0) {
        CHK_ERR(getDataStdInt64(std, data, n_elems));
    } else if (strcasecmp(dtype, "uint8") == 0) {
        CHK_ERR(getDataStdUint8(std, data, n_elems));
    } else {
        ERR_MSG("dtype: %s not supported ye, error.\n", dtype);
    }
    return SUCCESS;
}

static int getDataNormalizationFloat32(float *data, int n_elems, double mean, double std)
{
    CHK_NIL(data);
    CHK_ERR((n_elems > 0)? 0: 1);

    //double mean, std;
    //CHK_ERR(getDataMeanFloat32(&mean, data, n_elems));
    //CHK_ERR(getDataStdFloat32(&std, data, n_elems));
    //fprintf(stdout, "mean = %f\nstd = %f\n", mean, std); 

    int i;
    for (i = 0; i < n_elems; ++i) {
        data[i] = (data[i] - mean) / std; // 原则上数据集不可能std为0
    }

    return SUCCESS;
}

static int getDataNormalizationFloat64(double *data, int n_elems, double mean, double std)
{
    ERR_MSG("notimplementederror, error.\n");
    return ERR_COD;
}

/**
 * @brief normalize mnist images, see main.py in https://github.com/Cerebras/ccc
 */
int getDataNormalization(void *data, const char *dtype, int n_elems, double mean, double std)
{
    CHK_NIL(dtype);

    if (strcasecmp(dtype, "float32") == 0) {
        CHK_ERR(getDataNormalizationFloat32(data, n_elems, mean, std));
    } else if (strcasecmp(dtype, "float64") == 0) {
        CHK_ERR(getDataNormalizationFloat64(data, n_elems, mean, std));
    } else {
        ERR_MSG("dtype: %s not supported ye, error.\n", dtype);
    }
    return SUCCESS;
}

int savetxtDataFlot32(const char *pth, const float *data, int n_samples, int n_features)
{
    CHK_NIL(data);
    CHK_NIL(pth);

    struct timeval t0, t1, t2;
    CHK_ERR(gettimeofday(&t0, NULL));
    
    FILE *fp = fopen(pth, "wb");
    if (fp == NULL) {
        ERR_MSG("fopen() failed, pth: %s, detail: %s, error.\n", pth, ERRNO_DETAIL(errno));
        return ERR_COD;
    }

    char buf[256];
    int i, j, offset;
    for (i = 0; i < n_samples; ++i) {
        for (j = 0; j < n_features; ++j) {
            offset = snprintf(buf, 256, "%f ", data[i * n_features + j]); 
            if (fwrite(buf, sizeof(char), offset, fp) != offset) {
                ERR_MSG("fwrite() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
                goto err_end;
            }
        }
        if (fseek(fp, -1, SEEK_CUR) == -1) {
            ERR_MSG("fseek() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
        if (fwrite("\n", sizeof(char), 1, fp) != 1) {
            ERR_MSG("fwrite() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
    }
    if (fclose(fp) == EOF) {
        ERR_MSG("fclose() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        goto err_end;
    }
    CHK_ERR(gettimeofday(&t1, NULL));
    timersub(&t1, &t0, &t2);
    fprintf(stdout, "savetxtDataFloat32 %s finish, time elapsed: %lu.%06lu.s\n", pth, t2.tv_sec, t2.tv_usec);
    return SUCCESS;

err_end:
    if (fclose(fp) == EOF) {
        ERR_MSG("fclose() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
    }
    return ERR_COD;
}

int savetxtDataUint8(const char *pth, const unsigned char *data, int n_samples, int n_features)
{
    CHK_NIL(data);
    CHK_NIL(pth);

    struct timeval t0, t1, t2;
    CHK_ERR(gettimeofday(&t0, NULL));
    
    FILE *fp = fopen(pth, "wb");
    if (fp == NULL) {
        ERR_MSG("fopen() failed, pth: %s, detail: %s, error.\n", pth, ERRNO_DETAIL(errno));
        return ERR_COD;
    }

    char buf[256];
    int i, j, offset;
    for (i = 0; i < n_samples; ++i) {
        for (j = 0; j < n_features; ++j) {
            offset = snprintf(buf, 256, "%u ", data[i * n_features + j]); 
            if (fwrite(buf, sizeof(char), offset, fp) != offset) {
                ERR_MSG("fwrite() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
                goto err_end;
            }
        }
        if (fseek(fp, -1, SEEK_CUR) == -1) {
            ERR_MSG("fseek() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
        if (fwrite("\n", sizeof(char), 1, fp) != 1) {
            ERR_MSG("fwrite() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
    }
    if (fclose(fp) == EOF) {
        ERR_MSG("fclose() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        goto err_end;
    }
    CHK_ERR(gettimeofday(&t1, NULL));
    timersub(&t1, &t0, &t2);
    fprintf(stdout, "savetxtDataUint8 %s finish, time elapsed: %lu.%06lu.s\n", pth, t2.tv_sec, t2.tv_usec);

    return SUCCESS;

err_end:
    if (fclose(fp) == EOF) {
        ERR_MSG("fclose() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
    }
    return ERR_COD;
}
