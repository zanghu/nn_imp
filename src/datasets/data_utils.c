#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    