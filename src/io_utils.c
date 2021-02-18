#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "debug_macros.h"

int savetxtMatrixFlot32(const char *pth, const float *matrix, int n_samples, int n_features)
{
    CHK_NIL(matrix);
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
            offset = snprintf(buf, 256, "%f ", matrix[i * n_features + j]); 
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

int savetxtMatrixUint8(const char *pth, const unsigned char *matrix, int n_samples, int n_features)
{
    CHK_NIL(matrix);
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
            offset = snprintf(buf, 256, "%u ", matrix[i * n_features + j]); 
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