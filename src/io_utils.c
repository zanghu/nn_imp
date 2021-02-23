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
            offset = snprintf(buf, 256, "%.18f ", matrix[i * n_features + j]); 
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
    //fprintf(stdout, "savetxtDataFloat32 %s finish, time elapsed: %lu.%06lu.s\n", pth, t2.tv_sec, t2.tv_usec);
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
    //fprintf(stdout, "savetxtDataUint8 %s finish, time elapsed: %lu.%06lu.s\n", pth, t2.tv_sec, t2.tv_usec);

    return SUCCESS;

err_end:
    if (fclose(fp) == EOF) {
        ERR_MSG("fclose() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
    }
    return ERR_COD;
}

int getFileBytes(int *size, FILE *fp)
{
    CHK_NIL(size);
    CHK_NIL(fp);

    if (fseek(fp, 0L, SEEK_END) == -1) {
        ERR_MSG("fseek() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }

    int tmp = ftell(fp);
    if (tmp == -1) {
        ERR_MSG("ftell() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }

    if (fseek(fp, 0L, SEEK_SET) == -1) {
        ERR_MSG("fseek() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }

    *size = tmp;
    return SUCCESS;
}

int loadtxtBlobFloat32(int *n_loaded, const char *pth, float *blob, int size_blob)
{
    CHK_NIL(n_loaded);
    CHK_NIL(blob);
    CHK_NIL(pth);
    fprintf(stdout, "size_blob = %d\n", size_blob);

    struct timeval t0, t1, t2;
    CHK_ERR(gettimeofday(&t0, NULL));
    
    FILE *fp = fopen(pth, "rb");
    if (fp == NULL) {
        ERR_MSG("fopen() failed, pth: %s, detail: %s, error.\n", pth, ERRNO_DETAIL(errno));
        return ERR_COD;
    }

    int size;
    CHK_ERR(getFileBytes(&size, fp)); // 获取文件大小
    fprintf(stdout, "file_size = %d\n", size);

    char *buf = calloc(size + 1, sizeof(char));
    if (buf == NULL) {
        ERR_MSG("calloc() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        goto err_close;
    }

    if (fread(buf, size, sizeof(char), fp) < size) {
        if (ferror(fp)) {
            ERR_MSG("fread() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
            goto err_close;
        }
    }

    if (fclose(fp) == EOF) {
        ERR_MSG("fclose() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        goto err_end;
    }

    if (buf[size - 1] == '\n') {
        buf[size - 1] = '\0';
    }

    // parse data
    char *cur = buf;
    char *pos = NULL;
    while ((pos = strchr(cur, '\n')) != NULL) {
        *pos = ' ';
        cur = pos + 1;
    }

    int cnt = 0;
    cur = buf;
    while ((pos = strchr(cur, ' ')) != NULL) {
        if (cnt == size_blob) {
            ERR_MSG("blob size error, error.\n");
            goto err_end;
        }
        *pos = '\0';
        blob[cnt] = atof(cur);
        ++cnt;
        cur = pos + 1;
    }
    blob[cnt] = atof(cur);
    ++cnt;
    if (cnt > size_blob) {
        ERR_MSG("blob size not match, error.\n");
        goto err_end;
    }
    free(buf);
    *n_loaded = cnt; // read num

    CHK_ERR(gettimeofday(&t1, NULL));
    timersub(&t1, &t0, &t2);
    fprintf(stdout, "savetxtDataFloat32 %s finish, time elapsed: %lu.%06lu.s\n", pth, t2.tv_sec, t2.tv_usec);
    return SUCCESS;

err_close:
    if (fclose(fp) == EOF) {
        ERR_MSG("fclose() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
    }
err_end:
    if (buf) {
        free(buf);
    }
    return ERR_COD;
}