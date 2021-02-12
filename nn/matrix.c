#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "debug_macros.h"
#include "math_utils.h"

struct Matrix {
    int row;
    int col;
    int c; // channel
    int b; // batch_size
    float *data;
};

void initMatrixParameterAsWeight(struct Matrix *matrix)
{
    int inputs = matrix->row;
    float scale = sqrt(2. / inputs);
    int i = 0;
    for(i = 0; i < matrix->b * matrix->row * matrix->col * matrix->c; ++i){
        matrix->data[i] = scale * rand_uniform(-1, 1);
    }
}

void initMatrixParameterAsBias(struct Matrix *matrix)
{
    memset(matrix->data, 0, matrix->b * matrix->row * matrix->col * matrix->c * sizeof(float));
}

struct Matrix *createMatrix(int batch_size, int row, int col, int channel)
{
    if (batch_size <= 0) return NULL;
    if (row <= 0) return NULL;
    if (col <= 0) return NULL;
    if (channel <= 0) return NULL;

    struct Matrix *matrix = calloc(1, sizeof(struct Matrix));
    if (!matrix) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return NULL;
    }
    matrix->data = calloc(batch_size * row * col * channel, sizeof(struct Matrix));
    if (matrix->data == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        free(matrix);
        return NULL;
    }
    matrix->row = row;
    matrix->col = col;
    matrix->b = batch_size;
    return matrix;
}

void destroyMatrix(struct Matrix *matrix)
{
    if (matrix) {
        free(matrix->data);
    }
    free(matrix);
}

int getMatrixRow(const struct Matrix *matrix)
{
    return matrix->row;
}

int getMatrixCol(const struct Matrix *matrix)
{
    return matrix->col;
}

int getMatrixChannel(const struct Matrix *matrix)
{
    return matrix->c;
}

int getMatrixBatch(const struct Matrix *matrix)
{
    return matrix->b;
}

/**
 * @param x: n*k的矩阵
 * @param w: m*k的矩阵，对应于网络中的w转置矩阵
 * @param b: m维行向量
 * @param h: 返回矩阵，h = mul(x, w_T) + b, b作为行向量进行复制
 */
static void mul2DForward(float *h, const float *x, const float *w, const float *b, int n, int k, int m)
{
    int i, j, t;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            int tmp = 0;
            for (t = 0; t < k; ++t) {
                tmp += x[i * k + t] * w[k * j + t]; // mul(x, w_T)[i][j]
            }
            h[i * m + j] = tmp + b[j];
        }
    }
}

/*
struct Matrix *mulMatrix(const struct Matrix *w, const struct Matrix *x, const struct Matrix *b)
{
    if (!w) return NULL;
    if (!x) return NULL;
    if (!b) return NULL;

    if (w->row != b->col || w->c != 1 || w->col != x->col) {
        ERR_MSG("Matrix size not match, multiply failed, error.\n");
        return NULL;
    }

    int n_out = w->row;
    struct Matrix *h = createMatrix(x->b, n_out, 1, 1);
    if (h == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        return NULL;
    }
    
    // 矩阵乘法
    mul2D(h->data, x->data, w->data, x->b, w->col, w->row);
    h->b = x->b;
    h->row = 1;
    h->col = w->row;
    h->c = 1;
    return h;
}
*/

int linearMatrixForward(struct Matrix *y, const struct Matrix *w, const struct Matrix *x, const struct Matrix *b)
{
    CHK_NIL(y);
    CHK_NIL(w);
    CHK_NIL(x);
    CHK_NIL(b);

    CHK_ERR((y->row == x->row)? 0: 1);
    CHK_ERR((y->col == w->row)? 0: 1);
    CHK_ERR((w->row == b->col)? 0: 1);
    CHK_ERR((w->c == 1)? 0: 1);
    CHK_ERR((w->col == x->col)? 0:1);

    int n_out = w->row;
    struct Matrix *h = createMatrix(x->b, n_out, 1, 1);
    if (h == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        return ERR_COD;
    }
    
    // 矩阵乘法
    mul2DForward(y->data, x->data, w->data, b->data, x->b, w->col, w->row);
    return SUCCESS;
}

/**
 * @param x: n*k的矩阵
 * @param w: k*m的矩阵
 * @param h: 返回矩阵，h = mul(x, w)
 */
static void mul2DBackward(float *h, const float *x, const float *w, int n, int k, int m)
{
    int i, j, t;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            int tmp = 0;
            for (t = 0; t < k; ++t) {
                tmp += x[i * k + t] * w[k * m + j]; // mul(x, w_T)[i][j]
            }
            h[i * m + j] = tmp;
        }
    }
}

int linearMatrixBackward(struct Matrix *y, const struct Matrix *w, const struct Matrix *x)
{
    CHK_NIL(y);
    CHK_NIL(w);
    CHK_NIL(x);

    CHK_ERR((y->row == x->row)? 0: 1);
    CHK_ERR((y->col == w->col)? 0: 1);
    CHK_ERR((w->c == 1)? 0: 1);
    CHK_ERR((w->row == x->col)? 0:1);

    int n_out = w->col;
    struct Matrix *h = createMatrix(x->b, n_out, 1, 1);
    if (h == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        return ERR_COD;
    }
    
    // 矩阵乘法
    mul2DBackward(y->data, x->data, w->data, x->b, w->row, w->col);
    return SUCCESS;
}

int activateMatrix(struct Matrix *y, const struct Matrix *x, const char *actvation_type)
{
    CHK_NIL(x);
    CHK_NIL(actvation_type);

    int i = 0;
    if  (strcasecmp(actvation_type, "logistic") == 0) {
        for (i = 0; i < x->b * x->row * x->col * x->c; ++i) {
            y->data[i] = 1./(1. + exp(-1 * (x->data[i])));
        }
    } else {
        ERR_MSG("Unknow activation_type, error.\n");
        return ERR_COD;
    }
    return SUCCESS;
}

int deactivateMatrix(struct Matrix *y, const struct Matrix *output, const struct Matrix *delta, const char *deactvation_type)
{
    int i = 0;
    if (strcasecmp(deactvation_type, "logistic") == 0) {
        for (i = 0; i < output->b * output->row * output->col * output->c; ++i) {
            y->data[i] = ((1 - output->data[i]) * output->data[i]) * delta->data[i]; // 注意这里的x应该是outputs[i]，而不是hiddens[i]
        }
    } else {
        ERR_MSG("Unknow dectivation_type, error.\n");
        return ERR_COD;
    }
    return SUCCESS;
}