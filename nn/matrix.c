#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "debug_macros.h"
#include "math_utils.h"
#include "activations.h"
#include "gemm.h"

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
 * @param A: n*k或k*n的矩阵, 乘法左侧矩阵
 * @param B: k*m或m*k的矩阵, 乘法右侧矩阵
 * @param C: n*m或m*n的矩阵, 偏置矩阵, 可以为NULL
 * @param R: 返回矩阵，h = mul(A, B) + C
 */
/*
static void mul2DBackward(float *R, const float *A, int a_t, const float *B, int b_t, const float *C, int c_t, int n, int k, int m)
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
*/

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

// 前向传播过程的非线性运算部分
int activateMatrix(struct Matrix *y, const struct Matrix *x, enum ActivationType act_type)
{
    CHK_NIL(x);

    int i = 0;
    switch (act_type) {
        case LOGISTIC:
        for (i = 0; i < x->b * x->row * x->col * x->c; ++i) {
            //y->data[i] = 1./(1. + exp(-1 * (x->data[i])));
            y->data[i] = logistic_activate(x->data[i]);
        }
        break;

        default:
        ERR_MSG("Unknow activation_type, error.\n");
        return ERR_COD;
    }
    return SUCCESS;
}

// 反向传播过程的非线性运算部分, 计算后的结果会替换输入值delta的原有内容
int deactivateMatrix(const struct Matrix *delta, const struct Matrix *output, enum ActivationType act_type)
{
    int i = 0;
    switch (act_type) {
        case LOGISTIC:
        for (i = 0; i < output->b * output->row * output->col * output->c; ++i) {
            delta->data[i] = logistic_gradient(output->data[i]) * delta->data[i]; // 注意这里的x应该是outputs[i]，而不是hiddens[i]
        }
        break;

        default:
        ERR_MSG("Unknow dectivation_type, error.\n");
        return ERR_COD;
    }
    return SUCCESS;
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

/**
 * @param x: 输入参数，shape = (batch_size, n_input)
 * @param y: 输入参数，shape = (barch_size, n_output)
 * @param z: 输出参数, shape = (n_input, n_output)
 */
int mulMatrix(struct Matrix *z, const struct Matrix *x, const struct Matrix *y)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->b == y->b)? 0: 1);
    CHK_ERR((z->row == x->col)? 0: 1);
    CHK_ERR((z->col == y->col)? 0: 1);

    // gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
    //     float *A, int lda, 
    //     float *B, int ldb,
    //     float BETA,
    //     float *C, int ldc)
    gemm(1, 0, x->col, y->col, x->b, 1., 
        x->data, x->col, 
        y->data, y->col, 
        1., 
        z->data, y->col);

    return SUCCESS;
}

// 2个形状完全相同的矩阵x和y的元素做Pointwise乘法, 之后每行求和, 压缩成一个向量, 结果保存在z中
// 该函数用来在反向传播时计算偏置bias的梯度
int mulMatrixPointwiseAndSum(struct Matrix *z, const struct Matrix *x, const struct Matrix *y)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->b == y->b)? 0: 1);
    CHK_ERR((x->col == y->col)? 0: 1);
    CHK_ERR((x->col == z->col)? 0: 1);

    int b = x->b; // batch_size
    int c = x->col; // n_output
    int i, j;
    for (i = 0; i < c; ++i) {
        z->data[c] = 0.;
        for (j = 0; j < b; ++j) {
            z->data[c] += x->data[j * c + i] * y->data[j * c + i];
        }
    }
    return SUCCESS;
}

// x = x + lr * y
// y = momentum * (lr * y), 动量法，保存用作下一轮使用
int addMatrix(struct Matrix *x, struct Matrix *y, float lr, float momentum)
{
    CHK_NIL(x);
    CHK_NIL(y);

    // STEP 0: prepare
    int n = y->b * y->row * y->col * y->c;

    // STEP 1: y = lr * y
    int i = 0;
    for (i = 0; i < n; ++i) {
        y->data[i] *= lr;
    }

    // STEP 2: x = x + lr
    for (i = 0; i < n; ++i) {
        x->data[i] += y->data[i];
    }

    // STEP 3: y = momentum * y
    for (i = 0; i < n; ++i) {
        y->data[i] *= momentum;
    }

    return SUCCESS;
}