#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "debug_macros.h"
#include "math_utils.h"
#include "activations.h"
#include "gemm.h"
#include "tensor.h"

struct Tensor {
    enum DType dtype;
    int b; // batch_size
    int row;
    int col;
    int c; // channel
    int n; // 实际装填的样本数， n<=b
    float *data;
    int *data_i32;
    //long long *data_i64;
    //double *data_f64;
};

char *getTensorDtypeStrFromEnum(enum DType dtype)
{
    switch (dtype) {
        case FLOAT32:
        return "float32";

        case FLOAT64:
        return "float64";

        case INT32:
        return "int32";

        case INT64:
        return "int64";

        case UINT8:
        return "uint8";

        default:
        return "unknow dtype";
    }
    return "unknow dtype";
}

void initTensorParameterAsWeight(struct Tensor *matrix)
{
    int inputs = matrix->row;
    float scale = sqrt(2. / inputs);
    int i = 0;
    for(i = 0; i < matrix->b * matrix->row * matrix->col * matrix->c; ++i){
        matrix->data[i] = scale * rand_uniform(-1, 1);
    }
}

void initTensorParameterAsBias(struct Tensor *matrix)
{
    memset(matrix->data, 0, matrix->b * matrix->row * matrix->col * matrix->c * sizeof(float));
}

int createTensor(struct Tensor **t, int batch_size, int row, int col, int channel)
{
    CHK_NIL(t);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((row > 0)? 0: 1);
    CHK_ERR((col > 0)? 0: 1);
    CHK_ERR((channel > 0)? 0: 1);

    struct Tensor *matrix = calloc(1, sizeof(struct Tensor));
    if (!matrix) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    matrix->data = calloc(batch_size * row * col * channel, sizeof(float));
    if (matrix->data == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        free(matrix);
        return ERR_COD;
    }
    matrix->b = batch_size;
    matrix->dtype = FLOAT32;
    matrix->row = row;
    matrix->col = col;
    matrix->c = channel;
    matrix->n = 0; // 初始样本数为0

    *t = matrix;
    return SUCCESS;
}

int createTensorI32(struct Tensor **t, int batch_size, int row, int col, int channel)
{
    CHK_NIL(t);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((row > 0)? 0: 1);
    CHK_ERR((col > 0)? 0: 1);
    CHK_ERR((channel > 0)? 0: 1);

    struct Tensor *matrix = calloc(1, sizeof(struct Tensor));
    if (!matrix) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return ERR_COD;
    }
    matrix->data_i32 = calloc(batch_size * row * col * channel, sizeof(int));
    if (matrix->data_i32 == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        free(matrix);
        return ERR_COD;
    }
    matrix->b = batch_size;
    matrix->dtype = INT32;
    matrix->row = row;
    matrix->col = col;
    matrix->c = channel;
    matrix->n = 0; // 初始样本数为0

    *t = matrix;
    return SUCCESS;
}

void destroyTensor(struct Tensor *matrix)
{
    if (matrix) {
        free(matrix->data);
        free(matrix->data_i32);
    }
    free(matrix);
}

int getTensorShape(int *b, int *row, int *col, int *c, const struct Tensor *tensor)
{
    CHK_NIL(b);
    CHK_NIL(row);
    CHK_NIL(col);
    CHK_NIL(c);
    CHK_NIL(tensor);

    *b = tensor->b;
    *row = tensor->row;
    *col = tensor->col;
    *c = tensor->c;

    return SUCCESS;
}

int getTensorRow(int *row, const struct Tensor *matrix)
{
    CHK_NIL(row);
    CHK_NIL(matrix);
    *row = matrix->row;
    return SUCCESS;
}

int getTensorCol(int *col, const struct Tensor *matrix)
{
    CHK_NIL(col);
    CHK_NIL(matrix);
    *col = matrix->col;
    return SUCCESS;
}

int getTensorChannel(int *c, const struct Tensor *matrix)
{
    CHK_NIL(c);
    CHK_NIL(matrix);
    *c = matrix->c;
    return SUCCESS;
}

int getTensorBatch(int *b, const struct Tensor *matrix)
{
    CHK_NIL(b);
    CHK_NIL(matrix);
    *b = matrix->b;
    return SUCCESS;
}

int getTensorSamples(int *n, const struct Tensor *matrix)
{
    CHK_NIL(n);
    CHK_NIL(matrix);
    *n = matrix->n;
    return SUCCESS;
}

int getTensorDType(enum DType *dtype, const struct Tensor *matrix)
{
    CHK_NIL(dtype);
    CHK_NIL(matrix);
    *dtype = matrix->dtype;
    return SUCCESS;
}

int getTensorData(void **data, struct Tensor *tensor)
{
    CHK_NIL(data);
    CHK_NIL(tensor);

    switch (tensor->dtype) {
        case FLOAT32:
        *data = tensor->data;
        break;

        case INT32:
        *data = tensor->data_i32;
        break;

        default:
        *data = tensor->data;
        break;
    }
    return SUCCESS;
}

int setTensorData(struct Tensor *tensor, const void *data, enum DType dtype, int n)
{
    CHK_NIL(tensor); 
    CHK_NIL(data);

    int n_features = tensor->col * tensor->row * tensor->c;
    switch (tensor->dtype) {
        case FLOAT32:
        switch (dtype) {
            case UINT8:
            {
                const unsigned char *src = data;
                int i, j;
                for (i = 0; i < n; ++i) {
                    for (j = 0; j < n_features; ++j) {
                        tensor->data[i * n_features + j] = (float)(src[i * n_features + j]);
                    }
                }
            }
            break;

            default:
            ERR_MSG("src_dtype: %s to dst_dtype: %s not supported, error.\n", getTensorDtypeStrFromEnum(dtype), getTensorDtypeStrFromEnum(tensor->dtype));
            return ERR_COD;
        }
        break;

        default:
        return ERR_COD;
    }
    tensor->n = n;
    return SUCCESS;
}

// 前向传播过程的非线性运算部分
int activateTensor(struct Tensor *y, const struct Tensor *x, enum ActivationType act_type)
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
int deactivateTensor(struct Tensor *delta_out, const struct Tensor *delta_in, const struct Tensor *input, enum ActivationType act_type)
{
    int i = 0;
    switch (act_type) {
        case LOGISTIC:
        for (i = 0; i < input->b * input->row * input->col * input->c; ++i) {
            delta_out->data[i] = logistic_gradient(input->data[i]) * delta_in->data[i]; // 注意这里的x应该是outputs[i]，而不是hiddens[i]
        }
        break;

        default:
        ERR_MSG("Unknow dectivation_type, error.\n");
        return ERR_COD;
    }
    return SUCCESS;
}

/**
 * @param x: 输入参数，shape = (batch_size, n_input)
 * @param y: 输入参数，shape = (barch_size, n_output)
 * @param z: 输出参数, shape = (n_input, n_output)
 * @param b: 输入参数, shape = (1, n_output)
 */
int linearTensor(struct Tensor *z, const struct Tensor *x, int xt, const struct Tensor *y, int yt, const struct Tensor *b)
{
    CHK_NIL(z);
    CHK_NIL(x);
    CHK_NIL(y);
    CHK_ERR((x->b == y->b)? 0: 1);
    CHK_ERR((z->row == x->col)? 0: 1);
    CHK_ERR((z->col == y->col)? 0: 1);

    if (b) {
        int i = 0;
        for (i = 0; i < x->b; ++i) {
            memcpy(z->data + i * x->col, b->data, sizeof(float) * x->col);
        }
    }

    // gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
    //     float *A, int lda, 
    //     float *B, int ldb,
    //     float BETA,
    //     float *C, int ldc)
    gemm(xt, yt, x->col, y->col, x->b, 1., 
        x->data, x->col, 
        y->data, y->col, 
        1., 
        z->data, y->col);

    return SUCCESS;
}

// 2个形状完全相同的矩阵x和y的元素做Pointwise乘法, 之后每行求和, 压缩成一个向量, 结果保存在z中
// 该函数用来在反向传播时计算偏置bias的梯度
int mulTensorPointwiseAndSum(struct Tensor *z, const struct Tensor *x, const struct Tensor *y)
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
int addTensor(struct Tensor *x, struct Tensor *y, float lr, float momentum)
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

int softmaxTensor(struct Tensor *output, const struct Tensor *input)
{
    CHK_NIL(output);
    CHK_NIL(input);

    int n = output->b;
    int k = output->col;

    int i;
    for (i = 0; i < n; ++i) {
        float largest = -FLT_MAX;
        int j;
        for(j = 0; j < k; ++j){
            if(input->data[i * k + j] > largest) {
                largest = input->data[i * k + j];
            }
        }
        float sum = 0;
        for(j = 0; j < k; ++j){
            float e = exp(input->data[i * k + j] - largest);
            sum += e;
            output->data[i * k + j] = e;
        }
        for(j = 0; j < k; ++j){
            (output->data[i * k + j]) /= sum;
        }
    }
    return SUCCESS;
}

int addTensor2(struct Tensor *delta, const struct Tensor *y, const struct Tensor *gt)
{
    CHK_NIL(delta);
    CHK_NIL(y);
    CHK_NIL(gt);

    int n = delta->b * delta->col;
    int i;
    for (i = 0; i < n; ++i) {
        delta->data[i] = gt->data_i32[i] - y->data[i];
    }

    return SUCCESS;
}

// 一个batch上的对数极大似然
int probTensor(float *val, const struct Tensor *p, const struct Tensor *gt)
{
    CHK_NIL(val);
    CHK_NIL(p);
    CHK_NIL(gt);

    int b = p->b;
    int c = p->col;
    float sum_log_p = 0.;
    int i, j;
    for (i = 0; i < b; ++i) {
        for (j = 0; j < c; ++j) {
            if (gt->data_i32[i * c + j] != 0) {
                sum_log_p += p->data[i * c + j];
                break;
            }
        }
    }
    *val = sum_log_p / b; // 计算平均值, 用于观察评估寻俩效果的代价值建议与样本数无关
    return SUCCESS;
}
 

