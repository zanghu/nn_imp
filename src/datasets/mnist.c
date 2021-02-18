#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "debug_macros.h"
#include "data_utils.h"
#include "io_utils.h"
#include "mnist.h"

#define MNIST_IMG_OFFSET (16)
#define MNIST_LABEL_OFFSET (8)

// mnist original file names
#define MNIST_TRAIN_IMAGES_NAME ("train-images-idx3-ubyte")
#define MNIST_TRAIN_LABELS_NAME ("train-labels-idx1-ubyte")
#define MNIST_TEST_IMAGES_NAME  ("t10k-images-idx3-ubyte")
#define MNIST_TEST_LABELS_NAME  ("t10k-labels-idx1-ubyte")

//#define NORM_CONST (255)

// 用法：声明栈变量data, load(&data, src_dir)
int loadMnistAll(struct MNIST *data, const char *src_dir)
{
    CHK_NIL(data);
    CHK_NIL(src_dir);

    struct timeval t0, t1, t2;
    CHK_ERR(gettimeofday(&t0, NULL));

    int fd = -1;
    int i;
    void *results[10] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    const char *names[4] = {MNIST_TRAIN_IMAGES_NAME, MNIST_TRAIN_LABELS_NAME, MNIST_TEST_IMAGES_NAME, MNIST_TEST_LABELS_NAME};
    const unsigned int offsets[4] = {MNIST_IMG_OFFSET, MNIST_LABEL_OFFSET, MNIST_IMG_OFFSET, MNIST_LABEL_OFFSET};
    const unsigned int sizes[4] = {MNIST_WIDTH * MNIST_HEIGHT * MNIST_N_TRAIN, MNIST_N_TRAIN, MNIST_WIDTH * MNIST_HEIGHT * MNIST_N_TEST, MNIST_N_TEST};

    char path[1024];
    for (i = 0; i < 4; ++i) {
        snprintf(path, 1024, "%s/%s", src_dir, names[i]);

        int fd = open(path, O_RDONLY); // open file
        if (fd == -1) {
            ERR_MSG("open() failed, path: %s, err_detail: %s, error.\n", path, ERRNO_DETAIL(errno));
            goto err_end;
        }

        results[i] = calloc(sizes[i], sizeof(unsigned char)); // alloc memory
        if (results[i] == NULL) {
            ERR_MSG("calloc() failed, %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }

        if (lseek(fd, offsets[i], SEEK_SET) == -1) { // skip the data header
            ERR_MSG("lseek() failed, %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }

        if (read(fd, results[i], sizes[i] * sizeof(unsigned char)) == -1) {; // read bytes
            ERR_MSG("calloc() failed, %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }

        if (close(fd) == -1) { // close file
            ERR_MSG("close() failed, err_detail: %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
    }

    int n_train_elems = 50000 * MNIST_HEIGHT * MNIST_WIDTH;
    int n_valid_elems = 10000 * MNIST_HEIGHT * MNIST_WIDTH;
    if ((results[4] = calloc(n_train_elems, sizeof(float))) == NULL) {
        ERR_MSG("calloc() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        goto err_end;
    }
    if ((results[6] = calloc(n_valid_elems, sizeof(float))) == NULL) {
        ERR_MSG("calloc() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        goto err_end;
    }
    int n_test_elems = MNIST_N_TEST * MNIST_HEIGHT * MNIST_WIDTH; 
    if ((results[8] = calloc(n_test_elems, sizeof(float))) == NULL) {
        ERR_MSG("calloc() failed, detail: %s, error.\n", ERRNO_DETAIL(errno));
        goto err_end;
    }

    CHK_ERR_GOTO(transformToFloat32FromUint8((float *)(results[4]), (unsigned char *)(results[0]), n_train_elems));

    double mean, std; // 训练集(前50000个)上的均值和方差
    CHK_ERR_GOTO(getDataMean(&mean, (float *)(results[4]), "float32", n_train_elems));
    CHK_ERR_GOTO(getDataStd(&std, (float *)(results[4]), "float32", n_train_elems));
    fprintf(stdout, "mean = %f\nstd = %f\n", mean, std); 

    CHK_ERR_GOTO(getDataNormalization(results[4], "float32", n_train_elems, mean, std));
    CHK_ERR_GOTO(transformToFloat32FromUint8((float *)(results[6]), (unsigned char *)(results[0]) + n_train_elems, n_valid_elems));
    CHK_ERR_GOTO(getDataNormalization(results[6], "float32", n_valid_elems, mean, std));
    CHK_ERR_GOTO(transformToFloat32FromUint8((float *)(results[8]), (unsigned char *)(results[2]), n_test_elems));
    CHK_ERR_GOTO(getDataNormalization(results[8], "float32", n_test_elems, mean, std)); // 测试集也使用训练集的均值和方差

    CHK_ERR_GOTO(transformOnehot((void **)(&(results[5])), "uint8", results[1], "uint8", 50000, MNIST_N_CLASSES));
    CHK_ERR_GOTO(transformOnehot((void **)(&(results[7])), "uint8", (void *)((unsigned char *)(results[1]) + 50000), "uint8", 10000, MNIST_N_CLASSES));
    CHK_ERR_GOTO(transformOnehot((void **)(&(results[9])), "uint8", results[3], "uint8", MNIST_N_TEST, MNIST_N_CLASSES));

    CHK_ERR_GOTO(gettimeofday(&t1, NULL));
    timersub(&t1, &t0, &t2);
    fprintf(stdout, "MNIST data read finish, time elapsed: %lu.%06lus\n", t2.tv_sec, t2.tv_usec);

    memset(data, 0, sizeof(struct MNIST));
    data->train_images = results[0];
    data->train_labels = results[1];
    data->test_images = results[2];
    data->test_labels = results[3];
    data->train_images_norm = results[4];
    data->train_labels_onehot = results[5];
    data->valid_images_norm = results[6];
    data->valid_labels_onehot = results[7];
    data->test_images_norm = results[8];
    data->test_labels_onehot = results[9];

    return SUCCESS;

err_end:
    for (i = 0; i < 10; ++i) {
        free(results[i]);
    }
    if (close(fd) == -1) {
        if (errno != EBADF) {
            ERR_MSG("close() failed, err_detail: %s, error.\n", ERRNO_DETAIL(errno));
            return ERR_COD;
        }
    }
    return ERR_COD;
}

int loadMnist(struct MNIST *mnist, const char *src_dir)
{
    CHK_ERR(loadMnistAll(mnist, src_dir));
    free(mnist->train_images);
    free(mnist->train_labels);
    free(mnist->test_images);
    free(mnist->test_labels);
    mnist->train_images = NULL;
    mnist->train_labels = NULL;
    mnist->test_images = NULL;
    mnist->test_labels = NULL;

    return 0;
}

void freeMnist(struct MNIST *data)
{
    if (data) {
        free(data->train_images);
        free(data->train_labels);
        free(data->test_images);
        free(data->test_labels);
        free(data->train_images_norm);
        free(data->train_labels_onehot);
        free(data->valid_images_norm);
        free(data->valid_labels_onehot);
        free(data->test_images_norm);
        free(data->test_labels_onehot);
    }
}

/*
int getMnistTrainBatchNumber(int *n_batches, int batch_size)
{
    CHK_NIL(n_batches);
    CHK_ERR((batch_size > 0)? 0: 1);

    int ret = MNIST_N_TRAIN / batch_size;
    if (MNIST_N_TRAIN % batch_size != 0) {
        ++ret;
    }
    *n_batches = ret;
    return SUCCESS;
}
*/

/**
 * @brief 取出指定序号的batch的训练数据
 *
 * @param data_float    返回参数, 图片数据读取的起始位置
 * @param label_onehot  返回参数, onehot格式类标数据读取的起始位置
 * @param n_samples     返回参数, 读取的样本数，可能小于batch_size
 * @param type          输入参数, 数据来自训练集还是测试集
 * @param mnist         输入参数, 完成初始化的数据集对象
 * @param n_train       输入参数, 用作训练样本的总数, 即索引在[0, n_train)区间的样本作为训练集 
 * @param batch_size    输入参数, 每个batch包含的样本数
 * @param batch_idx     输入参数, 本次取索引为batch_idx的batch 
 */
int getMnistNthBatch(const float *(*data_float), const unsigned char *(*label_onehot), int *n_samples, const char *type, const struct MNIST *mnist, int n_use, int batch_size, int batch_idx)
{
    CHK_NIL(data_float);
    CHK_NIL(label_onehot);
    CHK_NIL(type);
    CHK_NIL(n_samples);
    CHK_NIL(mnist);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((batch_idx >= 0)? 0: 1);
    CHK_ERR((n_use > 0)? 0: 1);
    CHK_NIL(mnist->train_images_norm);
    CHK_NIL(mnist->train_labels_onehot);
    CHK_NIL(mnist->test_images_norm);
    CHK_NIL(mnist->test_labels_onehot);

    float *data_all = NULL;
    unsigned char *label_onehot_all = NULL;
    if (strcasecmp(type, "train") == 0) {
        CHK_ERR((n_use < MNIST_N_TRAIN)? 0: 1); 
        data_all = mnist->train_images_norm;
        label_onehot_all = mnist->train_labels_onehot;
    } else if (strcasecmp(type, "test") == 0) {
        CHK_ERR((n_use < MNIST_N_TEST)? 0: 1); 
        data_all = mnist->test_images_norm;
        label_onehot_all = mnist->test_labels_onehot;
    } else {
        ERR_MSG("MNIST type: %s is not supported, error.\n", type);
        return ERR_COD;
    }

    if(batch_idx * batch_size >= n_use) { // 训练循环的一个epoch结束的标识
        *data_float = NULL;
        *label_onehot = NULL;
        return SUCCESS;
    }

    int offset_data = batch_idx * batch_size * MNIST_WIDTH * MNIST_HEIGHT;
    *data_float = data_all + offset_data;
    int offset_label = batch_idx * batch_size * MNIST_N_CLASSES; // 注意: onehot类标中头部没有偏移量MNIST_LABEL_OFFSET
    *label_onehot = label_onehot_all + offset_label;

    if ((batch_idx + 1) * batch_size > n_use) {
        *n_samples = n_use - batch_idx * batch_size;
    }
    else {
        *n_samples = batch_size;
    }
    return SUCCESS;
}

/**
 * @brief 取出指定序号的batch的训练数据
 *
 * @param data          返回参数, 图片数据读取的起始位置
 * @param label         返回参数, 类标数据读取的起始位置
 * @param n_samples     返回参数, 读取的样本数，可能小于batch_size
 * @param type          输入参数, 数据来自训练集还是测试集
 * @param mnist         输入参数, 完成初始化的数据集对象
 * @param n_train       输入参数, 用作训练样本的总数, 即索引在[0, n_train)区间的样本作为训练集 
 * @param batch_size    输入参数, 每个batch包含的样本数
 * @param batch_idx     输入参数, 本次取索引为batch_idx的batch 
 */
int getMnistNthBatchOrin(const unsigned char *(*data), const unsigned char *(*label), int *n_samples, const char *type, const struct MNIST *mnist, int n_use, int batch_size, int batch_idx)
{
    CHK_NIL(data);
    CHK_NIL(label);
    CHK_NIL(type);
    CHK_NIL(n_samples);
    CHK_NIL(mnist);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((batch_idx >= 0)? 0: 1);
    CHK_ERR((n_use > 0)? 0: 1);
    CHK_NIL(mnist->train_images);
    CHK_NIL(mnist->train_labels);
    CHK_NIL(mnist->test_images);
    CHK_NIL(mnist->test_labels);

    unsigned char *data_all = NULL;
    unsigned char *label_all = NULL;
    if (strcasecmp(type, "train") == 0) {
        CHK_ERR((n_use < MNIST_N_TRAIN)? 0: 1); 
        data_all = mnist->train_images;
        label_all = mnist->train_labels;
    } else if (strcasecmp(type, "test") == 0) {
        CHK_ERR((n_use < MNIST_N_TEST)? 0: 1); 
        data_all = mnist->test_images;
        label_all = mnist->test_labels;
    } else {
        ERR_MSG("MNIST type: %s is not supported, error.\n", type);
        return ERR_COD;
    }

    if(batch_idx * batch_size >= n_use) { // 训练循环的一个epoch结束的标识
        *data = NULL;
        *label = NULL;
        return SUCCESS;
    }

    int offset_data = batch_idx * batch_size * MNIST_WIDTH * MNIST_HEIGHT;
    *data = data_all + offset_data;
    int offset_label = batch_idx * batch_size;
    *label = label_all + offset_label;

    if ((batch_idx + 1) * batch_size > n_use) {
        *n_samples = n_use - batch_idx * batch_size;
    }
    else {
        *n_samples = batch_size;
    }
    return SUCCESS;
}

// 选出若干张图片，导出为numpy.savetxt的格式. 该方法用来验证测试原始文件中图形格式
int dumpMnistNumpyTxt(const struct MNIST *data, const char * type, const char *dst_dir,  unsigned int start, unsigned int end)
{
    CHK_NIL(dst_dir);
    CHK_NIL(data);
    CHK_ERR((start >= 0)? 0: 1);
    CHK_ERR((start < end)? 0: 1);

    char *prefix = NULL;
    unsigned char *images = NULL;
    if (strcasecmp(type, "train") == 0) {
        CHK_ERR((end <= MNIST_N_TRAIN)? 0: 1);
        images = data->train_images;
        prefix = "mnist_train_";
    } else if (strcasecmp(type, "test") == 0) {
        CHK_ERR((end <= MNIST_N_TEST)? 0: 1);
        images = data->test_images;
        prefix = "mnist_test_";
    } else {
        ERR_MSG("Unknow MNIST type: %s, error.\n", type);
        return ERR_COD;
    }

    struct timeval t0, t1, t2;
    CHK_ERR(gettimeofday(&t0, NULL));

    int fd;
    char path[1024];
    char img[4096];
    int i, j, k;
    for (i = start; i < end; ++i) {
        snprintf(path, 1024, "%s/%s%05d.txt", dst_dir, prefix, i);
        fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH); // chmod 664
        if (fd == -1) {
            ERR_MSG("open() failed, path: %s, err_detail: %s, error.\n", path, ERRNO_DETAIL(errno));
            return ERR_COD;
        }
        int offset = 0;
        for (j = 0; j < MNIST_HEIGHT; ++j) {
            for (k = 0; k < MNIST_WIDTH; ++k) {
                offset += snprintf(img + offset, 4096 - offset, "%u ", (unsigned int)(images[MNIST_HEIGHT * MNIST_WIDTH * i + j * MNIST_WIDTH + k]));
            }
            offset += snprintf(img + offset, 4096 - offset, "%s", "\n");
        }
        //fprintf(stderr, "offset = %d\nfd = %d\n", offset, fd);
        if (write(fd, img, offset * sizeof(char)) == -1) {
            ERR_MSG("write() failed, %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
        if (close(fd) == -1) { // close file
            ERR_MSG("close() failed, err_detail: %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
    }

    CHK_ERR(gettimeofday(&t1, NULL));
    timersub(&t1, &t0, &t2);
    fprintf(stdout, "MNIST data dump finish, number = %d, time elapsed: %lu.%06lus\n", end - start, t2.tv_sec, t2.tv_usec);

    return SUCCESS;

err_end:
    if (close(fd) == -1) {
        if (errno != EBADF) {
            ERR_MSG("close() failed, err_detail: %s, error.\n", ERRNO_DETAIL(errno));
            return ERR_COD;
        }
    }
    return ERR_COD;
}

/**
 * @brief 导出加工后的数据, 用于与python加工后数据进行对比，确保处理函数的正确性
 */
int dumpMnistTransformed(const struct MNIST *mnist, const char *dst_dir)
{
    CHK_NIL(mnist);
    CHK_NIL(dst_dir);

    char train_data_txt[1024];
    char train_label_txt[1024];
    char valid_data_txt[1024];
    char valid_label_txt[1024];
    char test_data_txt[1024];
    char test_label_txt[1024];

    snprintf(train_data_txt, 1024, "%s/mnist_train_images_transformed.txt", dst_dir);
    snprintf(train_label_txt, 1024, "%s/mnist_train_label_transformed.txt", dst_dir);
    snprintf(valid_data_txt, 1024, "%s/mnist_valid_images_transformed.txt", dst_dir);
    snprintf(valid_label_txt, 1024, "%s/mnist_valid_label_transformed.txt", dst_dir);
    snprintf(test_data_txt, 1024, "%s/mnist_test_images_transformed.txt", dst_dir);
    snprintf(test_label_txt, 1024, "%s/mnist_test_label_transformed.txt", dst_dir);

    CHK_ERR(savetxtMatrixFlot32(train_data_txt, mnist->train_images_norm, 50000, MNIST_HEIGHT * MNIST_WIDTH));
    CHK_ERR(savetxtMatrixUint8(train_label_txt, mnist->train_labels_onehot, 50000, MNIST_N_CLASSES));
    CHK_ERR(savetxtMatrixFlot32(valid_data_txt, mnist->valid_images_norm, 10000, MNIST_HEIGHT * MNIST_WIDTH));
    CHK_ERR(savetxtMatrixUint8(valid_label_txt, mnist->valid_labels_onehot, 10000, MNIST_N_CLASSES));
    CHK_ERR(savetxtMatrixFlot32(test_data_txt, mnist->test_images_norm, MNIST_N_TEST, MNIST_HEIGHT * MNIST_WIDTH));
    CHK_ERR(savetxtMatrixUint8(test_label_txt, mnist->test_labels_onehot, MNIST_N_TEST, MNIST_N_CLASSES));

    return SUCCESS;
}

