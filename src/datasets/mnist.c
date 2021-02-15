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
#include "mnist.h"

// mnist original file names
static const char *s_train_images = "train-images-idx3-ubyte";
static const char *s_train_labels = "train-labels-idx1-ubyte";
static const char *s_test_images = "t10k-images-idx3-ubyte";
static const char *s_test_labels = "t10k-labels-idx1-ubyte";

// mnist file size
static const unsigned int s_size_train_img = MNIST_N_TEST + MNIST_WIDTH * MNIST_HEIGHT * MNIST_N_TRAIN;
static const unsigned int s_size_train_label = MNIST_LABEL_OFFSET + MNIST_N_TRAIN * 1;
static const unsigned int s_size_test_img = MNIST_N_TEST + MNIST_WIDTH * MNIST_HEIGHT * MNIST_N_TEST;
static const unsigned int s_size_test_label = MNIST_LABEL_OFFSET + MNIST_N_TEST * 1;

// 用法：声明栈变量data, load(&data, src_dir)
int loadMnist(struct MNIST *data, const char *src_dir)
{
    CHK_NIL(data);
    CHK_NIL(src_dir);

    struct timeval t0, t1, t2;
    CHK_ERR(gettimeofday(&t0, NULL));

    int fd = -1;
    int i;
    unsigned char *results[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    const char *names[4] = {s_train_images, s_train_labels, s_test_images, s_test_labels};
    const unsigned int sizes[4] = {s_size_train_img, s_size_train_label, s_size_test_img, s_size_test_label};

    char path[1024];
    for (i = 0; i < 4; ++i) {
        snprintf(path, 1024, "%s/%s", src_dir, names[i]);

        int fd = open(path, O_RDONLY); // open file
        if (fd == -1) {
            ERR_MSG("open() failed, path: %s, err_detail: %s, error.\n", path, ERRNO_DETAIL(errno));
            goto err_end;
        }

        results[i] = calloc(s_size_train_img, sizeof(unsigned char)); // alloc memory
        if (results[i] == NULL) {
            ERR_MSG("read() failed, %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }

        if (read(fd, results[i], sizes[i] * sizeof(unsigned char)) == -1) {; // read bytes
            ERR_MSG("read() failed, %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }

        if (close(fd) == -1) { // close file
            ERR_MSG("close() failed, err_detail: %s, error.\n", ERRNO_DETAIL(errno));
            goto err_end;
        }
    }
    CHK_ERR_GOTO(transformOnehot((void **)(&(results[4])), "uint8", results[1] + MNIST_LABEL_OFFSET, "uint8", MNIST_N_TRAIN, MNIST_N_CLASSES));
    CHK_ERR_GOTO(transformOnehot((void **)(&(results[5])), "uint8", results[3] + MNIST_LABEL_OFFSET, "uint8", MNIST_N_TEST, MNIST_N_CLASSES));

    CHK_ERR_GOTO(gettimeofday(&t1, NULL));
    timersub(&t1, &t0, &t2);
    fprintf(stdout, "MNIST data read finish, time elapsed: %lu.%06lus\n", t2.tv_sec, t2.tv_usec);

    memset(data, 0, sizeof(struct MNIST));
    data->train_images = results[0];
    data->train_labels = results[1];
    data->test_images = results[2];
    data->test_labels = results[3];
    data->train_labels_onehot = results[4];
    data->test_labels_onehot = results[5];

    return SUCCESS;

err_end:
    for (i = 0; i < 6; ++i) {
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

void freeMnist(struct MNIST *data)
{
    if (data) {
        free(data->train_images);
        free(data->train_labels);
        free(data->test_images);
        free(data->test_labels);
        free(data->train_labels_onehot);
        free(data->test_labels_onehot);
    }
}

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

/**
 * @brief 取出指定序号的batch的训练数据
 *
 * @param data          返回参数, 图片数据读取的起始位置
 * @param label         返回参数, 类标数据读取的起始位置
 * @param label_onehot  返回参数, onehot格式类标数据读取的起始位置
 * @param n_samples     返回参数, 读取的样本数，可能小于batch_size
 * @param type          输入参数, 数据来自训练集还是测试集
 * @param mnist         输入参数, 完成初始化的数据集对象
 * @param n_train       输入参数, 用作训练样本的总数, 即索引在[0, n_train)区间的样本作为训练集 
 * @param batch_size    输入参数, 每个batch包含的样本数
 * @param batch_idx     输入参数, 本次取索引为batch_idx的batch 
 */
int getMnistNthBatch(const unsigned char *(*data), const unsigned char *(*label), const unsigned char *(*label_onehot), int *n_samples, const char *type, const struct MNIST *mnist, int n_use, int batch_size, int batch_idx)
{
    CHK_NIL(data);
    CHK_NIL(type);
    CHK_NIL(n_samples);
    CHK_NIL(mnist);
    CHK_ERR((batch_size > 0)? 0: 1);
    CHK_ERR((batch_idx >= 0)? 0: 1);
    CHK_ERR((n_use > 0)? 0: 1);

    unsigned char *data_all = NULL;
    unsigned char *label_all = NULL;
    unsigned char *label_onehot_all = NULL;
    if (strcasecmp(type, "train") == 0) {
        CHK_ERR((n_use < MNIST_N_TRAIN)? 0: 1); 
        data_all = mnist->train_images;
        label_all = mnist->train_labels;
        label_onehot_all = mnist->train_labels_onehot;
    } else if (strcasecmp(type, "test") == 0) {
        CHK_ERR((n_use < MNIST_N_TEST)? 0: 1); 
        data_all = mnist->test_images;
        label_all = mnist->test_labels;
        label_onehot_all = mnist->test_labels_onehot;
    } else {
        ERR_MSG("MNIST type: %s is not supported, error.\n", type);
        return ERR_COD;
    }

    if(batch_idx * batch_size >= n_use) { // 训练循环的一个epoch结束的标识
        *data = NULL;
        *label = NULL;
    }

    int offset_data = MNIST_IMG_OFFSET + batch_idx * batch_size * MNIST_WIDTH * MNIST_HEIGHT;
    *data = data_all + offset_data;
    if (label) {
        int offset_label = MNIST_LABEL_OFFSET + batch_idx * batch_size * 1;
        *label = label_all + offset_label;
    }
    if (label_onehot) {
        int offset_label = batch_idx * batch_size * MNIST_N_CLASSES; // 注意: onehot类标中头部没有偏移量MNIST_LABEL_OFFSET
        *label_onehot = label_onehot_all + offset_label;
    }
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
                offset += snprintf(img + offset, 4096 - offset, "%u ", (unsigned int)(images[ MNIST_IMG_OFFSET + MNIST_HEIGHT * MNIST_WIDTH * i + j * MNIST_WIDTH + k]));
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


    