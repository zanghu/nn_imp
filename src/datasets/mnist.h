/**
 * @see http://yann.lecun.com/exdb/mnist/
 */
#pragma once

// mnist binary meta info
#define MNIST_N_CLASSES (10)
#define MNIST_WIDTH (28)
#define MNIST_HEIGHT (28)
#define MNIST_N_TRAIN (60000)
#define MNIST_N_TEST (10000)
#define MNIST_IMG_OFFSET (16)
#define MNIST_LABEL_OFFSET (8)
#define MNIST_ELEM_SIZE (sizeof(unsigned char))
#define MNIST_SAMPLE_SIZE (MNIST_WIDTH * MNIST_HEIGHT * MNIST_ELEM_SIZE)

struct MNIST
{
    unsigned char *train_images;
    unsigned char *train_labels;
    unsigned char *test_images;
    unsigned char *test_labels;
    unsigned char *train_labels_onehot;
    unsigned char *test_labels_onehot;
};

// 用法：声明栈变量data, load(&data, src_dir)
int loadMnist(struct MNIST *data, const char *src_dir);
void freeMnist(struct MNIST *data);
int dumpMnistToNumpyTxt(const struct MNIST *data, const char *dst_dir, unsigned int start, unsigned int end);
int getMnistTrainBatchNumber(int *n_batches, int batch_size);
int getMnistNthBatch(const unsigned char *(*data), const unsigned char *(*label), const unsigned char *(*label_onehot), int *n_samples, const char *type, const struct MNIST *mnist, int n_use, int batch_size, int batch_idx);