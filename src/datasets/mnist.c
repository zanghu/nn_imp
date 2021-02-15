#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "debug_macros.h"

// mnist original file names
static const char *s_train_images = "train-images-idx3-ubyte";
static const char *s_train_labels = "train-labels-idx1-ubyte";
static const char *s_test_images = "t10k-images-idx3-ubyte";
static const char *s_test_labels = "t10k-labels-idx1-ubyte";

// mnist file meta info
#define IMG_WIDTH (28)
#define IMG_HEIGHT (28)
#define N_TRAIN (60000)
#define N_TEST (10000)
#define IMG_OFFSET (16)
#define LABEL_OFFSET (8)

// mnist file size
static const unsigned int s_size_train_img = IMG_OFFSET + IMG_WIDTH * IMG_HEIGHT * N_TRAIN;
static const unsigned int s_size_train_label = LABEL_OFFSET + N_TRAIN * 1;
static const unsigned int s_size_test_img = IMG_OFFSET + IMG_WIDTH * IMG_HEIGHT * N_TEST;
static const unsigned int s_size_test_label = LABEL_OFFSET + N_TEST * 1;

int loadMnist(unsigned char **train_images, unsigned char **train_labels, unsigned char **test_images, unsigned char **test_labels, const char *src_dir)
{
    CHK_NIL(train_images);
    CHK_NIL(train_labels);
    CHK_NIL(test_images);
    CHK_NIL(test_labels);
    CHK_NIL(src_dir);

    struct timeval t0, t1, t2;
    CHK_ERR(gettimeofday(&t0, NULL));

    int fd = -1;
    int i;
    unsigned char *results[4] = {NULL, NULL, NULL, NULL};
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
    CHK_ERR(gettimeofday(&t1, NULL));
    timersub(&t1, &t0, &t2);
    fprintf(stdout, "MNIST data read finish, time elapsed: %lu.%06lus\n", t2.tv_sec, t2.tv_usec);

    *train_images = results[0];
    *train_labels = results[1];
    *test_images = results[2];
    *test_labels = results[3];

    return SUCCESS;

err_end:
    for (i = 0; i < 4; ++i) {
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

// 选出若干张图片，导出为numpy.savetxt的格式. 该方法用来验证测试原始文件中图形格式
int dumpMnistNumpyTxt(const char *dst_dir, const unsigned char *images, unsigned int start, unsigned int end)
{
    CHK_NIL(dst_dir);
    CHK_NIL(images);
    CHK_ERR((start >= 0)? 0: 1);
    CHK_ERR((start < end)? 0: 1);

    struct timeval t0, t1, t2;
    CHK_ERR(gettimeofday(&t0, NULL));

    int fd;
    char path[1024];
    char img[4096];
    int i, j, k;
    for (i = start; i < end; ++i) {
        snprintf(path, 1024, "%s/%05d.txt", dst_dir, i);
        fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
        if (fd == -1) {
            ERR_MSG("open() failed, path: %s, err_detail: %s, error.\n", path, ERRNO_DETAIL(errno));
            return ERR_COD;
        }
        int offset = 0;
        for (j = 0; j < IMG_HEIGHT; ++j) {
            for (k = 0; k < IMG_WIDTH; ++k) {
                offset += snprintf(img + offset, 4096 - offset, "%u ", (unsigned int)(images[IMG_OFFSET + IMG_HEIGHT * IMG_WIDTH * i + j * IMG_WIDTH + k]));
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
    