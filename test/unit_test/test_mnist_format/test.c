#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "mnist.h"

int main()
{
    unsigned char *train_images = NULL;
    unsigned char *train_labels = NULL;
    unsigned char *test_images = NULL;
    unsigned char *test_labels = NULL;
    CHK_ERR(loadMnist(&train_images, &train_labels, &test_images, &test_labels, "/home/zanghu/data_base/mnist"));
    CHK_ERR(dumpMnistNumpyTxt("txt", train_images, 0, 10));
    CHK_ERR(dumpMnistNumpyTxt("txt", test_images, 200, 210));
    
    return 0;
}