#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "mnist.h"

int main()
{
    struct MNIST mnist;
    CHK_ERR(loadMnist(&mnist, "/home/zanghu/data_base/mnist"));

    CHK_ERR(dumpMnistTransformed(&mnist, "txt"));
    fprintf(stdout, "all finish\n");

    freeMnist(&mnist);
    return 0;
}