#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "io_utils.h"

int main()
{
    float *blob_w_784x256 = calloc(784 * 256, sizeof(float));
    CHK_NIL(blob_w_784x256);
    float *blob_b_256 = calloc(256, sizeof(float));
    CHK_NIL(blob_b_256);

    float *blob_w_256x128 = calloc(256 * 128, sizeof(float));
    CHK_NIL(blob_w_256x128);
    float *blob_b_128 = calloc(128, sizeof(float));
    CHK_NIL(blob_b_128);

    float *blob_w_128x10 = calloc(128 * 10, sizeof(float));
    CHK_NIL(blob_w_128x10);
    float *blob_b_10 = calloc(10, sizeof(float));
    CHK_NIL(blob_b_10);


    int n_loaded = 0;
    CHK_ERR(loadtxtBlobFloat32(&n_loaded, "txt/NET_L00_W_784x256.txt", blob_w_784x256, 784 * 256));
    fprintf(stdout, "n_loaded = %d\n", n_loaded);
    CHK_ERR((n_loaded == 784 * 256)? 0: 1);
    CHK_ERR(loadtxtBlobFloat32(&n_loaded, "txt/NET_L00_b_256.txt", blob_b_256, 256));
    fprintf(stdout, "n_loaded = %d\n", n_loaded);
    CHK_ERR((n_loaded == 256)? 0: 1);

    CHK_ERR(loadtxtBlobFloat32(&n_loaded, "txt/NET_L02_W_256x128.txt", blob_w_256x128, 256 * 128));
    fprintf(stdout, "n_loaded = %d\n", n_loaded);
    CHK_ERR((n_loaded == 256 * 128)? 0: 1);
    CHK_ERR(loadtxtBlobFloat32(&n_loaded, "txt/NET_L02_b_128.txt", blob_b_128, 128));
    fprintf(stdout, "n_loaded = %d\n", n_loaded);
    CHK_ERR((n_loaded == 128)? 0: 1);

    CHK_ERR(loadtxtBlobFloat32(&n_loaded, "txt/NET_L04_W_128x10.txt", blob_w_128x10, 128 * 10));
    fprintf(stdout, "n_loaded = %d\n", n_loaded);
    CHK_ERR((n_loaded == 128 * 10)? 0: 1);
    CHK_ERR(loadtxtBlobFloat32(&n_loaded, "txt/NET_L04_b_10.txt", blob_b_10, 10));
    fprintf(stdout, "n_loaded = %d\n", n_loaded);
    CHK_ERR((n_loaded == 10)? 0: 1);

    free(blob_w_784x256);
    free(blob_b_256);
    free(blob_w_256x128);
    free(blob_b_128);
    free(blob_w_128x10);
    free(blob_b_10);

    fprintf(stdout, "all finish.\n");
    return 0;
}