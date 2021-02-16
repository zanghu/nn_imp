#pragma once

int transformOnehot(void **onehot, const char *dtype_onehot, void *orin, const char *dtype_orin, int n_samples, int n_classes);
int transformToFloat32FromUint8(float *dst, const unsigned char *src, int n_elems, int norm_const);