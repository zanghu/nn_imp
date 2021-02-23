#pragma once

int savetxtMatrixFlot32(const char *pth, const float *data, int n_samples, int n_features);
int savetxtMatrixUint8(const char *pth, const unsigned char *data, int n_samples, int n_features);
int loadtxtBlobFloat32(int *n_loaded, const char *pth, float *blob, int size);