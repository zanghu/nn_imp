#pragma once

int loadMnist(unsigned char **train_images, unsigned char **train_labels, unsigned char **test_images, unsigned char **test_labels, const char *src_dir);
int dumpMnistNumpyTxt(const char *dst_dir, const unsigned char *images, unsigned int start, unsigned int end);