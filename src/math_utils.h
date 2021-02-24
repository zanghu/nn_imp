#pragma once

#include <math.h>

float rand_uniform(float min, float max);
int getAccuracyFollowProbilityAndGroundtruth(int *n_success, const float *p, const unsigned char *gt_onehot, int b, int k);