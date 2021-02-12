#pragma once

#include <math.h>

float rand_uniform(float min, float max);

static inline float logistic_activate(float x){
    return 1./(1. + exp(-x));
}