#pragma once

struct UpdateArgs
{
    int batch_size;
    float lr;
    float momentum;
    float decay;
    int n_epochs;
    int cur_epoch; // 当前的epoch
    int cur_iter; // 当前的iter
    int cur_samples;
};

int checkUpdateArgs(const struct UpdateArgs *args);