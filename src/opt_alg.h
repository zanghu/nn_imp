#pragma once

struct UpdateArgs
{
    int batch_size;
    float lr;
    float momentum;
    float decay;
    int n_epochs;
};

int checkUpdateArgs(const struct UpdateArgs *args);