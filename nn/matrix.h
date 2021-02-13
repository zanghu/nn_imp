#pragma once

#include "activations.h"

struct Matrix;

struct Matrix *createMatrix(int batch_size, int row, int col, int channel);

void destroyMatrix(struct Matrix *matrix);

void initMatrixParameterAsWeight(struct Matrix *matrix);

void initMatrixParameterAsBias(struct Matrix *matrix);

int getMatrixRow(const struct Matrix *matrix);

int getMatrixCol(const struct Matrix *matrix);

int getMatrixChannel(const struct Matrix *matrix);

int getMatrixBatch(const struct Matrix *matrix);

//struct Matrix *mulMatrix(const struct Matrix *w, const struct Matrix *x, const struct Matrix *b);

int linearMatrix(struct Matrix *y, const struct Matrix *w, const struct Matrix *x, const struct Matrix *b);

int activateMatrix(struct Matrix *y, const struct Matrix *x, enum ActivationType act_type);

int deactivateMatrix(const struct Matrix *delta, const struct Matrix *output, enum ActivationType act_type);

int mulMatrix(struct Matrix *z, const struct Matrix *x, const struct Matrix *y);

int mulMatrixPointwiseAndSum(struct Matrix *z, const struct Matrix *x, const struct Matrix *y);

int addMatrix(struct Matrix *x, struct Matrix *y, float lr, float momentum);