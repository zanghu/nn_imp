#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_macros.h"
#include "math_utils.h"
#include "matrix.h"



struct Layer {
    struct Matrix *w;
    struct Matrix *b;
    struct Matrix *w_grad;
    struct Matrix *b_grad;
    int (*f)(struct Matrix *, struct Matrix *);
};

struct Layer *createLayer(int n_in, int n_out)
{
    if (n_in <= 0) return NULL;
    if (n_in <= 0) return NULL;

    struct Layer *layer = calloc(1, sizeof(struct Layer));
    if (layer == NULL) {
        ERR_MSG("calloc failed, detail: %s\n", ERRNO_DETAIL(errno));
        return NULL;
    }

    layer->w = createMatrix(1, n_in, n_out, 1);
    if (layer->w == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        goto err_end;
    }
    initMatrixParameterAsWeight(layer->w);

    layer->b = createMatrix(1, 1, n_out, 1);
    if (layer->w == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        goto err_end;
    }
    initMatrixParameterAsBias(layer->b);

    layer->w_grad = createMatrix(1, n_in, n_out, 1);
    if (layer->w_grad == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        goto err_end;
    }

    layer->b_grad = createMatrix(1, 1, n_out, 1);
    if (layer->b_grad == NULL) {
        ERR_MSG("createMatrix() failed, error.\n");
        goto err_end;
    }

    return layer;

err_end:
    if (layer) {
        free(layer->b_grad);
        free(layer->w_grad);
        free(layer->b);
        free(layer->w);
    }
    free(layer);
    return NULL;
}

void destroyLayer(struct Layer *layer)
{
    if (layer) {
        free(layer->b_grad);
        free(layer->w_grad);
        free(layer->b);
        free(layer->w);
    }
    free(layer);
}

int getLayerInputNeuronNumber(const struct Layer *layer)
{
    CHK_NIL(layer);
    return getMatrixCol(layer->w);
}

int getLayerOutputNeuronNumber(const struct Layer *layer)
{
    CHK_NIL(layer);
    return getMatrixRow(layer->w);
}

int forwardLayer(struct Matrix *output, struct Matrix *hidden, const struct Layer *layer, const struct Matrix *input)
{
    CHK_NIL(output);
    CHK_NIL(hidden);
    CHK_NIL(layer);
    CHK_NIL(input);

    CHK_ERR(linearMatrix(hidden, layer->w, input, layer->b));
    CHK_ERR(activateMatrix(output, hidden, "logistic"));

    return SUCCESS;
}

int backwardLayer(struct Matrix *delta_new, const struct Matrix *output, const struct Layer *layer, const struct Matrix *delta)
{
    CHK_NIL(delta_new);
    CHK_NIL(output);
    CHK_NIL(layer);
    CHK_NIL(delta);

    CHK_ERR(deactivateMatrix(delta_new, output, delta, "logistic"));
    CHK_ERR(linearMatrix(delta_new, layer->w, delta, layer->b));

    return SUCCESS;
}

