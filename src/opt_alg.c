#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opt_alg.h"
#include "debug_macros.h"

int checkUpdateArgs(const struct UpdateArgs *args)
{
    CHK_NIL(args);

    CHK_ERR((args->batch_size > 0)? 0: 1);
    CHK_ERR((args->lr > 0)? 0: 1);
    CHK_ERR((args->momentum >= 0)? 0: 1);
    return SUCCESS;
}