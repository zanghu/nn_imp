#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "debug_macros.h"

float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

/**
 * @brief 基于概率矩阵和真值矩阵计算分类准确率
 * @param p 一个样本batch的概率矩阵，p[i*k]-p[i*(k+1)-1]表示样本i在k各类别上的概率分布
 * @param gt_onehot 数据样本的one_hot表示的类别
 */
int getAccuracyFollowProbilityAndGroundtruth(int *n_success, const float *p, const unsigned char *gt_onehot, int b, int k)
{
    CHK_NIL(n_success);
    CHK_NIL(p);
    CHK_NIL(gt_onehot);
    CHK_ERR((b > 0)? 0: 1);
    CHK_ERR((k > 0)? 0: 1);

    int n_succ = 0;
    int i, j;
    for (i = 0; i < b; ++i) {
        float p_max = FLT_MIN;
        int j_max = -1;
        for (j = 0; j < k; ++j) {
            if (p[i * k + j] >= p_max) {
                p_max = p[i * k + j];
                j_max = j;
            }
        }
        if (gt_onehot[j_max] == 1) {
            ++n_succ;
        }
    }
    *n_success = n_succ;
    return SUCCESS;
}