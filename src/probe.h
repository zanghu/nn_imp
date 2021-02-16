/**
 * @brief 探针相当于用户插入到训练过程中的信息收集器
 *        探针时模型运行数据和用户统计数据之间的媒介
 *        模型只负责无状态（相对于每个iter、每个epoch而言）、元数据的计算，计量避免与用户层的统计指标（计算规则）耦合
 *        用户负责选择合适的时机收集元数据，再基于元数据需要的统计指标、记录日志等等。
 * 
 *        探针的意义：将用户层运行指标的收集和计算逻辑与训练信息计算解耦
 *        （1）以探针为中介，用户从探针读取数据，模型向探针添加数据，数据收集过程和模型运行过程互不影响；
 *        （2）用户需要的（基于元数据计算的）统计指标发生变化时、日志发生变化时，无需修改模型代码。
 *
 *        确保探针中的内容是“元数据”，例如：
 *        如果知道模型对样本的分类概率和样本的类别真值，即可计算出样本的分类结果、分类正确率、分类正确数和错误数
 *        因此，探针只需获取样本分类概率，其他用户可自行计算的指标不应该放在探针中。
 */
#pragma once

struct Probe
{
    int sw_p_class; // 计算分类概率
    float *p_class;
    int sw_ce_cost; // 计算交叉熵代价
    float ce_cost;
    int sw_sq_cost; // 计算平方损失代价
    float sq_cost;
};