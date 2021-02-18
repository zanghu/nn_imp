# Neural Network

C语言实现的神经网络（Neural Network）

## 1.计划

第1阶段：
实现 MLP、CONV、GAN、LSTM 等基本模型

第2阶段：
分析、研究新的模型和方法

## 2.进度

### 2021.02.18
1.修改探针类和层类增加了运行过程中导出数据到指定目录的功能；

2.修改MNIST测试案例导出参数和梯度数据；

3.TODO: 与python版训练过程的参数和数据比较，找出权重上溢问题的原因。

### 2021.02.17
1.修复数据集最后一个batch样本不足batch_size报错的问题；

2.增加日志记录功能；

3.Layer类支持命名，用于DEBUG；

4.现在MNIST数据集直接提供float32和onehot的数据和类标，无需用户手动转换；

5.修改了MNIST数据集的normalization预处理方法，参考了网上公开资料；

6.验证python版本与处理结果与C版本处理结果数据的一致性，误差约为5e-7。

7.TODO: 目前尚存在训练过程梯度和参数上溢的问题。

### 2021.02.16
1.MNIST数据集增加对训练迭代数据、onehot类标的支持；

2.增加了探针，方便数据统计；

3.mlp模型集成测试案例补充了训练迭代过程。

4.现在训练参数不再和模型绑定，而是再每一轮迭代中ing和探针一起作为参数传给模型；

5.测试循环迭代，目前初步调通。

6.测试中尚存的问题：(1)最后一个batch样本数不足时会出现SIGSEGV；(2)一个epoch的不同iter下ce_cost没有明显下降。

### 2021.02.15
1.重新设计目录结构；

2.新增MNIST数据集读取，补充测试案例。

### 2021.02.14
1.将每一层的线性变换和非线性变换拆分成两个独立的层；

2.完成代价函数；

3.完成网络层间关联；

4.Layer, Cost现在分别是层类和代价类的基类；

5.完成交叉熵代价CECost类和SoftMaxLayer类；

6.统一函数返回值为错误码；

7.新增了XXX_GOTO宏，配合函数内使用goto进行异常处理时的资源释放；

8.增加了测试用例。

### 2021.02.13
1.完成全连接层反向传播的参数梯度计算;

2.完成全连接层forward、backward、update方法；

3.引入YOLOv2的GEMM计算。

### 2021.02.12
1.创建仓库。