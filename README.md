# TensorflowProjects
Some projects using Tensorflow

手打一些书上的代码来进行TensorFlow的学习
## fizzbuzz.py:
拟合出这样的规律：对于15的倍数输出fizzbuzz，3的倍数输出fizz，5的倍数输出buzz，其他数字直接输出
通过这个项目来初步了解TensorFlow：包括特征和label如何编码，神经网络各层之间的连接
## circles.py
拟合出这样的规律：对于数字的十进制表示的末四位（不足4位的用0补齐）输出其中圈的个数：每有1位是0或6或9则圈数+1，每有一位是8则圈数+2
通过这个项目来初步了解调参，例如增加神经网络的层数和神经元个数、学习率、提取特征的方式
调参得到的对比和心得写在注释里面
## mnist,py
mnist数据集的数字识别问题
增加了l2正则项、滑动平均类用于准确率的测试、学习率的衰减
## mnist_interface等三个文件
增加了模型的保存功能；把程序进一步模块化、也把训练和测试分开
