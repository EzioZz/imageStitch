数字图像处理课作业

无人机航拍图像拼接

算法思路

1. harris角点查找特征点
2. 非极大值抑制
3. 航拍图像没有严重的尺度旋转变化，使用了berief描述子
4. 使用RANSAC求H
5. 拼接



使用方法

cmake需要配置opencv库和eigen库



实现结果

![KTaAVP.jpg](https://s2.ax1x.com/2019/10/31/KTaAVP.jpg)