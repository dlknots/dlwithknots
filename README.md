# dpwithknots
系北京师范大学北创项目——验证纽结等价的深度学习方法项目组成果
## introduction
- 尝试设计深度学习算法寻找纽结间的形变路径，进而验证对任意给定的两个纽结，它们是否等价
- 本项目最终实现了一些简单的平凡纽结间的等价性验证
## files & codes
- net.ipynb —— 设计了全连接神经网络net1与基于realNVP的神经网络net2，经测试，net2性能更佳
### figure2mat
- figure2mat.py —— 实现纽结图转化为三维参数曲线并将坐标写入mat文件，便于读取
- Transformation.py —— 调用figure2mat，进行转化
- pics —— 五个平凡纽结的纽结图
## problems
- net2仅能实现一些简单的平凡纽结间的等价性验证，在处理较复杂的纽结变换的能力有待优化
### 如果有任何建议或疑问，欢迎联系我们：dpknot_bnu@163.com
