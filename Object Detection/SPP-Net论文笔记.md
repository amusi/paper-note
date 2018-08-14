SPP-Net 论文笔记


**关于最新最全的目标检测论文，可以查看[awesome-object-detection](https://github.com/amusi/awesome-object-detection)**

《Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition》

TPAMI 2015

ECCV 2014

- arxiv: https://arxiv.org/abs/1406.4729

- slides: https://dl.dropboxusercontent.com/s/vlyrkgd8nz8gy5l/fast-rcnn.pdf?dl=0

- github(Caffe): https://github.com/ShaoqingRen/SPP_net
- github(PyTorch): https://github.com/yueruchen/sppnet-pytorch
- 


空间金字塔池化可以把任何尺度的图像的卷积特征转化成相同维度，这不仅可以让CNN处理任意尺度的图像，还能避免cropping和warping操作，导致一些信息的丢失，具有非常重要的意义。 

一般的CNN都需要输入图像的大小是固定的，这是因为全连接层的输入需要固定输入维度，但在卷积操作是没有对图像尺度有限制，所有作者提出了空间金字塔池化，先让图像进行卷积操作，然后转化成维度相同的特征输入到全连接层，这个可以把CNN扩展到任意大小的图像 

空间金字塔池化的思想来自于Spatial Pyramid Model，它一个pooling变成了多个scale的pooling。**用不同大小池化窗口作用于卷积特征，我们可以得到1X1,2X2,4X4的池化结果**，由于conv5中共有256个过滤器，所以得到1个256维的特征，4个256个特征，以及16个256维的特征，然后把这21个256维特征链接起来输入全连接层，通过这种方式把不同大小的图像转化成相同维度的特征。 


# 讨论

## 为什么全连接层的输入需要固定维度？

我们知道SPP-Net的提出一方面解决了固定的输入图像大小问题，即输入图像大小不受限制。因为SPP-Net解决了全连接层的输入需要固定维度的问题。

那么为什么全连接层的输入需要固定维度呢？

答：全连接层的计算其实相当于输入的特征图数据矩阵和全连接层权值矩阵进行内积。在配置一个网络时，全连接层的参数维度是固定的，所以两个矩阵要能够进行内积，则输入的特征图的数据矩阵维数也需要固定[X1][X2]。


# 参考阅读

- [SPPNet论文翻译——空间金字塔池化](http://www.dengfanxin.cn/?p=403)
- [SPP-Net论文详解](https://blog.csdn.net/v1_vivian/article/details/73275259)
- 【X1】[为什么全连接层输入需要固定尺度](https://blog.csdn.net/tigerda/article/details/78652447?%3E)
- 【X2】[对含有全连接层的网络输入数据大小固定问题的理解](https://blog.csdn.net/z13653662052/article/details/80252696)