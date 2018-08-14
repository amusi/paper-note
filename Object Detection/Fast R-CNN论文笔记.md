Fast R-CNN 论文笔记


**关于最新最全的目标检测论文，可以查看[awesome-object-detection](https://github.com/amusi/awesome-object-detection)**

《Fast R-CNN》

ICCV 2015

- arxiv: http://arxiv.org/abs/1504.08083

- slides: https://dl.dropboxusercontent.com/s/vlyrkgd8nz8gy5l/fast-rcnn.pdf?dl=0

- slides: http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf

- github(Official): https://github.com/rbgirshick/fast-rcnn

- github(COCO-branch): https://github.com/rbgirshick/fast-rcnn/tree/coco

- webcam demo: https://github.com/rbgirshick/fast-rcnn/pull/29

- github("Fast R-CNN in MXNet"): https://github.com/precedenceguo/mx-rcnn

- github: https://github.com/mahyarnajibi/fast-rcnn-torch

- github: https://github.com/apple2373/chainer-simple-fast-rnn

- github: https://github.com/zplizzi/tensorflow-fast-rcnn

- video: https://www.youtube.com/watch?v=xzw3lcdllOU

推荐阅读

- [（推荐）Fast R-CNN论文详解](https://blog.csdn.net/wopawn/article/details/52463853)

- [Fast-RCNN解析：训练阶段代码导读](http://blog.csdn.net/linj_m/article/details/48930179)

- [Region of interest pooling explained](https://deepsense.ai/region-of-interest-pooling-explained/)



# 创新点

在介绍Fast R-CNN的创新点之前，我们先来介绍一下它的爸爸：[R-CNN](https://blog.csdn.net/amusi1994/article/details/81081023) 存在的不足：

1. Multiple-stage pipeline：训练分为多个阶段，region proposals、ConvNet、SVM、BB Regression。

2. 训练耗时，占有磁盘空间大。卷积出来的特征数据还要保持至本地磁盘。

3. 重复计算，目标检测速度慢。通过selective search 提取近2000左右的候选框，即今2000个ROI，RCNN对每个ROI，都跑一遍CNN，计算量很大，而且其实这些ROI之间是有overlap，显然有大量计算是重复的。（所以SPP-Net和Fast-RCNN对其进行改进）


简单来说，最大的问题就是速度慢！"深度学习"范儿也不足。于是R-CNN原作者 Ross Girshick就进一步提出了R-CNN的改进版：Fast R-CNN。


但更准确来说，Fast R-CNN是基于R-CNN和[SPPnets](https://arxiv.org/abs/1406.4729)进行的改进。SPPnets，其创新点在于计算整幅图像的the shared feature map，然后根据object proposal在shared feature map上映射到对应的feature vector（就是不用重复计算feature map了）。当然，SPPnets也有缺点：和R-CNN一样，训练是多阶段（multiple-stage pipeline）的，速度还是不够"快"，特征还要保存到本地磁盘中。


Fast R-CNN创新点

1. 只对整幅图像进行一次特征提取，避免R-CNN中的冗余特征提取
2. 用RoI pooling层替换最后一层的max pooling层，同时引入建议框数据，提取相应建议框特征
3. Fast R-CNN网络末尾采用并行的不同的全连接层，可同时输出分类结果和窗口回归结果，实现了end-to-end的多任务训练【建议框提取除外】，也不需要额外的特征存储空间【R-CNN中的特征需要保持到本地，来供SVM和Bounding-box regression进行训练】
4. 采用SVD对Fast R-CNN网络末尾并行的全连接层进行分解，减少计算复杂度，加快检测速度。

拥有上述4个创新点，那么Fast R-CNN究竟得到哪些优势呢？

Fast R-CNN 优势

1. 比R-CNN和SPPnets更高的检测质量（mAP）
2. Single stage training
3. 训练可以更新所有网络层
4. 减少内存资源。不需要将特征保存至本地磁盘
5. 


# RoI Pooling层详解

RoI Pooling 是Pooling层的一种，而且是针对RoI的Pooling，其特点是输入特征图尺寸不固定，但是输出特征图尺寸固定（如7x7）。


什么是ROI呢？

ROI是Region of Interest的简写，一般是指图像上的区域框，但这里指的是由Selective Search提取的候选框”；


![Fast R-CNN architecture](https://img-blog.csdn.net/20180119150020632?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbGFucmFuMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

往往经过rpn后输出的不止一个矩形框，所以这里我们是对多个ROI进行Pooling。

## RoI Pooling的输入
输入有两部分组成： 
1. 特征图（feature map）：指的是上面所示的特征图，在Fast RCNN中，它位于RoI Pooling之前，在Faster RCNN中，它是与RPN共享那个特征图，通常我们常常称之为“share_conv”； 

2. RoIs，其表示所有RoI的N*5的矩阵。其中N表示RoI的数量，第一列表示图像index，其余四列表示其余的左上角和右下角坐标。

在Fast RCNN中，指的是Selective Search的输出；在Faster RCNN中指的是RPN的输出，一堆矩形候选框，形状为1x5x1x1（4个坐标+索引index），其中值得注意的是：坐标的参考系不是针对feature map这张图的，而是针对原图的（神经网络最开始的输入）。其实关于ROI的坐标理解一直很混乱，到底是根据谁的坐标来。其实很好理解，我们已知原图的大小和由Selective Search算法提取的候选框坐标，那么根据"映射关系"可以得出特征图（featurwe map）的大小和候选框在feature map上的映射坐标。至于如何计算，其实就是比值问题，下面会介绍。所以这里把ROI理解为原图上各个候选框（region proposals），也是可以的。

注：说句题外话，由Selective Search算法提取的一系列可能含有object的boudning box，这些通常称为region proposals或者region of interest（ROI）。


## RoI的具体操作

1. 根据输入image，将ROI映射到feature map对应位置

    注：映射规则比较简单，就是把各个坐标除以“输入图片与feature map的大小的比值”，得到了feature map上的box坐标

2. 将映射后的区域划分为相同大小的sections（sections数量与输出的维度相同）

3. 对每个sections进行max pooling操作
 

这样我们就可以从不同大小的方框得到固定大小的相应 的feature maps。值得一提的是，输出的feature maps的大小不取决于ROI和卷积feature maps大小。ROI pooling 最大的好处就在于极大地提高了处理速度。


## RoI Pooling的输出
输出是batch个vector，其中batch的值等于RoI的个数，vector的大小为channel * w * h；RoI Pooling的过程就是将一个个大小不同的box矩形框，都映射成大小固定（w * h）的矩形框；


## ROI Pooling示例

![](https://deepsense.ai/wp-content/uploads/2017/02/1.jpg)

![](https://deepsense.ai/wp-content/uploads/2017/02/2.jpg)

![](https://deepsense.ai/wp-content/uploads/2017/02/3.jpg)

![](https://deepsense.ai/wp-content/uploads/2017/02/output.jpg)

![](https://deepsense.ai/wp-content/uploads/2017/02/roi_pooling-1.gif)


## RoI总结

1. 用于目标检测任务
2. 允许我们对CNN中的feature map进行reuse
3. 可以显著加速training和testing速度
4. 允许end-to-end的形式训练目标检测系统


上面参考：

- [Region of interest pooling explained](https://deepsense.ai/region-of-interest-pooling-explained/)

- [RoI pooling层解析](https://blog.csdn.net/lanran2/article/details/60143861/)

- [RoI Pooling层详解](https://blog.csdn.net/auto1993/article/details/78514071)

- [RoI Pooling Layer in Caffe](https://github.com/rbgirshick/caffe-fast-rcnn/blob/bcd9b4eadc7d8fbc433aeefd564e82ec63aaf69c/src/caffe/layers/roi_pooling_layer.cu)

- [RoI Pooling Layer in TensorFlow](https://github.com/deepsense-ai/roi-pooling)