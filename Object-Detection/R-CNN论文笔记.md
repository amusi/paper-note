# R-CNN

**关于最新最全的目标检测论文，可以查看[awesome-object-detection](https://github.com/amusi/awesome-object-detection)**

《Rich feature hierarchies for accurate object detection and semantic segmentation》 

CVPR 2014
 
- arxiv: http://arxiv.org/abs/1311.2524

- supp: http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf

- slides: http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf

- slides: http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf

- github(caffe): https://github.com/rbgirshick/rcnn

- notes: http://zhangliliang.com/2014/07/23/paper-note-rcnn/

- caffe-pr("Make R-CNN the Caffe detection example"): https://github.com/BVLC/caffe/pull/482

推荐阅读

- [（推荐阅读）R-CNN论文详解](https://blog.csdn.net/WoPawn/article/details/52133338)
- [R-CNN详解](https://blog.csdn.net/shenxiaolu1984/article/details/51066975)
- [R-CNN学习总结](https://zhuanlan.zhihu.com/p/30316608)
- [基于R-CNN的物体检测](https://blog.csdn.net/hjimce/article/details/50187029)


R-CNN：Regions + CNN

# 创新点
- 使用CNN（ConvNet）对 region proposals 计算 feature vectors。从经验驱动特征（SIFT、HOG）到数据驱动特征（CNN feature map），提高特征对样本的表示能力。

- 采用大样本下（ILSVRC）有监督预训练和小样本（PASCAL）微调（fine-tuning）的方法解决小样本难以训练甚至过拟合等问题。

注：ILSVRC其实就是众所周知的ImageNet的挑战赛，数据量极大；PASCAL数据集（包含目标检测和图像分割等），相对较小。

# 结果

在VOC2012中，将mAP（mean average percision）提高了30%以上

先看一下 PASCAL VOC历年（2007~2012）的检测冠军，可见DPM的统治力有多强大！（刚荣获CVPR 2018 Longuet-Higgins Prize）
![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCEd161f02f2c52e9756e424c6ead1e7f61/51027)


但直到2013年 R-CNN的横空出世，一切都被打破了！
之后目标检测领域就进入 R-CNN系列的疯狂统治中......
![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE692bb03739ff9cfb2c8938fbfa93ab93/51032)


# R-CNN流程

![R-CNN Pipeline](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/more_images/RCNNSimple.png)

![R-CNN Pipeline](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE53d72f0dc19d4808772ed4739a56fbf1/50897)
图像来源: r-cnn-ilsvrc2013-workshop.pdf


题外话：R-CNN作为R-CNN系列的第一代算法，其实没有过多的使用“深度学习”思想，而是将“深度学习”和传统的“计算机视觉”的知识相结合。

比如pipeline中的第二步和第四步其实就属于传统的“计算机视觉”技术。使用selective search提取region proposals，使用SVM实现分类。

而R-CNN系列的第三代算法：Faster R-CNN是使用RPN来提取 region proposals，而使用softmax实现分类。Faster R-CNN才是纯正的深度学习算法。

![R-CNN CV and DL](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE5c18d00408bac263ba6bf59095a1a514/50958)
图像来源: r-cnn-ilsvrc2013-workshop.pdf

原论文中R-CNN pipeline只有4个步骤，光看上图无法深刻理解R-CNN处理机制，下面结合图示补充相应文字

1. 预训练模型。选择一个预训练 （pre-trained）神经网络（如AlexNet、VGG）

2. 重新训练全连接层。使用需要检测的目标重新训练（re-train）最后全连接层（connected layer）

3. 提取 proposals并计算CNN 特征。利用选择性搜索（Selective Search）算法提取所有proposals（大约2000个 / image），调整（resize/warp）它们成固定大小，以满足 CNN输入要求（因为全连接层的限制），然后将feature map 保存到本地磁盘。

![](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/more_images/Step3RCNN.png)

4. 训练SVM。利用feature map 训练SVM来对目标和背景进行分类（每个类一个二进制SVM）

5. 边界框回归（Bounding boxes Regression）。训练将输出一些校正因子的线性回归分类器

![](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/more_images/Step5RCNN.png)

看完上述，如果你可能会存在疑问？深度学习明明分训练和测试两个阶段，为什么上述没有说明两者的区别？（很多解读性文章都忽略了这一点）

放心，作为小白而言，当然要一步步搞懂这个内容，下面将单独介绍训练过程和测试过程。

# R-CNN 训练过程

1. 在大型数据集（ImageNet，即ILSVRC竞赛）上预训练用于图像分类的CNN。

2. 在小型目标数据集（PASAC VOC）上微调（fine-tuning）CNN。

3. 利用Selective Search提取2k多个region proposals并warp到同一size，然后输入训练好的CNN网络中，最后对每类训练SVM（因为SVM本身是二分类，如何通过对每类训练SVM而实现多分类呢？这个需要你百度自行了解。提示两个名词：one-versus-rest和one-versus-one）

训练过程如下所示

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE05119d37ca66f09a6d973b0e2f3b7e62/50904)
图像来源: r-cnn-ilsvrc2013-workshop.pdf

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE56bb53543d1a2171396bacfdf4736ca3/50953)
图像来源: r-cnn-ilsvrc2013-workshop.pdf

那么上述所说的CNN网络究竟长什么样子呢？

答：其实和AlexNet差不多，这里就不赘述，下面附上2012年ImageNet夺冠的AlexNet网络。

![AlexNet](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE930587cf3df17914a3de82da74d32941/51106)



# R-CNN 测试过程
1. 输入一张多目标图像，采用selective search算法提取约2000个建议框；

![用selective search](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE52cf1faafbb819f576f697242dc64a0d/51052)

2. 先在每个建议框周围加上16个像素值，然后裁剪（crop），再直接 scale 为227×227的大小（AlexNet网络输入图像大小：227x227）；将所有建议框像素减去该建议框像素平均值后【预处理操作】，再依次将每个227×227的建议框输入AlexNet CNN网络获取4096维的特征【比以前的人工经验特征低两个数量级】，2000个建议框的CNN特征组合成2000×4096维矩阵；

![Dilate proposal](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE296348fb4b54552a19b8c13fe5c294b2/51056)

![Corp and Scale](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCEdb4360abd764337af47cd76aa8dac485/51058)

![CNN](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCEcb6657350c3277f65ece9a14a41c2964/51068)


3. 将2000×4096维特征与20个SVM组成的权值矩阵4096×20相乘【20种分类，SVM是二分类器，则有20个SVM】，获得2000×20维矩阵表示每个建议框是某个物体类别的得分；分别对上述2000×20维矩阵中每一列即每一类进行非极大值抑制剔除重叠建议框，得到该列即该类中得分最高的一些建议框；

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCEb44020f399f5cc7aa866bf82255ab0f0/51071)


4. 分别用20个回归器对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别的修正后的得分最高的bounding box。

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE2e20ca98695989a2ba5093ee22b2ca3c/51074)

检测结果如下图所示

![R-CNN results on VOC 2007](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE20b8293af68e781034d080ceb77b52a4/51080)

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCEfc308722c1dc62c377309c80f969cdad/51084)

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE78f6248d41559b94d8eabe8c5d6ab1ef/51086)

# 细节分析
1. selective search 
采取过分割手段，将图像分割成小区域，再通过颜色直方图，梯度直方图相近等规则进行合并，最后生成约2000个建议框的操作，具体见[博客](https://blog.csdn.net/mao_kun/article/details/50576003)。

2. 为什么要将建议框变形为227×227？怎么做？ 
本文采用AlexNet CNN网络进行CNN特征提取，为了适应AlexNet网络的输入图像大小：227×227，故将所有建议框变形为227×227。 

那么问题来了，如何进行变形操作呢？作者在[补充材料](http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf)中给出了四种变形方式：

① 考虑context【图像中context指RoI周边像素】的各向同性变形，建议框像素周围像素扩充到227×227，若遇到图像边界则用建议框像素均值填充，下图第二列； 

② 不考虑context的各向同性变形，直接用建议框像素均值填充至227×227，下图第一行第三列； 

③ 各向异性变形，简单粗暴对图像就行缩放至227×227，下图第一行第四列；

④ 变形前先进行边界像素填充【padding】处理，即向外扩展建议框边界，以上三种方法中分别采用padding=0下图第一行，padding=16下图第二行进行处理。

经过作者一系列实验表明采用padding=16的各向异性变形（anisotropically warp）即下图第二行第四列效果最好，能使mAP提升3-5%。 


3. CNN特征如何可视化？ 

文中采用了巧妙的方式将AlexNet CNN网络中Pool5层特征进行了可视化。该层的size是6×6×256，即有256种表示不同的特征，这相当于原始227×227图片中有256种195×195的感受视野【相当于对227×227的输入图像，卷积核大小为195×195，padding=4，step=8，输出大小(227-195+2×4)/8+1=6×6】； 

文中将这些特征视为”物体检测器”，输入10million的Region Proposal集合，计算每种6×6特征即“物体检测器”的激活量，之后进行非极大值抑制【下面解释】，最后展示出每种6×6特征即“物体检测器”前几个得分最高的Region Proposal，从而给出了这种6×6的特征图表示了什么纹理、结构，很有意思。

4. 为什么要进行非极大值抑制？非极大值抑制又如何操作？ 
先解释什么叫IoU。如下图所示IoU即表示(A∩B)/(A∪B) 

![IoU](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE8fecb126bf980149071098a9e658df2a/46474)
 

在测试过程完成到第4步之后，获得2000×20维矩阵表示每个建议框是某个物体类别的得分情况，此时会遇到下图所示情况，同一个车辆目标会被多个建议框包围，这时需要非极大值抑制操作去除得分较低的候选框以减少重叠框。 

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCEb3738912a4906ba9c7f09860bd12042a/46477)


具体怎么做呢？ 列（类）—> 框

① 对2000×20维矩阵中每列按从大到小进行排序； 

② 从每列（同一类）最大的得分建议框(共2000个)开始，分别与该列后面的得分建议框进行IoU计算，若IoU>阈值，则剔除得分较小的建议框（说明重叠率很大，两个建议框被视为描述一个目标，此时只保留最大得分建议框），否则（说明重叠率很小），则认为图像中存在多个同一类物体； 

③ 从每列次大的得分建议框开始，重复步骤②（这里的次大得分建议框的起始位置应该是在大的得分建议框后，或者在满足”若IoU>阈值，则剔除得分较小的建议框“里的建议框后）； 

④ 重复步骤③直到遍历完该列所有建议框； 

⑤ 遍历完2000×20维矩阵所有列(共20个)，即所有物体种类都做一遍非极大值抑制（保留了同一个目标最高的得分建议框，即剔除了同一目标（这里是指目标，对象而不是类别）满足重叠率非最大得分的建议框）； 

⑥ 最后剔除各个类别中剩余建议框得分少于该类别阈值的建议框。【文中没有讲，博主觉得有必要做】


5.为什么要采用回归器？回归器是什么有什么用？如何进行操作？ 

首先要明确目标检测不仅是要对目标进行识别，还要完成定位任务，所以最终获得的bounding-box也决定了目标检测的精度。 

这里先解释一下什么叫定位精度：定位精度可以用算法得出的物体检测框与实际标注的物体边界框的IoU值来近似表示。

如下图所示，绿色框为实际标准的卡宴车辆框，即Ground Truth；黄色框为selective search算法得出的建议框，即Region Proposal。即使黄色框中物体被分类器识别为卡宴车辆，但是由于绿色框和黄色框IoU值并不大，所以最后的目标检测精度并不高。采用回归器是为了对建议框进行校正，使得校正后的Region Proposal与selective search更接近， 以提高最终的检测精度。论文中采用bounding-box回归使mAP提高了3~4%。 

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE726759abbde88c033e443395b6499ec4/46538)

上述参考：[R-CNN论文详解](https://blog.csdn.net/WoPawn/article/details/52133338/)

注：关于Bounding box regression，可以参考下面链接中的文件

链接: https://pan.baidu.com/s/1nlbNxlQsh-FK3_SOmRHj7A 密码: f7hk


# R-CNN不足

- 重复计算。通过selective search 提取近2000左右的候选框，即今2000个ROI，RCNN对每个ROI，都跑一遍CNN，计算量很大，而且其实这些ROI之间是有overlap，显然有大量计算是重复的。（所以SPP-Net和Fast-RCNN对其进行改进）

- Multiple-stage pipeline：训练分为多个阶段，region proposals、ConvNet、SVM、BB Regression。

- 训练耗时，占有磁盘空间大。卷积出来的特征数据还要保持至本地磁盘。

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE70364415ee997831f5af38dfbe863c90/18036)


一点题外话：
当初R-CNN也不是说只用于目标检测，其实语义分割也是杠杠的

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE048b6bf273274caadecbfcf0df14d4bf/51011)


# 思考
神经网络究竟学习到了什么？

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE4ba8b5ad2d8c0ab6d7f43e4d4d2f3321/51095)

![](https://note.youdao.com/yws/public/resource/ce9042561d5534d8f5796017494744f5/xmlnote/WEBRESOURCE8b3304c5a89450306044490ec889b5ef/51097)


# R-NN原论文精简内容

结果：在VOC2012中，将mAP（mean average percision）提高了30%以上
two key insights：

1. 将CNN应用到bottom-up的region proposal中，实现定位和分割

2. 监督预训练
简单来说，将region proposals和CNN相结合，故称为R-CNN（Regions with CNN features）

## 1.引言
传统的目标检测方法是SIFT和HOG方法实现，此类方法都是blockwise orientation histograms。
1989年LeCun等人使用反向传播实现梯度下降，大大提高了CNN的训练
2012年，Hinton和他的学生在ILSVRC2012比赛中，使用AlexNet（一个含有ReLU激活和"dropout"正则化的大型CNN）获得相当高的图像分类准确率。
这引发了RBG的深思：如何将CNN在ImageNet上的分类结果应用到PASCAL VOC上的object detection上呢？（即如何将CNN的image classification与object detection建立联系）。于是RBG就开始干了，将image classification和object detection结合起来。
在介绍方法前，先讨论两个问题（也是R-CNN的贡献所在）：
(1)使用deep network来localize object；
(2)训练一个仅有少量标注的检测数据的high-capacity model
第一个问题localize object：
答：使用region proposals
与image classification不同，detection需要在一幅图像中locaizing很多目标（可能图像中只有n个目标，但要局部确定m个目标，m远远大于n，后面会筛洗）。R-CNN将localization看作regression问题（注意，只使用DNN去实现object detection，效果并不好，见2013 NISP）。
借鉴CNN的sliding-window detector方法，R-CNN也采用sliding-window approch。但典型的CNN只有两个卷积和pooling层，而R-CNN中有5层卷积层，receptive fields为195*195，strides为32*32。这也使得在sliding-window中达到准确定位成为一个难题。

于是，R-CNN通过recognition using regions（selective search）解决了CNN localization问题。如此一来，在测试时，这种方法对于输入图像会生成近2000独立"类别"区域的proposals，使用CNN从每个proposal中提取fixed-length feature vector，然后使用SVM对每个区域进行分类（即对CNN提取的feature进行分类）。R-CNN使用affine image warping从每个region proposal计算fixed-size CNN（这里不用考虑region的shape）。
下图显示了R-CNN整个方法的综述并强调一些结果：


这里停顿强调一下，R-CNN是将region proposals和CNN结合一起，故称为R-CNN：Regions with CNN features

文章中，将R-CNN和同年2014年提出的OverFeat在200-class ILSVRC2013 detection datsset上进行比较。OverFeat使用sliding-window CNN用于检测，虽然在ILSVRC2013 detection比赛时，OverFeat达到最佳性能，但R-CNN后面"显著地"超过了OverFeat的效果（论科研的生产力），

第二个问题：标记的数据是scarce（稀缺的）, 目前可用的数量不足以训练一个大型 CNN
答：先使用supervised pre-training，再使用domain-specific fine-tuning
传统的解决方法是先使用unsupervised pre-traing，再使用supervised fine-tuning；而R-CNN是先使用在ILSVRC（较大的数据集）上的supervised pre-training，再使用在PASACL（较小的数据集）进行domain-specific fine-tuning。当数据较少时，这是一个学习high-capacity CNN的有效paradigm（方法）。
fine-tuning对于detection提高了8%mAP。在fine-tuning后，R-CNN在VOC2010上达到54%mAP（以前最高是基于HOG的DPM，达到33%mAP）。

注意：R-CNN还可以用于semantic segmentation

# 2.R-CNN方法
R-CNN实现目标检测，是由3个模块组成：
(1)[Region proposals]产生category-independent region proposals，即与类别无关的region proposals（毕竟没有还用到CNN），这些proposals定义了对于检测器可用的所有候选检测集。
(2)[Feature extraction]大型的CNN，用于从上述检测到的每个区域来提取fixed-length特征向量
(3)[classification]classspecific linear SVM集合。

### Region proposals
目的是实现category-independent region proposals，但有很多方法可以实现，如objectness、selective search、CPMC等。但R-CNN对特定region propoal method是agnostic，所以使用selective search使之前的检测工作controlled comparision。

参考：https://blog.csdn.net/zijin0802034/article/details/77685438
http://caffecn.cn/?/question/160

### Feature extraction
R-CNN使用Caffe实现的CNN（5个卷积层和2个全连接层）来对每个region proposal提取4096-dimentational feature vector。为了对region proposal计算特征，我们首先必须将region中的image data转换成与CNN input fixed pixel size一致的size（确保dimension相同），如227*227 pixel size。
不管candidate region的size或者aspect ratio（长宽比），R-CNN都将tight bounding box中的all pixels warp（变换到）所需要的size。在warping之前，我们dilate（膨胀/扩大）tight bounding box，使其在warped size有确切的 p 像素扭曲图像周围的原始框（没读懂）

### Test-time detection
在测试图像上，运行select search提取近2000个region proposals。将这些proposal进行warp使之前向传播到CNN来提取feature。
然后，对于每一类，使用SVM对提取的feature进行score（评分）——其实就是对每个proposal进行分类
对图像中所有的scored regions应用greed non-maximum suppression（贪婪非极大值因抑制）来reject具有intersection-over-union（IoU）（交并比）较小的区域。

### Run-time analysis
两种特性使得检测efficient。第一，所有的CNN参数都在所有的categories中共享；第二，由CNN计算的feature vectors都是low-dimensional（

### Training
Superivised pre-training
在ILSVRC2012分类的image-level annotations上使用CNN进行pre-trained，使用Caffe CNN库进行pre-training。4

### Domain-specific fine-tuning
为了CNN适应detection和new domain（warped proposal windows），R-CNN仅使用warped region proposals来进行stochastic gradient descent（SGD随机梯度下降）从而训练CNN parameters。

### Object category classifiers
如何解决给图像中包含目标的局部区域贴上lable（注意不是给整个图像贴上label，不如成为classification）？R-CNN使用IoU（Intersection over Union）overlap threshold解决这个问题，即低于该threshold的region proposals定义为negatives，相反定义为positive。R-CNN在validation set上将overlap threshold设置为0.3。
一旦提取好feature，贴上training labels后，R-CNN对每一类使用one linear SVM来优化。

### Results
在PASCAL VOC 2010-12上进行Object Classification测试，R-CNN在VOC 2011/12测试中达到53.3%
在200-Class ILSVRC2013 detection dataset上进行object detection测试（使用与PASCAL VOC一致的hyperparameters），R-CNN达到24.3%mAP，极大地超越了第二名24.3%mAP。

