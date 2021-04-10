# Few Shot Learning based on GAN
基于GAN的小样本学习实验(pytorch)
# 实验内容
训练生成式对抗网络，利用训练得到的生成器生成样本；然后训练分类器，对测试样本进行分类；分析数据扩充前后分类模型的性能。
# 程序功能说明
Dataprocess.py：下载MNIST手写数字数据集和Fashion-MNIST数据集，将其划分为训练集（800张图片）、验证集（800张图片）和测试集（10000张图片）；  
Augment.py：AugmentPipe网络结构，用于DCGAN判别器的输入层，实现对真实图像/生成图像的扭曲性甚至破坏性的变化；  
DCGAN.py：DCGAN网络框架，包括DCGAN的训练函数、利用训练好的DCGAN生成数据及生成结果可视化、利用DCGAN对训练集进行扩充（扩充至60800张）；  
CNN.py：CNN网络框架，包括CNN的训练函数及模型准确率、F1score计算；  
Main.py：主函数  
    &emsp;&emsp;  step1：将MNIST/Fashion-MNIST数据集划分为训练集和测试集；  
    &emsp;&emsp;  step2：使用训练集训练DCGAN并保存模型，同时保存模型生成手写数字的可视化结果；  
    &emsp;&emsp;  step3：利用DCGAN进行数据生成和数据扩充；  
    &emsp;&emsp;  step4：分别用源训练集和数据增强后的训练集训练CNN；  
    &emsp;&emsp;  step5：测试上述两个模型在测试集上的分类准确性；  
# 文件内容说明
torch_utils, dnnlib：AugmentPipe底层代码，用于Augment.py；
data：MNIST手写数字数据集和Fashion-MNIST数据集；
Img：执行Main.py后存储训练过程中的相应图片，包括MNIST手写数字/Fashion-MNIST真实图像、DCGAN训练过程中每轮生成的手写数字/Fashion-MNIST图像、最优DCGAN模型生成的手写数字/Fashion-MNIST图像；  
model：执行Main.py后存储训练好的模型，包括DCGAN生成器模型(Generator.pt), 小样本数据训练的分类模型(CNN.PT), 扩充数据后训练的分类模型(CNN_reinforce.pt)；
注：可在Main.py中修改'dataflag'的值改变使用的数据集，0表示MNIST数据集，1表示Fashion-MNIST数据集。
