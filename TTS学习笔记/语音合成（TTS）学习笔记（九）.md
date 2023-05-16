# 注意力机制（九）学习笔记

------

### 前情提要：

### RNN（循环神经网络）

1.RNN如何来的？

这是一般的神经网络应该有的结构： 

<img align=“center” src="/img/sjnet.png"/>


既然我们已经有了人工神经网络和卷积神经网络，为什么还要循环神经网络？ 
原因很简单，无论是卷积神经网络，还是人工神经网络，他们的前提假设都是：元素之间是相互独立的，输入与输出也是独立的，比如猫和狗。 
但现实世界中，很多元素都是相互连接的，比如股票随时间的变化，一个人说了：我喜欢旅游，其中最喜欢的地方是云南，以后有机会一定要去______.这里填空，人应该都知道是填“云南“。因为我们是根据上下文的内容推断出来的，但机会要做到这一步就相当得难了。因此，就有了现在的循环神经网络，他的本质是：像人一样拥有记忆的能力。因此，他的输出就依赖于当前的输入和记忆。

2.RNN原理

RNN是一个序列到序列的模型，假设xt−1,xt,xt+1xt−1,xt,xt+1是一个输入：“我是中国“，那么ot−1,otot−1,ot就应该对应”是”，”中国”这两个，预测下一个词最有可能是什么？就是ot+1ot+1应该是”人”的概率比较大。

是一类以[序列](https://baike.baidu.com/item/序列/1302588)（[sequence](https://so.csdn.net/so/search?q=sequence&spm=1001.2101.3001.7020)）数据为输入，在序列的演进方向进行[递归](https://baike.baidu.com/item/递归/1740695)（recursion）且所有节点（循环单元）按链式连接的[递归神经网络](https://baike.baidu.com/item/递归神经网络/16020230)（recursive neural network） 。RNN不仅能够处理序列输入，也能够得到序列输出，这里的序列指的是向量的序列。

3.RNN建模方式

序列样本一般分为：一对多(生成图片描述)，多对一（视频解说，文本归类），多对多（语言翻译），针对不同的序列建模方式也不一样。

>**1、一对多（vector-to-sequence ）**
>
>   输入是一个单独的值，输出是一个序列。此时，有两种主要建模方式：
>
>  方式一：可只在其中的某一个序列进行计算，比如序列第一个进行输入计算：
>
<img align=“center” src="/img/jianmo01.png"/>
方式二：把输入信息X作为每个阶段的输入：

<img align=“center” src="/img/jm02.png"/>

应用场景：
1、从图像生成文字，输入为图像的特征，输出为一段句子 2、根据图像生成语音或音乐，输入为图像特征，输出为一段语音或音乐



>**2、多对一（sequence-to-vector ）**
>
>输入是一个序列，输出是一个单独的值，此时通常在最后的一个序列上进行输出变换:
>

<img align=“center” src="/img/jm3.png"/>
应用场景：1、输出一段文字，判断其所属类别 2、输入一个句子，判断其情感倾向 3、输入一段视频，判断其所属类别

>**3、多对多（Encoder-Decoder ）**
>
>**步骤一**：将输入数据编码成一个上下文向量c，这部分称为Encoder，得到c有多种方式，最简单的方法就是把Encoder的最后一个隐状态赋值给c，还可以对最后的隐状态做一个变换得到c，也可以对所有的隐状态做变换。
>
<img align=“center” src="/img/jm03.png"/>
步骤二：**用另一个RNN网络（我们将其称为Decoder）对其进行编码，方法一是将步骤一中的c作为初始状态输入到Decoder，示意图如下所示：

<img align=“center” src="/img/jm04.png"/>


## 注意力模型（self-attention）

**导入**
注意力模型借鉴了人类的视觉注意力机制，视觉注意力机制是人类视觉所特有的大脑信号处理机制。人类视觉通过快速扫描全局图像，获得需要重点关注的目标区域，也就是一般所说的注意力焦点，而后对这一区域投入更多注意力资源，以获取更多所需要关注目标的细节信息，而抑制其他无用信息。

深度学习中的注意力机制从本质上讲和人类的选择性视觉注意力机制类似，核心目标也是从众多信息中选择出对当前任务目标更关键的信息。

**Encoder-Decoder框架**
目前大多数注意力模型附着在Encoder-Decoder框架下，当然，其实注意力模型可以看作一种通用的思想，本身并不依赖于特定框架。

<img align=“center” src="/img/encode-decoder.png"/>

文本处理领域的Encoder-Decoder框架可以这么直观地去理解：可以把它看作适合处理由一个句子（或篇章）生成另外一个句子（或篇章）的通用处理模型。对于句子对<Source,Target>，我们的目标是给定输入句子Source，期待通过Encoder-Decoder框架来生成目标句子Target。Source和Target可以是同一种语言，也可以是两种不同的语言。

Encoder顾名思义就是对输入句子Source进行编码，将输入句子转化为中间语义 C

Decoder的 任务是根据句子Source的中间语义表示C和之前已经生成的历史信息生成需要的输出信息

---

## **为什么需要Attention**

了解Attention之前，首先应该了解为什么需要注意利机制。

以传统的机器翻译为例子来说明为什么我们需要Attention

传统的机器翻译，也称机器翻译(Neural machine translation)，它是由encoder和decoder两个板块组成。其中Encoder和Decoder都是一个RNN，也可以是LSTM。不熟悉RNN是如何工作的读者，请参考RNN原理。假如现在想要将‘我是一个学生。’翻译成英文‘I am a student.’，传统的机器翻译是如何操作的呢？

> 在将中文 ‘我是一个学生’ 输入到encoder之前，应首先应该使用一些embedding技术将每一个词语表示成一个向量。encoder的工作原理和RNN类似，将词向量输入到Encoder中之后，我们将最后一个hidden state的输出结果作为encoder的输出，称之为context。Context可以理解成是encoder对当前输入句子的理解。之后将context输入进decoder中，然后每一个decoder中的hidden state的输出就是decoder 所预测的当前位子的单词。从encoder到decoder的过程中，encoder中的第一个hidden state 是随机初始化的且在encoder中我们只在乎它的最后一个hidden state的输出，但是在decoder中，它的初始hidden state 是encoder的输出，且我们关心每一个decoder中的hidden state 的输出。
>
> 比如还是以 ‘ 我是一个学生。’为例，我们希望模型可以在翻译student的时候，更加的关注 ‘学生’这个词而不是其他位子的词。这种需求下，提出Attention技术。

## Attention的基本原理

**核心逻辑就是从关注全部到关注重点**

> 第一步： query 和 key 进行相似度计算，归一化处理，得到权值
>
> 第二步：将权值进行归一化，得到直接可用的权重
>
> 第三步：将权重和 value 进行加权求和

**Attention本质思想**

将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数


<img align=“center” src="/img/attenall.png"/>

​	对于具体的计算过程，可以大致抽象为以下两个过程：一是根据Q和K计算权重系数，而是根据权重系数对V进行加权求和。第一个过程又可以分为两个阶段，一是根据Q和K计算两者相似性，而是对第一阶段的求值进行归一化处理。该过程可以归纳为下图：


<img align=“center” src="/img/attention.png"/>

自注意
由上文可知，Attention机制发生在Target的元素Query和Source中的所有元素之间。而自注意（self-attention）顾名思义，指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制。具体计算过程发生了较小的变化。	

如何看待Attention机制：

将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。所以本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。

也可以将Attention机制看作一种软寻址（Soft Addressing）:Source可以看作存储器内存储的内容，元素由地址Key和值Value组成，当前有个Key=Query的查询，目的是取出存储器中对应的Value值，即Attention数值。通过Query和存储器内元素Key的地址进行相似性比较来寻址，之所以说是软寻址，指的不像一般寻址只从存储内容里面找出一条内容，而是可能从每个Key地址都会取出内容，取出内容的重要性根据Query和Key的相似性来决定，之后对Value进行加权求和，这样就可以取出最终的Value值，也即Attention值。所以不少研究人员将Attention机制看作软寻址的一种特例，这也是非常有道理的。

---

本文主要是对为什么要使用注意力机制以及注意力机制原理的简单整理总结。

---

## 相关阅读

> - [一文看懂 Attention（本质原理+3大优点+5大类型）](https://zhuanlan.zhihu.com/p/91839581)
