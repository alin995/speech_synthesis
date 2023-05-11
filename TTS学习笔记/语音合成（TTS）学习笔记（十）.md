# 语音合成学习（十）学习笔记

---
## stable diffusion算法的应用

Stable Diffusion是一个文本到图像的潜在扩散模型，由CompVis、Stability AI和LAION的研究人员和工程师创建。它使用来自LAION-5B数据库子集的512x512图像进行训练。使用这个模型，可以生成包括人脸在内的任何图像，因为有开源的预训练模型，所以我们也可以在自己的机器上运行它。

Stable Diffusion是一种机器学习模型，它经过训练可以逐步对随机高斯噪声进行去噪以获得感兴趣的样本，例如生成图像。

---

### Stable Diffusion从功能上来说主要包括两方面：

1）其核心功能为仅根据文本提示作为输入来生成的图像（text2img）；

2）你也可以用它对图像根据文字描述进行修改（即输入为文本+图像）。



### **Stable Diffusion组件**

Stable Diffusion是一个由多个组件和模型组成的系统，而非单一的模型。当我们从模型整体的角度向模型内部观察时，可以发现，其包含一个文本理解组件用于将文本信息翻译成数字表示（numeric representation），以捕捉文本中的语义信息。

我们也可以大致推测这个文本编码器是一个特殊的Transformer语言模型（具体来说是CLIP模型的文本编码器）。模型的输入为一个文本字符串，输出为一个数字列表，用来表征文本中的每个单词/token，即将每个token转换为一个向量。然后这些信息会被提交到图像生成器（image generator）中，它的内部也包含多个组件。



### **图像生成器主要包括两个阶段：**

**1. Image information creator**这个组件是Stable Diffusion的独家秘方，相比之前的模型，它的很多性能增益都是在这里实现的。该组件运行多个steps来生成图像信息，其中steps也是Stable Diffusion接口和库中的参数，通常默认为50或100。

图像信息创建器完全在图像信息空间（或潜空间）中运行，这一特性使得它比其他在像素空间工作的Diffusion模型运行得更快；从技术上来看，该组件由一个UNet神经网络和一个调度（scheduling）算法组成。扩散（diffusion）这个词描述了在该组件内部运行期间发生的事情，即对信息进行一步步地处理，并最终由下一个组件（图像解码器）生成高质量的图像。

**2. 图像解码器**图像解码器根据从图像信息创建器中获取的信息画出一幅画，整个过程只运行一次即可生成最终的像素图像。

<img aligin="center" src="/img/three.png" />

Stable Diffusion总共包含三个主要的组件，其中每个组件都拥有一个独立的神经网络：

1）Clip Text用于文本编码。输入：文本输出：77个token嵌入向量，其中每个向量包含768个维度

2）UNet + Scheduler在信息（潜）空间中逐步处理/扩散信息。输入：文本嵌入和一个由噪声组成的初始多维数组（结构化的数字列表，也叫张量tensor）。输出：一个经过处理的信息阵列

3）自编码解码器（Autoencoder Decoder），使用处理过的信息矩阵绘制最终图像的解码器。



##  Stable Diffusion文字生成图片过程

​	Stable Diffusion其实是Diffusion的改进版本，主要是为了解决Diffusion的速度问题。那么Stable Diffusion是如何根据文字得出图片的呢？下图是Stable Diffusion生成图片的具体过程：

<img aligin="center" src="/img/stablediffusion.png" />

可以看到，对于输入的文字（图中的“An astronout riding a horse”）会经过一个CLIP模型转化为text embedding，然后和初始图像（初始化使用随机高斯噪声Gaussian Noise）一起输入去噪模块（也就是图中Text conditioned latent U-Net），最后输出 512×512 大小的图片。这里面Text conditioned latent U-net**，翻译过来就是**文本条件隐U-net网络**，其实是**通过对U-Net引入多头Attention机制，使得输入文本和图像相关联**。

文本编码器将把输入文字提示转换为U-Net可以理解的嵌入空间，这是一个简单的基于transformer的编码器，它将标记序列映射到潜在文本嵌入序列。从这里可以看到使用良好的文字提示以获得更好的预期输出。



##  Stable Diffusion的改进：图像压缩

Stable Diffusion原来的名字叫“**Latent Diffusion Model**”（**LDM**），很明显就是扩散过程发生隐空间中（latent space），其实就是对图片做了压缩，这也是Stable Diffusion比Diffusion速度快的原因。

<img aligin="center" src="/img/autocoder.png" />

Stable Diffusion会先训练一个自编码器，来学习将图像压缩成低维表示。

- 通过训练好的编码器 E，可以将原始大小的图像压缩成低维的latent data（图像压缩）
- 通过训练好的解码器 D，可以将latent data还原为原始大小的图像

---

## **怎么玩Stable Diffusion ？**

`stable-diffusion-webui` 的功能很多，主要有如下 2 个：

- Text-to-Image  文生图（text2img）：根据提示词（Prompt）的描述生成相应的图片， 是 Stable Diffusion 依据文字描述来生成图像。
- Image-to-Image 图生图（img2img）：将一张图片根据提示词（Prompt）描述的特点生成另一张新的图片。



在开始使用文生图之前，有必要了解以下几个参数的含义：

| 参数            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| Prompt          | 提示词（正向）                                               |
| Negative prompt | 消极的提示词（反向）                                         |
| Width & Height  | 要生成的图片尺寸。尺寸越大，越耗性能，耗时越久。             |
| CFG scale       | AI 对描述参数（Prompt）的倾向程度。值越小生成的图片越偏离你的描述，但越符合逻辑；值越大则生成的图片越符合你的描述，但可能不符合逻辑。 |
| Sampling method | 采样方法。有很多种，但只是采样算法上有差别，没有好坏之分，选用适合的即可。 |
| Sampling steps  | 采样步长。太小的话采样的随机性会很高，太大的话采样的效率会很低，拒绝概率高(可以理解为没有采样到,采样的结果被舍弃了)。 |
| Seed            | 随机数种子。生成每张图片时的随机种子，这个种子是用来作为确定扩散初始状态的基础。不懂的话，用随机的即可。 |



​     接下来我们来生成一张骑骆驼的宇航员图片，配置以下参数后，点击 "生成" 即可：

> 注：提示词（Prompt）越多，AI 绘图结果会更加精准，另外，目前中文提示词的效果不好，还得使用英文提示词。

<img aligin="center" src="/img/horse.png" />

他提供的模型还包含了一些可用的高级选项来改变生成的图像的质量，如下图所示:

<img aligin="center" src="/img/useimg.png" />

**images**:该选项控制的生成图像数量最多为4个。

**Steps**:此选项选择想要的扩散过程的步骤数。步骤越多，生成的图像质量越好。如果想要高质量，可以选择可用的最大步骤数，即50。如果你想要更快的结果，那么考虑减少步骤的数量。

**Guidance Scale**:Guidance Scale是生成的图像与输入提示的紧密程度与输入的多样性之间的权衡。它的典型值在7.5左右。增加的比例越多，图像的质量就会越高，但是你得到的输出就会越少。

**Seed**:随机种子够控制生成的样本的多样性

> 使用Diffuser 包
>
> 第二种使用的方法是使用Hugging Face的Diffusers库，它包含了目前可用的大部分稳定扩散模型，我们可以直接在谷歌的Colab上运行它。
>
> 
>
> 预训练的模型包括建立一个完整的管道所需的所有组件。它们存放在以下文件夹中:
>
> **text_encoder**:Stable Diffusion使用CLIP，但其他扩散模型可能使用其他编码器，如BERT。
>
> **tokenizer**:它必须与text_encoder模型使用的标记器匹配。
>
> **scheduler**:用于在训练过程中逐步向图像添加噪声的scheduler算法。
>
> **U-Net**:用于生成输入的潜在表示的模型。
>
> **VAE**，我们将使用它将潜在的表示解码为真实的图像。
>
> 可以通过引用组件被保存的文件夹，使用from_pretraining的子文件夹参数来加载组件。

**也可以根据自己的需求下载对应的模型， 模型下载安装后输入自定义prompt，也就是任意你想生成的图像内容，然后点击生成就好了。如果不满意，可以再次点击，每次将随机生成不同的图片，总有一些你感兴趣的。**

> 注：模型文件有 2 种格式，分别是 `.ckpt`（Model PickleTensor） 和 `.safetensors`（Model SafeTensor），据说 `.safetensors` 更安全，这两种格式 `stable-diffusion-webui` 都支持，随意下载一种即可。

将下载好的模型文件放到 `stable-diffusion-webui\models\Stable-diffusion` 目录下

---

**模型版本特性总结**

- stable-diffusion-v1-4
- 擅长绘制风景类画，整体偏欧美风，具有划时代意义；
- stable-diffusion-v1-5
- 同上，但生成的作品更具艺术性；
- stable-diffusion-2
- 图像生成质量大幅提升，原生支持768x768等；
- waifu-diffusion
- 设定随机种子后，每次将生成相同的图像，无随机性，可方便复现；
- Taiyi-Stable-Diffusion-1B-Chinese-v0.1
- 擅长中文古诗词绘画，整体绘画风格更偏中国风；
- Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1
- 同上，但额外支持英文输入；
- Stable_Diffusion_PaperCut_Model
- 擅长剪纸画；
- trinart_characters_19.2m_stable_diffusion_v1
- 擅长动漫角色绘制；
- trinart_derrida_characters_v2_stable_diffusion
- 擅长动漫角色绘制，出图效果更稳定。
- 更多第三方模型请参考其他文章，本文不多做介绍。

---

## 本地部署 ##

> - 部署python环境 python3.0 以及 cuda
> - 下载Stable Diffusion项目 
> - 安装Stable Diffusion环境 
> - 下载模型 https://huggingface.co/CompVis  （选择类型为CHECKPOINT的就是大模型了，点击进去可以下载）   **需要注意模型的路径**
> - 运行
>
> （ prompt：表示你的想法，你想要生成一副什么样的图片，包含主体、风格、色彩、质量要求等等    negative prompt：表示你不想要什么，比如不想要图片出现什么，不想图片质量差，不想人物模糊或者多手多脚等。）

---
**本文主要介绍什么是Stable Diffusion，并讨论它的主要组成部分。然后我们将使用模型不同的方式创建图像

---

## 相关资源

> **官网**：[http://ommer-lab.com/research/](https://link.zhihu.com/?target=http%3A//ommer-lab.com/research/)
>
> **论文**：[http://arxiv.org/abs/2112.1075](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/2112.1075)
>
> **Github地址**：[http://github.com/CompVis/stab](https://link.zhihu.com/?target=http%3A//github.com/CompVis/stab)
>
> **模型下载地址：**
>
> - [http://huggingface.co/CompVis/](https://link.zhihu.com/?target=http%3A//huggingface.co/CompVis/)
>
> - [http://huggingface.co/CompVis/](https://link.zhihu.com/?target=http%3A//huggingface.co/CompVis/)
>
> **整合包+模型下载**
>
> [AI绘画软件Stable Diffusion+WebUI+Chilloutmix/ControlNet模型（一键安装包）支持WIN/MAC](https://link.zhihu.com/?target=https%3A//muhou.net/235195.html)
